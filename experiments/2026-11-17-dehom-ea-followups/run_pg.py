"""Trial 14, pilot C: problem-grain (PG) mutation, LLM vs random operator (DESIGN.md C).

  python -u run_pg.py --obj stiff|buck --op RAND|LLM --seed S --gens G --workers W [--out DIR]

Base loop = run_arm.py's PCA arm, unchanged: same gen 0 (inv5 encoding, trial-13 PCA, nominal
evaluation), NSGA-II survival of 100 elites, 100 BLX-0.25 children per generation in whitened PCA
space (tournament parents, N(0, 0.1^2) noise), same objectives / constraint / HV normaliser. The
RNG draws for crossover and for the choice of mutation parents are made in the same order as
run_arm's PCA arm, so generation 1's crossover children and mutation parents are identical to
`run_arm.py --arm PCA` at the same seed (later generations diverge).

The 10 image mutants are replaced by 10 PG mutants (pg_ops.py): a parent drawn uniformly without
replacement from the elite set P (as run_arm's mutants) has its LF problem edited (RAND or LLM
operator), the LF optimiser is re-run on the edited problem (lf_passive.lf_optimise_passive with
lf.GEN0_SETTINGS, in the worker pool), and the result is encoded and PCA-projected like every child.

LF problem record per archive design (needed because only gen-0 designs come from an LF problem):
  gen 0          : V0, Lmin from results/gen0/lf_designs.npz; rmin 2, jitters 0, seed 0, no boxes
                   (lf.GEN0_SETTINGS; the record reproduces the gen-0 LF design exactly).
  crossover child: INHERITS the record of the closer of its two parents (Euclidean distance in
                   whitened PCA space). Choice: the BLX child is a blend with no LF problem of its
                   own; the closer parent's problem is the best available description. Consequence:
                   a PG edit of a crossover child starts from its closer parent's LF optimum, not
                   from the blend itself (logged: parent id and the parameter-source id).
  PG child       : the edited record.

Outputs (results/pg/<obj>_<op>_s<seed>/ unless --out):
  history.jsonl   one line per generation: HV (as run_arm), entry counts / rates per operator, LLM
                  latency / fallbacks, LF time, PCA retention.
  children.jsonl  one line per child (PRIMARY METRIC): op (cross | pg), in_P (entered the elite set
                  in its generation), on_front (on the archive's non-dominated front as of the last
                  rewrite; rewritten every generation, so after the last one = final front),
                  parents, param source, edited parameters, LLM rationale (truncated), latency,
                  tokens, cost, fallback flag/reason, pca_loss = |proj - raw| / |raw - parent| and
                  pca_kept = |proj - parent| / |raw - parent| in feature space (raw = the LF
                  optimiser's design, proj = its PCA projection, parent = the parent's evaluated
                  (projected) features; pca_kept is run_arm's mut_kept_frac definition).
  llm_calls.jsonl full prompt text + reply per LLM call; prompts_g1/ the first generation's pictures.
  archive.npz     as run_arm's plus params (JSON per design), parents, param_src.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea  # noqa: E402
import run_arm as ra  # noqa: E402
import pg_ops  # noqa: E402

N_ELITE, N_CHILD, N_PG = ra.N_ELITE, ra.N_CHILD, 10
ENC = "inv5"


def crossover(rng, Z, rank, crowd, n, noise=0.1):
    """= run_arm.latent_children(mode='pca') (same RNG draws, same order) + the parent indices."""
    a = ea.tournament(rng, rank, crowd, n)
    b = ea.tournament(rng, rank, crowd, n)
    t = rng.uniform(-0.25, 1.25, (n, 1))
    Zc = (1 - t) * Z[a] + t * Z[b] + rng.normal(size=(n, Z.shape[1])) * noise
    return Zc, a, b


def vf_map_text(obj):
    """Gen-0 relation V0 -> HF vf (linear fit), quoted in the LLM prompt."""
    lfd = np.load(ra.RES / "gen0" / "lf_designs.npz")
    _, rec = ra.load_gen0(obj)
    vf = np.array([r["vf"] for r in rec]); ok = np.array([r["connected"] for r in rec])
    b, a = np.polyfit(lfd["V0"][ok], vf[ok], 1)
    return ", ".join(f"V0 {v:.2f} -> HF vf ~{a + b * v:.3f}" for v in (0.25, 0.35, 0.50))


def front_points(idx, F, CV, recs, k=10):
    feas = idx[CV[idx] <= 0]
    if not len(feas):
        return []
    fr = feas[ea.nondominated_sort(F[feas]) == 0]
    fr = fr[np.argsort(F[fr, 0])]
    if len(fr) > k:
        fr = fr[np.unique(np.linspace(0, len(fr) - 1, k).round().astype(int))]
    return [(float(recs[i]["vf"]), float(recs[i]["compliance"]), float(recs[i]["blf"])) for i in fr]


def dominates(fa, fb):
    return bool(np.all(fa <= fb) and np.any(fa < fb))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run(args):
    buck = args.obj == "buck"
    rng = np.random.default_rng(args.seed)                    # crossover + parent choice (as run_arm)
    rng_edit = np.random.default_rng([args.seed, 777])        # RAND edits and LLM fallbacks
    pca = ra.load_pca(ENC)
    Z0, rec0 = ra.load_gen0(args.obj, ENC)
    cref = ra.cref_for(args.obj)
    F0, cv0 = ra.objectives(rec0, args.obj, cref)
    norm = ra.shared_normaliser(args.obj)
    lfd = np.load(ra.RES / "gen0" / "lf_designs.npz")
    out = Path(args.out) if args.out else ra.RES / "pg" / f"{args.obj}_{args.op}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args)))
    for f in ("history.jsonl", "children.jsonl", "llm_calls.jsonl"):
        (out / f).unlink(missing_ok=True)
    vfmap = vf_map_text(args.obj) if args.op == "LLM" else None

    AZ, AF, ACV, AG, AOP = [Z0], [F0], [cv0], [np.zeros(len(Z0), int)], [np.array(["gen0"] * len(Z0))]
    AREC = list(rec0)
    PAR = [pg_ops.gen0_params(v, L) for v, L in zip(lfd["V0"], lfd["Lmin"])]
    PARENTS = [[] for _ in range(len(Z0))]; PSRC = list(range(len(Z0)))
    CH = []                                                   # child records
    P, _, _ = ea.nsga2_select(F0, N_ELITE, cv0)

    def archive():
        return np.concatenate(AZ), np.concatenate(AF), np.concatenate(ACV)

    def log_gen(g, extra):
        Z, F, CV = archive()
        Fn = norm(F[CV <= 0])
        h = dict(gen=g, n_eval=len(AREC), n_designs=len(F), hv11=ea.hv2d(Fn, np.array([1.1, 1.1])),
                 hv20=ea.hv2d(Fn, np.array([2.0, 2.0])),
                 pop_div=float(np.mean(np.linalg.norm(Z[P][:, None] - Z[P][None], axis=-1))), **extra)
        h = _clean(h)
        print(json.dumps(h), flush=True)
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(h) + "\n")

    def write_children():
        Z, F, CV = archive()
        feas = np.where(CV <= 0)[0]
        front = set(feas[ea.nondominated_sort(F[feas]) == 0].tolist())
        Pset = set(P.tolist())
        with open(out / "children.jsonl", "w") as f:
            for c in CH:
                c["on_front"] = c["idx"] in front; c["in_P_now"] = c["idx"] in Pset
                f.write(json.dumps(_clean(c)) + "\n")

    def save():
        Z, F, CV = archive()
        np.savez_compressed(out / "archive.npz", Z=Z, F=F, CV=CV, gen=np.concatenate(AG), op=np.concatenate(AOP), P=P,
                            compliance=np.array([r["compliance"] for r in AREC]), blf=np.array([r["blf"] for r in AREC]),
                            vf=np.array([r["vf"] for r in AREC]), params=np.array([json.dumps(p) for p in PAR]),
                            parents=np.array([json.dumps(p) for p in PARENTS]), param_src=np.array(PSRC))

    log_gen(0, {})
    with ra.make_pool(args.workers) as pool, ThreadPoolExecutor(N_PG) as tpool:
        for g in range(1, args.gens + 1):
            t0 = time.time()
            Z, F, CV = archive()
            _, rk, cr = ea.nsga2_select(F[P], len(P), CV[P])
            Zx, a, b = crossover(rng, Z[P], rk, cr, N_CHILD)
            ia, ib = P[a], P[b]
            da = np.linalg.norm(Zx - Z[ia], axis=1); db = np.linalg.norm(Zx - Z[ib], axis=1)
            psrc_x = np.where(da <= db, ia, ib)
            pg_par = P[rng.choice(len(P), N_PG, replace=False)]
            Fpar = pca.inverse(Z[pg_par])
            # pool queue: renders (LLM only) first, then the 100 crossover evaluations
            rend_async = pool.map_async(pg_ops.render_task, [(f, buck) for f in Fpar], chunksize=1) \
                if args.op == "LLM" else None
            cross_async = pool.map_async(ra._evaluate, [(f, ENC, buck, [None]) for f in pca.inverse(Zx)], chunksize=1)

            # ---- PG edits
            t_e = time.time(); info = {}
            erngs = [np.random.default_rng(s) for s in rng_edit.integers(0, 2**63, N_PG)]
            if args.op == "RAND":
                edits = [pg_ops.rand_edit(r, PAR[i]) for r, i in zip(erngs, pg_par)]
                for e in edits:
                    e[2]["fallback"] = False
                prompts = [None] * N_PG
            else:
                rends = rend_async.get()
                info["t_render"] = time.time() - t_e
                front = front_points(P, F, CV, AREC)
                pngs, prompts = [], []
                for i, rd in zip(pg_par, rends):
                    nums = dict(vf=AREC[i]["vf"], compliance=AREC[i]["compliance"], blf=AREC[i]["blf"],
                                n_frag=rd["n_frag"], frag_area=rd["frag_area"])
                    pngs.append(pg_ops.render_png(rd["s"], rd["removed"], rd["mode"]))
                    prompts.append(pg_ops.build_prompt(args.obj, PAR[i], nums, front, vfmap))
                if g == 1:
                    (out / "prompts_g1").mkdir(exist_ok=True)
                    for j, (png, pr) in enumerate(zip(pngs, prompts)):
                        (out / "prompts_g1" / f"m{j}_p{pg_par[j]}.png").write_bytes(png)
                        (out / "prompts_g1" / f"m{j}_p{pg_par[j]}.txt").write_text(pr)
                futs = [tpool.submit(pg_ops.llm_edit, r, PAR[i], png, pr, args.model, args.llm_timeout)
                        for r, i, png, pr in zip(erngs, pg_par, pngs, prompts)]
                edits = [f.result() for f in futs]
            info["t_edit"] = time.time() - t_e

            # ---- LF re-optimisation of the edited problems (pool), encode, project
            t_l = time.time()
            lfr = pool.map(pg_ops.lf_task, [e[0] for e in edits], chunksize=1)
            Fraw = ra.features(np.stack([r["w1"] for r in lfr]), np.stack([r["w2"] for r in lfr]),
                               np.stack([r["theta"] for r in lfr]), ENC)
            Zm = pca.transform(Fraw)
            Fproj = pca.inverse(Zm)
            dn = np.linalg.norm((Fraw - Fpar).reshape(N_PG, -1), axis=1)
            loss = np.linalg.norm((Fproj - Fraw).reshape(N_PG, -1), axis=1) / np.maximum(dn, 1e-12)
            kept = np.linalg.norm((Fproj - Fpar).reshape(N_PG, -1), axis=1) / np.maximum(dn, 1e-12)
            rec_m = [ra.aggregate(r) for r in pool.map(ra._evaluate, [(f, ENC, buck, [None]) for f in Fproj], chunksize=1)]
            info["t_lf_eval"] = time.time() - t_l
            rec_x = [ra.aggregate(r) for r in cross_async.get()]

            # ---- archive + survival (as run_arm PCA arm)
            Zc = np.concatenate([Zx, Zm]); recs = rec_x + rec_m
            Fc, cvc = ra.objectives(recs, args.obj, cref)
            idx0 = len(F)
            AZ.append(Zc); AF.append(Fc); ACV.append(cvc); AG.append(np.full(len(Zc), g))
            AOP.append(np.array(["cross"] * N_CHILD + ["pg"] * N_PG)); AREC += recs
            PAR += [PAR[s] for s in psrc_x] + [e[0] for e in edits]
            PARENTS += [[int(x), int(y)] for x, y in zip(ia, ib)] + [[int(i)] for i in pg_par]
            PSRC += [int(s) for s in psrc_x] + [-1] * N_PG
            cand = np.concatenate([P, idx0 + np.arange(len(Zc))])
            Z, F, CV = archive()
            s, _, _ = ea.nsga2_select(F[cand], N_ELITE, CV[cand]); P = cand[s]
            Pset = set(P.tolist())

            # ---- per-child records
            for j in range(N_CHILD):
                k = idx0 + j
                CH.append(dict(gen=g, idx=k, op="cross", in_P=k in Pset, parents=[int(ia[j]), int(ib[j])],
                               param_src=int(psrc_x[j]), vf=recs[j]["vf"], compliance=recs[j]["compliance"],
                               blf=recs[j]["blf"], feasible=bool(cvc[j] <= 0)))
            lat, nfb = [], 0
            for j, ((cp, changed, ei), r) in enumerate(zip(edits, lfr)):
                k = idx0 + N_CHILD + j; jj = N_CHILD + j; i = int(pg_par[j])
                d = dict(gen=g, idx=k, op="pg", operator=args.op, in_P=k in Pset, parent=i,
                         parent_param_src=PSRC[i], edited=changed, params=cp,
                         lf_compliance=r["compliance_lf"], lf_vol=r["vol_lf"], lf_iters=r["n_iter"],
                         lf_wall_s=r["wall_s"], n_box_cells=[r["n_solid_box"], r["n_void_box"]],
                         pca_loss=float(loss[j]), pca_kept=float(kept[j]),
                         vf=recs[jj]["vf"], compliance=recs[jj]["compliance"], blf=recs[jj]["blf"],
                         feasible=bool(cvc[jj] <= 0),
                         dominates_parent=bool(cvc[jj] <= 0 and CV[i] <= 0 and dominates(Fc[jj], F[i])),
                         fallback=bool(ei.get("fallback")), fallback_reason=ei.get("fallback_reason"),
                         notes=ei.get("notes"), rand_types=ei.get("rand_types"))
                if args.op == "LLM":
                    d.update(rationale=(ei.get("rationale") or "")[:400], latency_s=ei.get("latency_s"),
                             usage=ei.get("usage"), cost_usd=ei.get("cost_usd"))
                    lat.append(ei.get("latency_s") or np.nan); nfb += bool(ei.get("fallback"))
                    with open(out / "llm_calls.jsonl", "a") as f:
                        f.write(json.dumps(_clean(dict(gen=g, idx=k, parent=i, prompt=prompts[j], reply=ei.get("reply"),
                                                       raw_edit=ei.get("raw_edit"), fallback=ei.get("fallback"),
                                                       fallback_reason=ei.get("fallback_reason"),
                                                       latency_s=ei.get("latency_s"), usage=ei.get("usage")))) + "\n")
                CH.append(d)
            gen_in = np.array([idx0 + j in Pset for j in range(len(Zc))])
            info.update(t_gen=time.time() - t0, frac_conn=float(np.mean([r["connected"] for r in recs])),
                        mean_eval_s=float(np.mean([r["t"] for r in recs])),
                        n_cross_in=int(gen_in[:N_CHILD].sum()), n_pg_in=int(gen_in[N_CHILD:].sum()),
                        rate_cross=float(gen_in[:N_CHILD].mean()), rate_pg=float(gen_in[N_CHILD:].mean()),
                        frac_pg_in_P=float(np.mean(np.concatenate(AOP)[P] == "pg")),
                        lf_wall_mean=float(np.mean([r["wall_s"] for r in lfr])),
                        pca_kept_mean=float(np.mean(kept)), pca_loss_mean=float(np.mean(loss)))
            if args.op == "LLM":
                info.update(llm_lat_mean=float(np.nanmean(lat)), llm_lat_max=float(np.nanmax(lat)), n_fallback=nfb)
            log_gen(g, info)
            write_children()
            if g % 10 == 0 or g == args.gens:
                save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", choices=["stiff", "buck"], required=True)
    ap.add_argument("--op", choices=["RAND", "LLM"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--llm-timeout", type=float, default=120.0)
    ap.add_argument("--out", default=None, help="output dir (default results/pg/<obj>_<op>_s<seed>)")
    run(ap.parse_args())
