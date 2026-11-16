"""Run one arm of trial 13 (evolutionary de-homogenisation, Xu et al. 2026).

  python -u run_arm.py --obj stiff|buck --arm FULL|NOMUT|NOVAE|NOSEL --seed S --gens G --workers W
  python -u run_arm.py --obj stiff|buck --gen0-only          # shared gen-0 evaluation cache

Arms (identical HF budget: 110 evaluations per generation):
  FULL   NSGA-II selection + VAE latent crossover (100) + image mutation (10)   [paper]
  NOMUT  NSGA-II + VAE crossover (110), no mutation                              [paper Case 2]
  NOVAE  NSGA-II + the same BLX crossover in whitened PCA space (100) + mutation (10)
  NOSEL  FULL's operators, no selection pressure: uniform parents, random survivors
MORELF (more LF designs at matched budget) is run by run_morelf.py.

Objectives (minimised): stiff = (vf, compliance); buck = (vf, -BLF) subject to
compliance <= gen-0 front compliance at the same vf (paper Sec. 4.1.3).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea  # noqa: E402
import hf  # noqa: E402

RES = HERE / "results"
N_ELITE, N_CHILD, N_MUT = 100, 100, 10
BAD_CV = 1e3

MASKS = hf.coarse_masks()
DES = MASKS["design"]
SOLID_C, VOID_C = MASKS["solid_fixed"], MASKS["void_fixed"]


# ---------------------------------------------------------------------------- fields
def features(w1, w2, th):
    """(n, 20, 40) fields -> (n, Nd, 5) encoded design-element features."""
    return ea.encode(w1, w2, th)[:, DES, :]


def fields_from_features(Fd):
    """(Nd, 5) -> w1, w2, theta images (20, 40) with fixed non-design values."""
    img = np.zeros(DES.shape + (ea.NCH,))
    img[DES] = Fd
    w1, w2, th = ea.decode(img)
    w1[SOLID_C], w2[SOLID_C] = 1.0, 1.0
    w1[VOID_C], w2[VOID_C] = 0.0, 0.0
    return w1, w2, th


def image_from_features(Fd):
    img = np.zeros(DES.shape + (ea.NCH,))
    img[DES] = Fd
    w1, w2, th = fields_from_features(Fd)
    img[~DES] = ea.encode(w1, w2, th)[~DES]
    return img


# ---------------------------------------------------------------------------- workers
_W = {}


def _winit():
    import dehom  # noqa
    _W["dehom"] = dehom
    _W["prob"] = hf.Problem(hf.GIRDER, hf.H_FINE)


def _evaluate(task):
    Fd, buck, phase_seed = task
    t0 = time.time()
    w1, w2, th = fields_from_features(Fd)
    d = _W["dehom"].dehomogenise(w1, w2, th, phase_seed=phase_seed)
    r = _W["prob"].evaluate(d["phi"], buckling=buck, nev=1)
    return dict(vf=r["vf"], connected=r["connected"], compliance=r["compliance"], umax=r["umax"],
                blf=r.get("blf", np.nan), t=time.time() - t0)


def evaluate_many(pool, Fds, buck, phase_seeds=None):
    phase_seeds = phase_seeds if phase_seeds is not None else [None] * len(Fds)
    return pool.map(_evaluate, [(F, buck, s) for F, s in zip(Fds, phase_seeds)], chunksize=1)


def make_pool(workers):
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(workers, initializer=_winit)


# ---------------------------------------------------------------------------- objectives
def objectives(recs, obj, cref=None):
    vf = np.array([r["vf"] for r in recs]); C = np.array([r["compliance"] for r in recs])
    ok = np.array([r["connected"] and np.isfinite(r["compliance"]) for r in recs])
    if obj == "stiff":
        F = np.stack([vf, np.where(ok, C, 1e30)], 1)
        cv = np.where(ok, 0.0, BAD_CV)
    else:
        blf = np.array([r["blf"] for r in recs])
        F = np.stack([vf, np.where(ok, -blf, 1e30)], 1)
        cv = np.where(ok, np.maximum(0.0, C / cref(vf) - 1.0), BAD_CV)
    return F, cv


def compliance_reference(vf0, C0):
    """Lower envelope of gen-0 compliance vs vf (non-dominated set), interpolated
    in log C; flat extrapolation. Paper: 'stiffness not lower than that of the
    initial design with the same volume'; envelope + interpolation = OURS."""
    F = np.stack([vf0, C0], 1)
    r = ea.nondominated_sort(F)
    P = F[r == 0]; P = P[np.argsort(P[:, 0])]
    lv, lc = P[:, 0], np.log(P[:, 1])
    return lambda v: np.exp(np.interp(v, lv, lc))


# ---------------------------------------------------------------------------- gen 0
def gen0(obj, workers):
    out = RES / "gen0" / f"eval_{obj}.npz"
    lf = np.load(RES / "gen0" / "lf_designs.npz")
    F0 = features(lf["w1"], lf["w2"], lf["theta"])
    pca_path = RES / "gen0" / "pca.npz"
    pca = ea.ChannelPCA().fit(F0)
    np.savez(pca_path, **{f"Phi{c}": pca.Phi[c] for c in range(ea.NCH)}, **{f"ev{c}": pca.ev[c] for c in range(ea.NCH)},
             mean=pca.mean, sd=pca.sd, nc=np.array(pca.nc))
    Z0 = pca.transform(F0)
    F0r = pca.inverse(Z0)
    buck = obj == "buck"
    with make_pool(workers) as pool:
        t = time.time()
        rec_raw = evaluate_many(pool, list(F0), buck)
        rec_rec = evaluate_many(pool, list(F0r), buck)
        dt = time.time() - t
    np.savez(out, Z0=Z0, **{f"raw_{k}": np.array([r[k] for r in rec_raw]) for k in rec_raw[0]},
             **{f"rec_{k}": np.array([r[k] for r in rec_rec]) for k in rec_rec[0]})
    info = dict(obj=obj, n=len(F0), pca_nc=pca.nc, pca_dim=pca.dim, pca_recon_err=pca.recon_error(F0),
                wall_s=dt, mean_eval_s=float(np.mean([r["t"] for r in rec_rec])))
    (RES / "gen0" / f"eval_{obj}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def load_pca():
    d = np.load(RES / "gen0" / "pca.npz")
    p = ea.ChannelPCA()
    p.Phi = [d[f"Phi{c}"] for c in range(ea.NCH)]; p.ev = [d[f"ev{c}"] for c in range(ea.NCH)]
    p.mean, p.sd, p.nc = d["mean"], d["sd"], [int(x) for x in d["nc"]]
    return p


def load_gen0(obj):
    d = np.load(RES / "gen0" / f"eval_{obj}.npz")
    recs = [dict(vf=d["rec_vf"][i], connected=bool(d["rec_connected"][i]), compliance=d["rec_compliance"][i],
                 umax=d["rec_umax"][i], blf=d["rec_blf"][i]) for i in range(len(d["Z0"]))]
    return d["Z0"], recs


# ---------------------------------------------------------------------------- operators
def latent_children(rng, Z, rank, crowd, n, vae_seed, use_vae=True, noise=0.1):
    """BLX-0.25 blend of two tournament-selected parents (+ noise), in VAE latent
    space (paper: VAE crossover; operator = OURS) or directly in PCA space (NOVAE)."""
    a = tournament_or_uniform(rng, rank, crowd, n); b = tournament_or_uniform(rng, rank, crowd, n)
    t = rng.uniform(-0.25, 1.25, (n, 1))
    if use_vae:
        vae = ea.VAE(Z.shape[1], seed=vae_seed).fit(Z)
        mu, sd = vae.encode(Z)
        L = (1 - t) * mu[a] + t * mu[b] + rng.normal(size=(n, mu.shape[1])) * sd.mean(0)
        return vae.decode(L), dict(vae_loss=vae.final_loss, vae_sigma=float(sd.mean()))
    Zc = (1 - t) * Z[a] + t * Z[b] + rng.normal(size=(n, Z.shape[1])) * noise
    return Zc, {}


def tournament_or_uniform(rng, rank, crowd, n):
    if rank is None:
        return rng.integers(0, len(crowd), n)
    return ea.tournament(rng, rank, crowd, n)


def mutants(rng, pca, Zpar, n):
    import lf
    out, kept = [], []
    for i in rng.choice(len(Zpar), n, replace=False):
        Fd = pca.inverse(Zpar[i])
        w1, w2, th = fields_from_features(Fd)
        ux, uy = lf.lf_displacement(w1, w2, th)
        img = image_from_features(Fd)
        m = ea.mutate_image(img, np.asarray(ux), np.asarray(uy), rng)
        z = pca.transform(m[DES][None])[0]
        dz = np.linalg.norm(z - Zpar[i])
        dimg = np.linalg.norm(m[DES] - Fd)
        dimg_proj = np.linalg.norm(pca.inverse(z) - Fd)
        out.append(z); kept.append(dimg_proj / max(dimg, 1e-12))
    return np.array(out), dict(mut_kept_frac=float(np.mean(kept)))


# ---------------------------------------------------------------------------- run
def run(args):
    rng = np.random.default_rng(args.seed)
    pca = load_pca()
    Z0, rec0 = load_gen0(args.obj)
    cref = None
    if args.obj == "buck":
        d = np.load(RES / "gen0" / "eval_buck.npz")
        ok = d["rec_connected"].astype(bool)
        cref = compliance_reference(d["rec_vf"][ok], d["rec_compliance"][ok])
    F0, cv0 = objectives(rec0, args.obj, cref)
    feas0 = cv0 <= 0
    norm = ea.Normaliser(F0[feas0])
    out = RES / "arms" / f"{args.obj}_{args.arm}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args)))

    AZ, AF, ACV, AG, AOP, AREC = [Z0], [F0], [cv0], [np.zeros(len(Z0), int)], [np.array(["gen0"] * len(Z0))], list(rec0)
    P, rank, crowd = None, None, None
    sel, rk, cr = ea.nsga2_select(F0, N_ELITE, cv0)
    P = sel
    hist = []

    def archive():
        return np.concatenate(AZ), np.concatenate(AF), np.concatenate(ACV)

    def log_gen(g, extra):
        Z, F, CV = archive()
        Fn = norm(F[CV <= 0])
        h = dict(gen=g, n_eval=len(F), hv11=ea.hv2d(Fn, np.array([1.1, 1.1])), hv20=ea.hv2d(Fn, np.array([2.0, 2.0])),
                 pop_div=float(np.mean(np.linalg.norm(Z[P][:, None] - Z[P][None], axis=-1))), **extra)
        hist.append(h)
        print(json.dumps(h), flush=True)
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(h) + "\n")

    if (out / "history.jsonl").exists():
        (out / "history.jsonl").unlink()
    log_gen(0, {})
    with make_pool(args.workers) as pool:
        for g in range(1, args.gens + 1):
            t0 = time.time()
            Z, F, CV = archive()
            ZP = Z[P]
            if args.arm == "NOSEL":
                rk_p, cr_p = None, np.zeros(len(P))
            else:
                _, rk_full, cr_full = ea.nsga2_select(F[P], len(P), CV[P])
                rk_p, cr_p = rk_full, cr_full
            n_child = N_CHILD + (N_MUT if args.arm == "NOMUT" else 0)
            Zc, info = latent_children(rng, ZP, rk_p, cr_p, n_child, vae_seed=args.seed * 100003 + g,
                                       use_vae=args.arm != "NOVAE")
            ops = ["cross"] * len(Zc)
            if args.arm != "NOMUT":
                Zm, minfo = mutants(rng, pca, ZP, N_MUT); info.update(minfo)
                Zc = np.concatenate([Zc, Zm]); ops += ["mut"] * len(Zm)
            recs = evaluate_many(pool, list(pca.inverse(Zc)), args.obj == "buck")
            Fc, cvc = objectives(recs, args.obj, cref)
            idx0 = len(F)
            AZ.append(Zc); AF.append(Fc); ACV.append(cvc); AG.append(np.full(len(Zc), g)); AOP.append(np.array(ops)); AREC += recs
            cand = np.concatenate([P, idx0 + np.arange(len(Zc))])
            Z, F, CV = archive()
            if args.arm == "NOSEL":
                P = rng.choice(cand, N_ELITE, replace=False)
            else:
                s, _, _ = ea.nsga2_select(F[cand], N_ELITE, CV[cand]); P = cand[s]
            info.update(t_gen=time.time() - t0, frac_conn=float(np.mean([r["connected"] for r in recs])),
                        mean_eval_s=float(np.mean([r["t"] for r in recs])),
                        frac_mut_in_P=float(np.mean(np.concatenate(AOP)[P] == "mut")))
            log_gen(g, info)
            if g % 10 == 0 or g == args.gens:
                Z, F, CV = archive()
                np.savez_compressed(out / "archive.npz", Z=Z, F=F, CV=CV, gen=np.concatenate(AG), op=np.concatenate(AOP),
                                    P=P, compliance=np.array([r["compliance"] for r in AREC]),
                                    blf=np.array([r["blf"] for r in AREC]), vf=np.array([r["vf"] for r in AREC]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", choices=["stiff", "buck"], required=True)
    ap.add_argument("--arm", choices=["FULL", "NOMUT", "NOVAE", "NOSEL"], default="FULL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--gen0-only", action="store_true")
    a = ap.parse_args()
    if a.gen0_only:
        gen0(a.obj, a.workers)
    else:
        run(a)
