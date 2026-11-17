"""Run one arm of trial 14 (follow-ups to trial 13; evolutionary de-homogenisation, Xu et al. 2026).
dehom-dcbeam copy: the problem is hf.PROBLEM (DCBEAM, double-clamped beam); new objective
`umax` (the paper's "maximum displacement"): (vf, umax), defined like `stiff` but on umax.
`buck` is not available for this problem (fixed supports; hf.Problem raises).

  python -u run_arm.py --obj stiff|umax|buck --arm ARM --seed S --gens G --workers W
  python -u run_arm.py --obj stiff|umax|buck --gen0-only [--enc inv5|raw3]   # gen-0 evaluation cache

Arms (110 HF evaluations per generation unless stated):
  FULL     NSGA-II + VAE latent crossover (latent 8) (100) + image mutation (10)        [paper]
  VAE32    FULL with VAE latent 32;  VAE64: latent 64                                  [latent sweep]
  RAWTH    FULL with the paper's raw-theta encoding (w1, w2, theta), PCA per channel   [faithful enc.]
  PCA      NSGA-II + BLX-0.25 crossover in whitened PCA space (100) + image mutation (10)
  PCANOV   PCA + novelty as a third survival objective (mean distance to 15 nearest archive
           designs on the first 4 PCs of each channel); HV still on the 2 real objectives
  ROB3     PCA, but every child is evaluated under 3 random phasor seeds and scored on
           (mean vf, geometric-mean compliance | geometric-mean BLF): 330 evaluations/generation
  LOTTERY  no selection, no recombination: each child = a random gen-0 front design + N(0, 0.1^2)
           in whitened PCA space (noise-mining null)

Objectives (minimised): stiff = (vf, C); umax = (vf, max |u|); buck = (vf, -BLF) s.t. C <= gen-0 envelope at the same vf.
"""
from __future__ import annotations

import argparse, json, os, sys, time, zlib
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
_THETA_ND = np.load(HERE / "results" / "gen0" / "lf_designs.npz")["theta"][0]   # non-design entries only

ARM_ENC = {"RAWTH": "raw3"}


# ---------------------------------------------------------------------------- encodings
def features(w1, w2, th, enc="inv5"):
    """(n, 20, 40) fields -> (n, Nd, C) design-element features."""
    if enc == "raw3":
        return np.stack([w1, w2, th], -1)[:, DES, :]
    return ea.encode(w1, w2, th)[:, DES, :]


def fields_from_features(Fd, enc="inv5"):
    """(Nd, C) -> w1, w2, theta images (20, 40) with fixed non-design values."""
    if enc == "raw3":
        w1 = np.zeros(DES.shape); w2 = np.zeros(DES.shape); th = _THETA_ND.copy()
        w1[DES] = np.clip(Fd[:, 0], 0, 1); w2[DES] = np.clip(Fd[:, 1], 0, 1); th[DES] = Fd[:, 2]
    else:
        img = np.zeros(DES.shape + (ea.NCH,))
        img[DES] = Fd
        w1, w2, th = ea.decode(img)
        # non-design elements carry no encoded orientation: use the LF optimiser's fixed angle
        th[~DES] = _THETA_ND[~DES]
    w1[SOLID_C], w2[SOLID_C] = 1.0, 1.0
    w1[VOID_C], w2[VOID_C] = 0.0, 0.0
    return w1, w2, th


def image_from_features(Fd, enc="inv5"):
    w1, w2, th = fields_from_features(Fd, enc)
    if enc == "raw3":
        return np.stack([w1, w2, th], -1)
    return ea.encode(w1, w2, th)


def features_from_image(img, enc="inv5"):
    if enc == "raw3":
        return img[DES]
    return img[DES]


# ---------------------------------------------------------------------------- workers
_W = {}


def _winit():
    import dehom  # noqa
    _W["dehom"] = dehom
    _W["prob"] = hf.Problem(hf.PROBLEM, hf.H_FINE)


def _evaluate(task):
    """task = (Fd, enc, buck, seeds): evaluate one design under each phase seed in `seeds`
    (None = the deterministic default realisation)."""
    Fd, enc, buck, seeds = task
    t0 = time.time()
    w1, w2, th = fields_from_features(Fd, enc)
    out = []
    for s in seeds:
        d = _W["dehom"].dehomogenise(w1, w2, th, phase_seed=s)
        r = _W["prob"].evaluate(d["phi"], buckling=buck, nev=1)
        out.append(dict(vf=r["vf"], connected=r["connected"], compliance=r["compliance"], umax=r["umax"],
                        blf=r.get("blf", np.nan)))
    return dict(evals=out, t=time.time() - t0)


def aggregate(res):
    """Mean vf, geometric-mean C and BLF over the realisations of one design (any disconnected
    realisation makes the design disconnected)."""
    ev = res["evals"]
    conn = all(e["connected"] for e in ev)
    if len(ev) == 1:
        return dict(ev[0], t=res["t"], n_real=1)
    C = np.array([e["compliance"] for e in ev]); B = np.array([e["blf"] for e in ev])
    return dict(vf=float(np.mean([e["vf"] for e in ev])), connected=conn,
                compliance=float(np.exp(np.mean(np.log(C)))) if conn else np.inf,
                umax=float(np.mean([e["umax"] for e in ev])),
                blf=float(np.exp(np.mean(np.log(np.maximum(B, 1e-12))))) if conn and np.isfinite(B).all() else np.nan,
                t=res["t"], n_real=len(ev))


def evaluate_many(pool, Fds, enc, buck, seeds_list=None):
    seeds_list = seeds_list if seeds_list is not None else [[None]] * len(Fds)
    res = pool.map(_evaluate, [(F, enc, buck, s) for F, s in zip(Fds, seeds_list)], chunksize=1)
    return [aggregate(r) for r in res]


def make_pool(workers):
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(workers, initializer=_winit)


# ---------------------------------------------------------------------------- objectives
def objectives(recs, obj, cref=None):
    vf = np.array([r["vf"] for r in recs]); C = np.array([r["compliance"] for r in recs])
    ok = np.array([bool(r["connected"]) and np.isfinite(r["compliance"]) for r in recs])
    if obj == "stiff":
        F = np.stack([vf, np.where(ok, C, 1e30)], 1)
        cv = np.where(ok, 0.0, BAD_CV)
    elif obj == "umax":
        U = np.array([r["umax"] for r in recs])
        ok = ok & np.isfinite(U)
        F = np.stack([vf, np.where(ok, U, 1e30)], 1)
        cv = np.where(ok, 0.0, BAD_CV)
    else:
        blf = np.array([r["blf"] for r in recs])
        ok = ok & np.isfinite(blf)
        F = np.stack([vf, np.where(ok, -blf, 1e30)], 1)
        cv = np.where(ok, np.maximum(0.0, C / cref(vf) - 1.0), BAD_CV)
    return F, cv


def compliance_reference(vf0, C0):
    F = np.stack([vf0, C0], 1)
    r = ea.nondominated_sort(F)
    P = F[r == 0]; P = P[np.argsort(P[:, 0])]
    lv, lc = P[:, 0], np.log(P[:, 1])
    return lambda v: np.exp(np.interp(v, lv, lc))


# ---------------------------------------------------------------------------- gen 0
def pca_path(enc):
    return RES / "gen0" / ("pca.npz" if enc == "inv5" else f"pca_{enc}.npz")


def gen0_path(obj, enc, robust=False):
    tag = "" if enc == "inv5" else f"_{enc}"
    return RES / "gen0" / f"eval14_{obj}{tag}{'_rob3' if robust else ''}.npz"


def gen0(obj, workers, enc="inv5", robust=False):
    lf = np.load(RES / "gen0" / "lf_designs.npz")
    F0 = features(lf["w1"], lf["w2"], lf["theta"], enc)
    if enc == "inv5" and pca_path(enc).exists():
        pca = load_pca(enc)                              # trial-13 PCA (identical fit), kept for comparability
    else:
        pca = ea.ChannelPCA().fit(F0)
        np.savez(pca_path(enc), **{f"Phi{c}": pca.Phi[c] for c in range(len(pca.nc))},
                 **{f"ev{c}": pca.ev[c] for c in range(len(pca.nc))}, mean=pca.mean, sd=pca.sd, nc=np.array(pca.nc))
    Z0 = pca.transform(F0)
    F0r = pca.inverse(Z0)
    buck = obj == "buck"
    rng = np.random.default_rng(14_000 + (1 if buck else 0))
    seeds = [list(rng.integers(1, 2**31, 3)) for _ in range(len(F0r))] if robust else None
    with make_pool(workers) as pool:
        t = time.time()
        rec = evaluate_many(pool, list(F0r), enc, buck, seeds)
        dt = time.time() - t
    np.savez(gen0_path(obj, enc, robust), Z0=Z0, **{f"rec_{k}": np.array([r[k] for r in rec]) for k in rec[0]})
    info = dict(obj=obj, enc=enc, robust=robust, n=len(F0), pca_nc=list(map(int, pca.nc)), pca_dim=pca.dim,
                pca_recon_err=pca.recon_error(F0), wall_s=dt, mean_eval_s=float(np.mean([r["t"] for r in rec])))
    gen0_path(obj, enc, robust).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def load_pca(enc="inv5"):
    d = np.load(pca_path(enc))
    nc = [int(x) for x in d["nc"]]
    p = ea.ChannelPCA()
    p.Phi = [d[f"Phi{c}"] for c in range(len(nc))]; p.ev = [d[f"ev{c}"] for c in range(len(nc))]
    p.mean, p.sd, p.nc = d["mean"], d["sd"], nc
    return p


def load_gen0(obj, enc="inv5", robust=False):
    d = np.load(gen0_path(obj, enc, robust))
    keys = [k[4:] for k in d.files if k.startswith("rec_")]
    recs = [{k: d[f"rec_{k}"][i] for k in keys} for i in range(len(d["Z0"]))]
    for r in recs:
        r["connected"] = bool(r["connected"])
    return d["Z0"], recs


def cref_for(obj):
    """Buckling constraint reference: gen-0 (inv5, nominal) compliance envelope — shared by all arms."""
    if obj != "buck":
        return None
    d = np.load(gen0_path("buck", "inv5"))
    ok = d["rec_connected"].astype(bool)
    return compliance_reference(d["rec_vf"][ok], d["rec_compliance"][ok])


def shared_normaliser(obj):
    """HV normalisation from the inv5 nominal gen-0 feasible set — identical for every arm."""
    _, rec = load_gen0(obj, "inv5")
    F0, cv0 = objectives(rec, obj, cref_for(obj))
    return ea.Normaliser(F0[cv0 <= 0])


# ---------------------------------------------------------------------------- operators
def latent_children(rng, Z, rank, crowd, n, vae_seed, mode="vae", latent=8, noise=0.1):
    a = ea.tournament(rng, rank, crowd, n) if rank is not None else rng.integers(0, len(Z), n)
    b = ea.tournament(rng, rank, crowd, n) if rank is not None else rng.integers(0, len(Z), n)
    t = rng.uniform(-0.25, 1.25, (n, 1))
    if mode == "vae":
        vae = ea.VAE(Z.shape[1], latent=latent, seed=vae_seed).fit(Z)
        mu, sd = vae.encode(Z)
        L = (1 - t) * mu[a] + t * mu[b] + rng.normal(size=(n, mu.shape[1])) * sd.mean(0)
        return vae.decode(L), dict(vae_loss=vae.final_loss, vae_sigma=float(sd.mean()))
    Zc = (1 - t) * Z[a] + t * Z[b] + rng.normal(size=(n, Z.shape[1])) * noise
    return Zc, {}


def mutants(rng, pca, Zpar, n, enc):
    import lf
    out, kept = [], []
    for i in rng.choice(len(Zpar), n, replace=False):
        Fd = pca.inverse(Zpar[i])
        w1, w2, th = fields_from_features(Fd, enc)
        ux, uy = lf.lf_displacement(w1, w2, th)
        img = image_from_features(Fd, enc)
        if enc == "raw3":
            m = ea.mutate_image(img, np.asarray(ux), np.asarray(uy), rng)
        else:
            m = ea.mutate_image(img, np.asarray(ux), np.asarray(uy), rng)
        z = pca.transform(features_from_image(m, enc)[None])[0]
        dimg = np.linalg.norm(features_from_image(m, enc) - Fd)
        dimg_proj = np.linalg.norm(pca.inverse(z) - Fd)
        out.append(z); kept.append(dimg_proj / max(dimg, 1e-12))
    return np.array(out), dict(mut_kept_frac=float(np.mean(kept)))


def novelty(Zc, Zarch, pca, k=15, per_ch=4):
    """Mean distance to the k nearest archive designs on the first `per_ch` PCs of each channel."""
    idx = np.concatenate([np.arange(per_ch) + int(sum(pca.nc[:c])) for c in range(len(pca.nc))])
    from scipy.spatial import cKDTree
    d, _ = cKDTree(Zarch[:, idx]).query(Zc[:, idx], k=k + 1)
    return d[:, 1:].mean(1)                  # skip self (distance 0): candidates are in the archive


# ---------------------------------------------------------------------------- run
def run(args):
    enc = ARM_ENC.get(args.arm, "inv5")
    rng = np.random.default_rng(args.seed)
    pca = load_pca(enc)
    robust = args.arm == "ROB3"
    Z0, rec0 = load_gen0(args.obj, enc, robust)
    cref = cref_for(args.obj)
    F0, cv0 = objectives(rec0, args.obj, cref)
    norm = shared_normaliser(args.obj)
    out = RES / "arms" / f"{args.obj}_{args.arm}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args)))

    AZ, AF, ACV, AG, AOP = [Z0], [F0], [cv0], [np.zeros(len(Z0), int)], [np.array(["gen0"] * len(Z0))]
    AREC = list(rec0)
    sel, _, _ = ea.nsga2_select(F0, N_ELITE, cv0)
    P = sel
    front0 = np.where(cv0 <= 0)[0][ea.nondominated_sort(F0[cv0 <= 0]) == 0]
    hist_f = out / "history.jsonl"
    if hist_f.exists():
        hist_f.unlink()

    def archive():
        return np.concatenate(AZ), np.concatenate(AF), np.concatenate(ACV)

    def log_gen(g, extra):
        Z, F, CV = archive()
        Fn = norm(F[CV <= 0])
        h = dict(gen=g, n_eval=int(sum(r.get("n_real", 1) for r in AREC)), n_designs=len(F),
                 hv11=ea.hv2d(Fn, np.array([1.1, 1.1])), hv20=ea.hv2d(Fn, np.array([2.0, 2.0])),
                 pop_div=float(np.mean(np.linalg.norm(Z[P][:, None] - Z[P][None], axis=-1))), **extra)
        print(json.dumps(h), flush=True)
        with open(hist_f, "a") as f:
            f.write(json.dumps(h, default=float) + "\n")

    def save():
        Z, F, CV = archive()
        np.savez_compressed(out / "archive.npz", Z=Z, F=F, CV=CV, gen=np.concatenate(AG), op=np.concatenate(AOP), P=P,
                            compliance=np.array([r["compliance"] for r in AREC]), blf=np.array([r["blf"] for r in AREC]),
                            umax=np.array([r["umax"] for r in AREC]),
                            vf=np.array([r["vf"] for r in AREC]))

    log_gen(0, {})
    buck = args.obj == "buck"
    lat = {"VAE32": 32, "VAE64": 64}.get(args.arm, 8)
    mode = "vae" if args.arm in ("FULL", "VAE32", "VAE64", "RAWTH") else "pca"
    with make_pool(args.workers) as pool:
        for g in range(1, args.gens + 1):
            t0 = time.time()
            Z, F, CV = archive()
            info = {}
            if args.arm == "LOTTERY":
                par = rng.choice(front0, N_CHILD + N_MUT)
                Zc = Z0[par] + rng.normal(size=(len(par), Z0.shape[1])) * 0.1
                ops = ["lottery"] * len(Zc)
            else:
                _, rk, cr = ea.nsga2_select(F[P], len(P), CV[P])
                Zc, info = latent_children(rng, Z[P], rk, cr, N_CHILD, vae_seed=args.seed * 100003 + g, mode=mode, latent=lat)
                ops = ["cross"] * len(Zc)
                Zm, minfo = mutants(rng, pca, Z[P], N_MUT, enc); info.update(minfo)
                Zc = np.concatenate([Zc, Zm]); ops += ["mut"] * len(Zm)
            seeds = [list(rng.integers(1, 2**31, 3)) for _ in range(len(Zc))] if robust else None
            recs = evaluate_many(pool, list(pca.inverse(Zc)), enc, buck, seeds)
            Fc, cvc = objectives(recs, args.obj, cref)
            idx0 = len(F)
            AZ.append(Zc); AF.append(Fc); ACV.append(cvc); AG.append(np.full(len(Zc), g)); AOP.append(np.array(ops)); AREC += recs
            cand = np.concatenate([P, idx0 + np.arange(len(Zc))])
            Z, F, CV = archive()
            if args.arm == "LOTTERY":
                P = cand[:N_ELITE]                               # unused
            elif args.arm == "PCANOV":
                nov = novelty(Z[cand], Z, pca)
                F3 = np.concatenate([F[cand], -nov[:, None]], 1)
                s, _, _ = ea.nsga2_select(F3, N_ELITE, CV[cand]); P = cand[s]
            else:
                s, _, _ = ea.nsga2_select(F[cand], N_ELITE, CV[cand]); P = cand[s]
            info.update(t_gen=time.time() - t0, frac_conn=float(np.mean([r["connected"] for r in recs])),
                        mean_eval_s=float(np.mean([r["t"] for r in recs])),
                        frac_mut_in_P=float(np.mean(np.concatenate(AOP)[P] == "mut")))
            log_gen(g, info)
            if g % 25 == 0 or g == args.gens:
                save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", choices=["stiff", "umax", "buck"], required=True)
    ap.add_argument("--arm", choices=["FULL", "VAE32", "VAE64", "RAWTH", "PCA", "PCANOV", "ROB3", "LOTTERY"], default="FULL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--gen0-only", action="store_true")
    ap.add_argument("--enc", default="inv5")
    ap.add_argument("--robust", action="store_true")
    a = ap.parse_args()
    if a.gen0_only:
        gen0(a.obj, a.workers, a.enc, a.robust)
    else:
        run(a)
