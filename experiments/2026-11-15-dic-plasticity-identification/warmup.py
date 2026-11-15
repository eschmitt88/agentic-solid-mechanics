"""Warm-up gate: recover known parameters from synthetic full-field data.
Truth: O1 outline, fine mesh (h 1.2 / 0.6 mm), Hill + Voce + linear, boundary motion taken from the O1-3
cubic fits.  Synthetic 'DIC': FE displacement at the O1-3 point cloud + 3.6 um noise, force + 2 N noise,
boundary fits re-done from the noisy synthetic field.  Identification: coarser mesh (2.5 / 1.0 mm), start >= 30 % off."""
import os, sys, json, time, numpy as np
import data, specimen, femu

TRUTH = np.array([280.0, 700.0, 2.5, 600.0, 0.0, 1.0, 1.15, 0.42, 2.7])

def generate():
    out = os.path.join(data.CACHE, "O1-syn.npz")
    S = specimen.Specimen("O1-3", h=1.2, h_fine=0.6, tag="_truth")
    d = S.d
    pts = np.stack([d["X0"], d["Y0"]], 1)
    A, inside = specimen.interp_matrix(S.mesh.nodes, S.mesh.elems, pts)
    Ust = np.zeros((data.NST, len(pts))); Vst = np.zeros_like(Ust); alphas = {}
    def hook(j, u, R, state):
        Ust[j] = A @ u[0::2]; Vst[j] = A @ u[1::2]
        if j == 40: alphas["a"] = np.asarray(state[1]).ravel()
    t0 = time.time(); sim = S.simulate(TRUTH, hook=hook); print("truth solve %.0fs" % (time.time() - t0), sim["stats"])
    rng = np.random.default_rng(20261115)
    Ust += rng.normal(0, 3.6e-3, Ust.shape); Vst += rng.normal(0, 3.6e-3, Vst.shape)
    Ust[0] = 0; Vst[0] = 0
    F = sim["F"] + rng.normal(0, 2.0, data.NST); F[0] = 0
    bad = np.repeat((~inside)[None], data.NST, 0)
    coef, bc_rms, xr = data.fit_bc(d["X0"], d["Y0"], Ust, Vst, bad, float(d["ylo"]), float(d["yhi"]))
    coef[0] = 0
    fs = list(data.FIELD_ST)
    z = np.zeros((3, len(pts)), np.float32)
    np.savez_compressed(out, spec="O1-syn", time=d["time"], ext=d["ext"], force=d["force"], iF=d["iF"], xF=d["xF"], frames=d["frames"],
                        drop=np.zeros(data.NST, bool), X0=d["X0"], Y0=d["Y0"], U=Ust[fs], V=Vst[fs], valid=inside, ylo=d["ylo"], yhi=d["yhi"],
                        coef=coef, bc_rms=bc_rms, bc_xrange=xr, vic_Exx=z, vic_Eyy=z, vic_Exy=z, T_mean=d["T_mean"], Fst=F, ext_st=d["ext_st"])
    a95 = float(np.percentile(alphas["a"], 95)); amax = float(alphas["a"].max())
    json.dump({"truth": TRUTH.tolist(), "alpha95": a95, "alpha_max": amax, "truth_mesh": [S.mesh.nn, S.mesh.ne],
               "bc_refit_rms_max_mm": float(bc_rms.max())}, open("results/warmup_truth.json", "w"), indent=1)
    print("alpha95 %.3f alpha_max %.3f" % (a95, amax))

if __name__ == "__main__":
    if not os.path.exists(os.path.join(data.CACHE, "O1-syn.npz")):
        generate()
    tr = json.load(open("results/warmup_truth.json"))
    pool = femu.Pool([("O1-syn", {"h": 2.5, "h_fine": 1.0})], threads=6)
    starts = {"A": [400.0, 450.0, 4.0, 900.0, 0.8, 0.6, 3.6], "B": [180.0, 1000.0, 1.5, 300.0, 1.5, 0.3, 2.0]}
    res = {}
    for k, p0 in starts.items():
        print("start", k, p0); t0 = time.time()
        fit = femu.lm_fit(pool, "M4", p0, max_iter=30)
        p = np.array(fit["p"]); a = np.linspace(0, tr["alpha95"], 200)
        err = np.max(np.abs(femu.sigy(p, a) / femu.sigy(TRUTH, a) - 1))
        fit.update({"start": p0, "sigy_max_rel_err": float(err), "d_a": float(p[6] - TRUTH[6]), "d_h": float(p[7] - TRUTH[7]),
                    "d_c": float(p[8] - TRUTH[8]), "time_s": time.time() - t0, "sigy_max_rel_err_to_alpha_max": float(np.max(np.abs(femu.sigy(p, np.linspace(0, tr["alpha_max"], 200)) / femu.sigy(TRUTH, np.linspace(0, tr["alpha_max"], 200)) - 1)))})
        fit["pass"] = bool(err <= 0.02 and abs(fit["d_a"]) <= 0.05 and abs(fit["d_h"]) <= 0.05 and abs(fit["d_c"]) <= 0.3)
        print("  ->", np.round(p, 4), "sigy err %.4f" % err, "da %.4f dh %.4f dc %.4f" % (fit["d_a"], fit["d_h"], fit["d_c"]), "PASS" if fit["pass"] else "MISS")
        res[k] = fit
    json.dump({"truth": tr, "fits": res}, open("results/warmup.json", "w"), indent=1)
    pool.close()
