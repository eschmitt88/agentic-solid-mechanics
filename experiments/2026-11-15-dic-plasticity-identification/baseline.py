"""Uniaxial-only baseline: von Mises + each hardening law fitted to the mean true-stress / plastic-strain
curve of the slow transverse dog bones BT3, CT3, CT4, up to maximum force (pre-registered)."""
import os, json, numpy as np, scipy.io as sio
from scipy.optimize import least_squares
import data, femu

A0 = 12.7 * 1.5
E = 193000.0

def curve(name):
    import h5py
    m = h5py.File(os.path.join(data.RAW, "dogbone", f"{name}-Data.mat"), "r")
    e = np.array(m["strain"]).ravel(); F = np.array(m["force"]).ravel() * 1e3
    ok = np.isfinite(e) & np.isfinite(F)
    e, F = e[ok], F[ok]
    i = np.argmax(F)
    e, F = e[: i + 1], F[: i + 1]
    s_true = F / A0 * (1 + e); eps = np.log1p(e); ep = eps - s_true / E
    # drop spurious unloadings (true stress > 1 % of its maximum below its running maximum), then sort by ep
    keep = s_true >= np.maximum.accumulate(s_true) - 0.01 * s_true.max()
    e, F, ep, s_true = e[keep], F[keep], ep[keep], s_true[keep]
    o = np.argsort(ep, kind="stable")
    return e[o], F[o] / A0, ep[o], s_true[o]

def fit(form, ep, s, p0):
    free = femu.FORMS[form]["free"]
    f = lambda x: femu.sigy(femu.full_p(form, x), ep) - s
    best = None
    for p in p0:
        r = least_squares(f, np.log(p), method="lm", max_nfev=4000)
        if best is None or r.cost < best.cost: best = r
    return femu.full_p(form, best.x), float(np.sqrt(2 * best.cost / len(ep)))

if __name__ == "__main__":
    grid = None; curves = {}
    for n in ["BT3", "CT3", "CT4"]:
        e, s_eng, ep, s = curve(n)
        curves[n] = (e, s_eng, ep, s)
        print(n, "UTS %.1f MPa at e=%.3f, true %.0f MPa at ep=%.3f, npts %d, max gap in ep %.3f" % (s_eng[-1], e[-1], s[-1], ep[-1], len(e), np.max(np.diff(ep))))
    ep_max = min(c[2][-1] for c in curves.values())
    grid = np.linspace(0.002, ep_max, 300)
    S = np.array([np.interp(grid, c[2], c[3]) for c in curves.values()])
    print("spread between dog bones: max %.1f MPa (%.2f %%)" % (np.max(S.max(0) - S.min(0)), 100 * np.max((S.max(0) - S.min(0)) / S.mean(0))))
    mean = S.mean(0)
    out = {"ep_max": float(ep_max), "grid": grid.tolist(), "mean_true_stress": mean.tolist(), "A0": A0, "E": E, "fits": {}}
    starts = {"M1": [[300, 900, 3]], "M2": [[300, 300, 20, 1500], [250, 600, 3, 800]],
              "M3": [[250, 100, 50, 1500, 300, 3], [280, 80, 100, 1000, 800, 2], [250, 200, 20, 500, 1500, 1.0]]}
    for form in ["M1", "M2", "M3"]:
        p, rms = fit(form, grid, mean, starts[form])
        out["fits"][form] = {"p": p.tolist(), "rms_MPa": rms}
        print(form, np.round(p[:6], 3), "rms %.2f MPa" % rms)
    json.dump(out, open("results/baseline.json", "w"), indent=1)
