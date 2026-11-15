"""Fit the non-structural (rod contact + fixture) compliance from the 16 calibration beams.
delta_measured(P) = delta0 + delta_FE(P) + a * P**n ; delta_FE from the calibrated 3D phase-field runs."""
import json, numpy as np
from scipy.optimize import least_squares
import dmc_data
from dmc_post import res, calib_table

rows, s, rms = calib_table(1.0, 0.5, 26)
curves = {}
for g in ("HC", "HB", "HA", "H45"):
    r = res(g)
    rec = r["rec"]; i = int(rec[:, 1].argmax())
    curves[g] = (np.concatenate([[0], rec[: i + 1, 1] * s]), np.concatenate([[0], rec[: i + 1, 0] * s]))   # P, delta_FE up to peak

data = []
for g, names in dmc_data.BEAMS.items():
    for nme in names:
        d, P = dmc_data.load_ld(nme); Ps = dmc_data.smooth(P); i = int(np.argmax(Ps))
        sel = (np.arange(len(P)) < i) & (Ps > 0.15 * Ps[i]) & (Ps < 0.9 * Ps[i])
        idx = np.nonzero(sel)[0][::20]
        Pm, dFE = curves[g]
        data.append((nme, g, Ps[idx], d[idx], np.interp(Ps[idx], Pm, dFE)))


def resid(p):
    a, n = p[0], p[1]; d0 = p[2:]
    return np.concatenate([(d - (d0[k] + dfe + a * P ** n)) for k, (_, _, P, d, dfe) in enumerate(data)])

x0 = np.concatenate([[1e-3, 0.9], np.zeros(len(data))])
sol = least_squares(resid, x0, bounds=(np.concatenate([[1e-6, 0.3], -0.3 * np.ones(len(data))]), np.concatenate([[1.0, 1.5], 0.3 * np.ones(len(data))])))
a, n = sol.x[:2]; d0 = sol.x[2:]
r = resid(sol.x)
# per-test multiplicative scatter of the contact term (refit amplitude per test with n fixed)
amps = []
for k, (nme, g, P, d, dfe) in enumerate(data):
    A = np.stack([np.ones_like(P), P ** n], 1); c = np.linalg.lstsq(A, d - dfe, rcond=None)[0]; amps.append(c[1])
amps = np.array(amps)
out = dict(a=float(a), n=float(n), rms_mm=float(np.sqrt(np.mean(r ** 2))), toe_mean=float(d0.mean()), toe_std=float(d0.std(ddof=1)),
           amp_per_test={d_[0]: float(v) for d_, v in zip(data, amps)}, amp_cov=float(amps.std(ddof=1) / amps.mean()), amp_mean=float(amps.mean()),
           toes={d_[0]: float(v) for d_, v in zip(data, d0)})
json.dump(out, open("results/dmc/contact_fit.json", "w"), indent=1)
print({k: v for k, v in out.items() if not isinstance(v, dict)})
