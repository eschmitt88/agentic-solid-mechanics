"""Independent verification of a design: CalculiX (pre-registered verifier) + mesh convergence in both solvers.

usage: python verify.py <name> <json-with-p> [target]
Writes results/verify_<name>.json and results/ccx/<name>_*/curve.csv
"""
from __future__ import annotations

import json
import sys

import jax.numpy as jnp
import numpy as np

import ccx
import design
import fem


def ccx_sigma(name, p, lc_div, lateral="fixed", dmax=12.5):
    tmin = float(np.min(np.asarray(fem.thickness(jnp.asarray(p), jnp.linspace(0, 1, 201)))))
    d, F, info = ccx.run(name, p, Lh=design.LH, ws=design.WS, dmax=dmax, ninc=100, lc=tmin / lc_div, lateral=lateral)
    if not info["completed"]:
        sig = np.full(len(design.DS), np.nan)
        ok = design.DS <= d[-1] + 1e-9 if len(d) else np.zeros(len(design.DS), bool)
        if len(d):
            sig[ok] = np.interp(design.DS[ok], d, F) * design.KPA
    else:
        sig = np.interp(design.DS, d, F) * design.KPA
    return sig, info


def jax_sigma(p, nx, ny):
    S = fem.Solver(nx=nx, ny=ny, Lh=design.LH)
    F, us, infos = S.path(p, design.DS)
    return F * design.KPA, dict(ndof=int(S.mesh["nfree"]), switched=int(sum(i["switched"] for i in infos)),
                                bad=int(sum((not i["converged"]) or i.get("unstable", False) for i in infos)))


def checks(tname, sig, tgt):
    e = design.EPS
    out = dict(nrmse=design.nrmse(sig, tgt) if np.all(np.isfinite(sig)) else None)
    if not np.all(np.isfinite(sig)):
        out["pass"] = False
        out["note"] = "CalculiX run incomplete"
        return out
    ds = np.diff(np.concatenate([[0.0], sig]))
    if tname == "T1":
        out["monotonic"] = bool(np.all(ds > 0))
        out["pass"] = bool(out["nrmse"] <= 0.05 and out["monotonic"])
        out["tol"] = 0.05
    elif tname == "T2":
        m = (e >= 0.12 - 1e-9) & (e <= 0.33 + 1e-9)
        out["plateau_max_dev_kPa"] = float(np.max(np.abs(sig[m] - 0.40)))
        out["pass"] = bool(out["nrmse"] <= 0.05 and out["plateau_max_dev_kPa"] <= 0.06)
        out["tol"] = 0.05
    elif tname == "T3":
        imax = [i for i in range(1, len(sig) - 1) if sig[i] > sig[i - 1] and sig[i] >= sig[i + 1]]
        imin = [i for i in range(1, len(sig) - 1) if sig[i] < sig[i - 1] and sig[i] <= sig[i + 1]]
        out["has_peak_then_valley"] = bool(imax and imin and min(imax) < max(imin))
        if out["has_peak_then_valley"]:
            out["peak"] = [float(e[min(imax)]), float(sig[min(imax)])]
            out["valley"] = [float(e[max(imin)]), float(sig[max(imin)])]
        out["pass"] = bool(out["nrmse"] <= 0.08 and out["has_peak_then_valley"])
        out["tol"] = 0.08
    return out


def verify(name, p, tname=None):
    p = np.asarray(p, float)
    res = dict(name=name, p=p.tolist(), target=tname, eps=design.EPS.tolist())
    tgt = design.TARGETS[tname](design.EPS) if tname else None
    if tgt is not None:
        res["target_sigma"] = tgt.tolist()
    curves = {}
    for key, (nx, ny) in [("jax_40x2", (40, 2)), ("jax_80x2", (80, 2)), ("jax_80x4", (80, 4))]:
        curves[key], res[key + "_info"] = jax_sigma(p, nx, ny)
        print(name, key, res[key + "_info"], flush=True)
    for key, div, lat in [("ccx_t3", 3, "fixed"), ("ccx_t6", 6, "fixed"), ("ccx_t3_latfree", 3, "free")]:
        curves[key], res[key + "_info"] = ccx_sigma(f"{name}_{key}", p, div, lat)
        print(name, key, res[key + "_info"], flush=True)
    for k, v in curves.items():
        res[k] = [None if not np.isfinite(x) else float(x) for x in v]
    rng = (tgt.max() - tgt.min()) if tgt is not None else (np.nanmax(curves["ccx_t3"]) - np.nanmin(curves["ccx_t3"]))

    def gap(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)) / rng) if m.any() else None

    res["gap_jax40x2_vs_ccx_t3"] = gap(curves["jax_40x2"], curves["ccx_t3"])
    res["mesh_conv_jax_40x2_vs_80x4"] = gap(curves["jax_40x2"], curves["jax_80x4"])
    res["mesh_conv_jax_80x2_vs_80x4"] = gap(curves["jax_80x2"], curves["jax_80x4"])
    res["mesh_conv_ccx_t3_vs_t6"] = gap(curves["ccx_t3"], curves["ccx_t6"])
    res["lateral_free_vs_fixed"] = gap(curves["ccx_t3_latfree"], curves["ccx_t3"])
    if tgt is not None:
        res["optimizer_nrmse_jax40x2"] = design.nrmse(curves["jax_40x2"], tgt)
        res["jax80x4_nrmse"] = design.nrmse(curves["jax_80x4"], tgt)
        res["verdict"] = checks(tname, curves["ccx_t3"], tgt)  # pre-registered verifier
        res["verdict_fine_mesh"] = checks(tname, curves["ccx_t6"], tgt)
        res["verdict_lateral_free"] = checks(tname, curves["ccx_t3_latfree"], tgt)
    json.dump(res, open(f"results/verify_{name}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    name, src = sys.argv[1], sys.argv[2]
    tname = sys.argv[3] if len(sys.argv) > 3 else None
    p = json.load(open(src))["p"]
    r = verify(name, p, tname)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}, indent=1))
