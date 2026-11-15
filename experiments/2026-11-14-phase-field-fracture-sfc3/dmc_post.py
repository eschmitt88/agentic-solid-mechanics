"""Damage Mechanics Challenge: calibration summary, blind prediction files and figures."""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_pf import C, ORDER, SEQ
import dmc_data
from dmc import GEOMS, W, Hh, XS

R = "results/dmc"
BASE = dict(l=1.0, h=0.5, ny=26)


def res(geom, l=1.0, h=0.5, ny=26, suffix=""):
    p = f"{R}/r3d_{geom}_l{l}_h{h}_ny{ny}{suffix}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return dict(meta=json.loads(str(z["meta"])), rec=z["rec"], d=z["d"], xs=z["xs"], ys=z["ys"], zs=z["zs"])


def crack_surface(r, thr=0.6):
    """x(y,z) of the crack: location of max d along x for every (y,z) column where max d > thr."""
    d = np.nan_to_num(r["d"], nan=0.0)
    ix = d.argmax(0); mx = d.max(0)
    X = r["xs"][ix].astype(float); X[mx < thr] = np.nan
    return X            # (ny+1, nz)


def calib_table(l, h, ny, slab_suffix="_slab"):
    exp = json.load(open(f"{R}/calibration_data_summary.json"))
    rows = {}
    for g in ("HC", "HB", "HA", "H45"):
        r = res(g, l, h, ny) or (res(g, l, h, 1, slab_suffix) if g != "H45" else None)
        if r is None and g != "H45":
            r = res(g, l, h, 1, "_slab_full")
        if r is None:
            continue
        rows[g] = dict(P_model=r["meta"]["P_peak"], k_el=r["meta"]["k_el"], P_exp=exp[g]["P_mean"], P_exp_std=exp[g]["P_std"],
                       dims="3D" if r["meta"]["ny"] > 1 else "slab")
    logs = np.array([np.log(v["P_exp"] / v["P_model"]) for v in rows.values()])
    s = float(np.exp(logs.mean()))
    for v in rows.values():
        v["P_model_scaled"] = v["P_model"] * s; v["err_pct"] = 100 * (v["P_model"] * s / v["P_exp"] - 1)
    rms = float(np.sqrt(np.mean((logs - logs.mean()) ** 2) * len(logs) / max(len(logs) - 1, 1)))
    return rows, s, rms


if __name__ == "__main__":
    exp = json.load(open(f"{R}/calibration_data_summary.json"))
    out = {}
    for name, (l, h, ny) in {"base l=1.0 h=0.5": (1.0, 0.5, 26), "finer mesh l=1.0 h=0.35": (1.0, 0.35, 36), "smaller l=0.6 h=0.3": (0.6, 0.3, 42)}.items():
        ch = res("CH", l, h, ny)
        if ch is None:
            continue
        rows, s, rms = calib_table(l, h, ny)
        Gc0 = ch["meta"]["Gc"]
        out[name] = dict(l=l, h=h, ny=ny, calib=rows, load_scale=s, Gc_calibrated=Gc0 * s ** 2, rms_log_misfit=rms,
                         CH_P_raw=ch["meta"]["P_peak"], CH_P_pred=ch["meta"]["P_peak"] * s, CH_k_el=ch["meta"]["k_el"],
                         CH_delta_FE_peak=ch["meta"]["delta_peak"] * s)
        print(name, "scale", round(s, 4), "Gc", round(Gc0 * s ** 2, 4), "rms", round(rms, 3), "CH pred", round(ch["meta"]["P_peak"] * s, 1))
        for g, v in rows.items():
            print(f"   {g:4s} {v['dims']:4s} model {v['P_model_scaled']:.1f}  exp {v['P_exp']:.1f} ± {v['P_exp_std']:.1f}  err {v['err_pct']:+.1f}%")
    json.dump(out, open(f"{R}/calibration_fit.json", "w"), indent=1)
