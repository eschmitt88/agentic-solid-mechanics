"""Addendum B analysis: decompose each run's hypervolume gain into realisation fitting, baseline
selection and front extension, and report matched-volume attainment ratios.

  python decompose.py [EXPERIMENT_DIR ...]      -> results/decomposition.json (+ printed table)

Per run (final archive and gen-100 snapshot):
  nominal       archive HV gain vs the gen-0 nominal set (search signal)
  ho_front      held-out gain vs the held-out gen-0 *front* (pre-registered primary)
  ho_all        held-out gain vs the held-out front of *all 480* gen-0 designs (fair baseline)
  ho_all_inrng  same, HV clipped at the baseline's minimum volume fraction: run points below it are
                projected onto it (their objective counts, the extension area does not)
  att_inrng     geometric-mean ratio run/baseline of the attainment curve (best objective at vf <= v)
                on a grid over the baseline's vf range (< 1 = run better; for BLF the inverse is used)
  att_ext       same ratio vs the LF extension control (results/extension) over [ext vf min, baseline vf min]
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np


def load_ctx(E):
    E = Path(E).resolve()
    sys.path.insert(0, str(E))
    for m in ("heldout", "run_arm", "ea", "hf", "dehom"):
        sys.modules.pop(m, None)
    import heldout as H, ea, run_arm as ra
    sys.path.remove(str(E))
    return E, H, ea, ra


def averaged(H, f, obj):
    F, _ = H.averaged_objectives(np.load(f), obj)
    return F


def front(ea, F):
    P = F[ea.nondominated_sort(F) == 0]
    return P[np.argsort(P[:, 0])]


def hv(ea, norm, F, lo=None):
    F = np.array(F, float)
    if lo is not None:
        F[:, 0] = np.maximum(F[:, 0], lo)
    return ea.hv2d(norm(F), np.array([1.1, 1.1]))


def attainment(F, grid, maximise=False):
    out = []
    for v in grid:
        m = F[:, 0] <= v
        out.append(np.nan if not m.any() else (F[m, 1].min()))
    a = np.array(out)
    return -a if maximise else a          # F[:,1] holds -BLF for buck: min(-BLF) -> BLF


def ratio(a, b, maximise):
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if not ok.any():
        return np.nan
    r = (b[ok] / a[ok]) if maximise else (a[ok] / b[ok])     # < 1 = run better in both cases
    return float(np.exp(np.mean(np.log(np.abs(r)))))


def decompose(E):
    E, H, ea, ra = load_ctx(E)
    rows = []
    for f in sorted(glob.glob(str(E / "results/heldout/*__archive*.npz"))):
        run, tag = Path(f).stem.split("__")
        args = json.load(open(E / "results/arms" / run / "args.json")); obj = args["obj"]
        f_front = E / f"results/heldout/{obj}__gen0.npz"; f_all = E / f"results/heldout/{obj}__gen0all.npz"
        if not f_all.exists():
            continue
        norm = ra.shared_normaliser(obj)
        Fr = averaged(H, f, obj); Ffr = averaged(H, f_front, obj); Fall = averaged(H, f_all, obj)
        a = np.load(E / "results/arms" / run / f"{tag}.npz")
        _, rec0 = ra.load_gen0(obj); F0, cv0 = ra.objectives(rec0, obj, ra.cref_for(obj))
        nominal = hv(ea, norm, a["F"][a["CV"] <= 0]) / hv(ea, norm, F0[cv0 <= 0]) - 1
        ho_front = hv(ea, norm, np.concatenate([Ffr, Fr])) / hv(ea, norm, Ffr) - 1
        hb = hv(ea, norm, Fall)
        ho_all = hv(ea, norm, np.concatenate([Fall, Fr])) / hb - 1
        Pall = front(ea, Fall); lo, hi = Pall[0, 0], Pall[-1, 0]
        ho_all_in = hv(ea, norm, np.concatenate([Fall, Fr]), lo) / hv(ea, norm, Fall, lo) - 1
        mx = obj == "buck"
        grid = np.linspace(lo, hi, 26)
        att_in = ratio(attainment(np.concatenate([Fall, Fr]), grid, mx), attainment(Fall, grid, mx), mx)
        att_ext = np.nan; ext = E / f"results/extension/heldout_{obj}.npz"
        if ext.exists():
            Fx = np.concatenate([averaged(H, ext, obj), Fall])
            gx = np.linspace(max(Fx[:, 0].min(), Fr[:, 0].min()), lo, 16)
            att_ext = ratio(attainment(Fr, gx, mx), attainment(Fx, gx, mx), mx)
        rows.append(dict(trial=E.name, run=run, snapshot=tag, gen=int(a["gen"].max()), obj=obj,
                         nominal=nominal, ho_front=ho_front, ho_all=ho_all, ho_all_inrng=ho_all_in,
                         att_inrng=att_in, att_ext=att_ext, run_vf_min=float(Fr[:, 0].min()),
                         base_vf=[float(lo), float(hi)]))
    return rows


if __name__ == "__main__":
    rows = []
    for E in sys.argv[1:] or ["."]:
        rows += decompose(E)
    for r in rows:
        print(f"{r['run']:20s} g{r['gen']:3d} nominal {100*r['nominal']:6.1f} | held-out vs front {100*r['ho_front']:6.1f}"
              f" vs all {100*r['ho_all']:6.1f} in-range {100*r['ho_all_inrng']:6.1f} | att in-range {r['att_inrng']:.3f}"
              f" ext {r['att_ext']:.3f} | run vf min {r['run_vf_min']:.3f} base vf {r['base_vf'][0]:.3f}-{r['base_vf'][1]:.3f}")
    json.dump(rows, open("results/decomposition.json", "w"), indent=1, default=float)
