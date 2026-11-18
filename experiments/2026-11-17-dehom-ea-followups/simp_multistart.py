"""H7 SIMP baseline, multi-start (pre-registered): rmin in {1.0, 1.5, 2.0} mm x penal in {3, 4} at
vf in {0.25 ... 0.50}; each design evaluated by the trial-14 solid-only evaluator (static + buckling).
SIMP designs have no phasor realisation, so held-out = nominal. Output results/simp/multistart.json."""
import itertools, json, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hf, simp  # noqa

VFS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
GRID = list(itertools.product(VFS, [1.0, 1.5, 2.0], [3.0, 4.0]))


def one(task):
    vf, rmin, penal = task
    r = simp.simp_optimise(vf, rmin_mm=rmin, penal=penal)
    ev = hf.Problem(hf.GIRDER, hf.H_FINE).evaluate(r["phi"], buckling=True)
    return dict(vf_target=vf, rmin_mm=rmin, penal=penal, vf=ev["vf"], connected=bool(ev["connected"]),
                compliance=float(ev["compliance"]), blf=float(ev.get("blf", np.nan)), umax=float(ev["umax"]),
                n_iter=r["n_iter"], greyness=r["greyness"]), r["phi"]


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        res = list(ex.map(one, GRID))
    out = HERE / "results" / "simp"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "multistart_designs.npz", phi=np.stack([p for _, p in res]).astype(bool))
    json.dump(dict(rows=[d for d, _ in res], wall_s=time.time() - t0), open(out / "multistart.json", "w"), indent=1)
    print("done", time.time() - t0)
