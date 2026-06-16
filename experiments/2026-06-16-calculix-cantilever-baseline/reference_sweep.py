"""Deterministic mesh-convergence reference (no LLM).

Generates the cantilever deck at each mesh level in config.yaml, runs CalculiX,
parses tip deflection + max von Mises, and grades against the Euler-Bernoulli
analytical solution. Writes results/reference.json and updates metrics.json.

This establishes the ground truth that the agentic harness is graded against,
and is itself a sanity check that the solver/parser/deck generator all work.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

import cantilever as C

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    p = cfg["problem"]
    spec = C.Spec(
        length_m=p["length_m"], width_m=p["width_m"], height_m=p["height_m"],
        youngs_modulus_pa=p["youngs_modulus_pa"], poisson_ratio=p["poisson_ratio"],
        tip_load_n=p["tip_load_n"],
    )
    env = cfg["solver"]["micromamba_env"]
    ccx_bin = cfg["solver"]["ccx_bin"]
    ref = C.analytical(spec)
    print(f"Analytical: delta={ref['tip_deflection_m']:.6e} m  "
          f"sigma={ref['max_bending_stress_pa']:.4e} Pa")

    sweep = []
    for n in cfg["mesh_levels"]:
        job = RESULTS / f"cantilever_n{n}"
        meta = C.generate_inp(spec, n, job.with_suffix(".inp"))
        proc = C.run_ccx(job, env=env, ccx_bin=ccx_bin)
        dat = job.with_suffix(".dat")
        ok = proc.returncode == 0 and dat.exists()
        row = {"n": n, **meta, "ccx_returncode": proc.returncode, "ran": ok}
        if ok:
            defl = C.parse_displacement(dat)
            vm = C.parse_stress(dat)
            g = C.grade(spec, defl, vm,
                        cfg["tol_deflection_rel"], cfg["tol_stress_rel"])
            row.update({
                "tip_deflection_m": defl,
                "deflection_rel_err": g["deflection_rel_err"],
                "max_vm_stress_pa": vm,
                "stress_rel_err": g.get("stress_rel_err"),
            })
            print(f"n={n:2d} elems={meta['n_elements']:5d}  "
                  f"defl={defl:.6e} m  err={g['deflection_rel_err']*100:5.2f}%  "
                  f"vM_max={vm:.3e} Pa")
        else:
            print(f"n={n:2d}  CCX FAILED rc={proc.returncode}")
            row["stderr_tail"] = proc.stderr[-500:]
        sweep.append(row)

    finest = next((r for r in reversed(sweep) if r.get("ran")), None)
    out = {
        "spec": vars(spec),
        "analytical": ref,
        "sweep": sweep,
        "finest_deflection_rel_err": finest and finest.get("deflection_rel_err"),
        "finest_stress_rel_err": finest and finest.get("stress_rel_err"),
        "deflection_converges": _converges([r.get("tip_deflection_m") for r in sweep if r.get("ran")]),
    }
    (RESULTS / "reference.json").write_text(json.dumps(out, indent=2))

    # metrics.json = the search signal (validation-equivalent for trial 1).
    (HERE / "metrics.json").write_text(json.dumps({
        "finest_deflection_rel_err": out["finest_deflection_rel_err"],
        "finest_stress_rel_err": out["finest_stress_rel_err"],
        "deflection_converges": out["deflection_converges"],
        "n_mesh_levels_ran": sum(1 for r in sweep if r.get("ran")),
    }, indent=2))
    print(f"\nFinest deflection err: {out['finest_deflection_rel_err']}")
    print(f"Monotone convergence: {out['deflection_converges']}")
    print("Wrote results/reference.json and metrics.json")


def _converges(values: list[float], final_tol: float = 0.01) -> bool:
    """Mesh convergence: successive relative change shrinks and the last
    refinement step moves the solution by < final_tol (default 1%).

    This is the correct test — the 3D FEM converges to the true solution
    (slightly above Euler-Bernoulli due to shear), not to EB itself, so we
    check that the *solution stabilizes*, not that error-vs-EB → 0.
    """
    vals = [v for v in values if v is not None]
    if len(vals) < 3:
        return False
    changes = [abs(b - a) / abs(b) for a, b in zip(vals, vals[1:])]
    non_increasing = all(b <= a + 1e-9 for a, b in zip(changes, changes[1:]))
    return non_increasing and changes[-1] < final_tol


if __name__ == "__main__":
    main()
