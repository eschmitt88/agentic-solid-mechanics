"""Compute the FE-true reference optimum for the design-loop trial (no LLM).

Establishes the ground truth the agent is graded against: the minimum-mass
feasible (b, h), found with FE in the loop. Also prints the Euler-Bernoulli
analytic optimum for comparison (the gap is the shear effect the agent must
discover by closing the FE loop).
"""
from __future__ import annotations

import json
from pathlib import Path

import design as D

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def eb_optimum(prob: D.DesignProblem) -> dict:
    """Analytic EB optimum: deflection binding at b=b_min, h=(K_d/b)^(1/3)."""
    p = prob.p
    K_d = 4 * p["tip_load_n"] * p["length_m"] ** 3 / (p["youngs_modulus_pa"] * prob.delta_allow)
    b = prob.b_bounds[0]
    h = (K_d / b) ** (1 / 3)
    beam = prob.beam(b, h)
    return {"b": b, "h": h, "mass_kg": beam.mass_kg(),
            "eb_deflection_m": beam.eb_deflection_m(),
            "nominal_stress_pa": beam.nominal_bending_stress_pa()}


def main() -> None:
    prob = D.load(HERE / "config.yaml")
    wd = RESULTS / "_ref"

    eb = eb_optimum(prob)
    print(f"EB optimum:  b={eb['b']*1e3:.1f} mm  h={eb['h']*1e3:.2f} mm  "
          f"mass={eb['mass_kg']:.3f} kg  (EB defl={eb['eb_deflection_m']*1e3:.3f} mm)")

    ref = prob.reference_optimum(wd)
    opt = ref["optimum"]
    print("\nFE-true scan (min h meeting deflection at each b):")
    for r in ref["scan"]:
        print(f"  b={r['b']*1e3:5.1f} mm  h={r['h']*1e3:7.2f} mm  "
              f"mass={r['mass_kg']:6.3f} kg  sigma={r['nominal_stress_pa']/1e6:6.1f} MPa  "
              f"{'(stress OK)' if r['stress_ok'] else '(STRESS VIOLATED)'}")
    print(f"\nFE-true optimum: b={opt['b']*1e3:.1f} mm  h={opt['h']*1e3:.2f} mm  "
          f"mass={opt['mass_kg']:.4f} kg")

    out = {"eb_optimum": eb, "fe_optimum": opt, "fe_scan": ref["scan"],
           "constraints": {"delta_allow_m": prob.delta_allow,
                           "sigma_allow_pa": prob.sigma_allow},
           "bounds": {"b": prob.b_bounds, "h": prob.h_bounds}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reference.json").write_text(json.dumps(out, indent=2))
    (HERE / "metrics.json").write_text(json.dumps({
        "fe_optimum_mass_kg": opt["mass_kg"],
        "fe_optimum_b_m": opt["b"], "fe_optimum_h_m": opt["h"],
        "eb_optimum_mass_kg": eb["mass_kg"],
        "eb_vs_fe_h_gap_pct": (opt["h"] / eb["h"] - 1) * 100,
    }, indent=2))
    print("Wrote results/reference.json and metrics.json")


if __name__ == "__main__":
    main()
