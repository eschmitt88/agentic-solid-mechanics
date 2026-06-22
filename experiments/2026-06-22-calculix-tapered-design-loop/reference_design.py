"""FE-true reference optimum for the tapered design loop (no LLM).

Searches the taper ratio (width fixed at its lower bound; root height bisected
to meet the FE deflection cap) for the minimum-mass feasible tapered section.
Compares against the prismatic optimum from trial 2 (8.519 kg) to quantify the
material the taper saves, and prints the Euler-Bernoulli integral vs FE gap.
"""
from __future__ import annotations

import json
from pathlib import Path

import design as D

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PRISMATIC_OPT_KG = 8.519   # trial 2 FE-true optimum, same problem (P, L, delta_allow)


def main() -> None:
    prob = D.load(HERE / "config.yaml")
    ref = prob.reference_optimum(RESULTS / "_ref")
    opt = ref["optimum"]

    print("Taper-ratio scan (b at lower bound, root height bisected to the deflection cap):")
    for r in ref["scan"]:
        print(f"  r=h_tip/h_root={r['taper_ratio']:.2f}  h_root={r['h_root']*1e3:6.1f} mm  "
              f"h_tip={r['h_tip']*1e3:6.1f} mm  mass={r['mass_kg']:6.3f} kg  "
              f"sigma_max={r['max_nominal_stress_pa']/1e6:6.1f} MPa  "
              f"{'OK' if r['stress_ok'] else 'STRESS!'}")

    beam = prob.beam(opt["b"], opt["h_root"], opt["h_tip"])
    print(f"\nFE-true tapered optimum: b={opt['b']*1e3:.1f} mm  "
          f"h_root={opt['h_root']*1e3:.1f} mm  h_tip={opt['h_tip']*1e3:.1f} mm  "
          f"mass={opt['mass_kg']:.4f} kg")
    print(f"  vs prismatic optimum {PRISMATIC_OPT_KG} kg  ->  "
          f"{(1 - opt['mass_kg']/PRISMATIC_OPT_KG)*100:.1f}% lighter")
    print(f"  EB-integral deflection at optimum: {beam.eb_deflection_integral_m()*1e3:.3f} mm "
          f"(cap {prob.delta_allow*1e3:.1f} mm) — gap vs FE is the shear effect")
    print(f"  b_max(0.5-taper) mass check: {ref['b_max_check_mass_kg']:.3f} kg "
          f"(> optimum -> confirms b=b_min corner)")

    out = {"tapered_optimum": opt, "scan": ref["scan"],
           "prismatic_optimum_kg": PRISMATIC_OPT_KG,
           "b_max_check_mass_kg": ref["b_max_check_mass_kg"],
           "eb_integral_deflection_at_opt_m": beam.eb_deflection_integral_m(),
           "constraints": {"delta_allow_m": prob.delta_allow, "sigma_allow_pa": prob.sigma_allow}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reference.json").write_text(json.dumps(out, indent=2))
    (HERE / "metrics.json").write_text(json.dumps({
        "tapered_optimum_mass_kg": opt["mass_kg"],
        "tapered_optimum_h_root_m": opt["h_root"], "tapered_optimum_h_tip_m": opt["h_tip"],
        "prismatic_optimum_mass_kg": PRISMATIC_OPT_KG,
        "taper_mass_saving_pct": (1 - opt["mass_kg"] / PRISMATIC_OPT_KG) * 100,
    }, indent=2))
    print("\nWrote results/reference.json and metrics.json")


if __name__ == "__main__":
    main()
