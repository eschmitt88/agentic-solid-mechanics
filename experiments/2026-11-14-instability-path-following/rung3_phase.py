"""Rung 3 supplement — the other imperfection phase.

Symmetry boundary conditions on a cell admit only modes that are even about
the cell ends.  With xi > 0 the ends sit on outward crests of the ripple, with
xi < 0 on inward troughs; the infinitely long shell admits both families, so
its bifurcation load is the smaller of the two.  (Found late: the first
knockdown runs used xi > 0 only, which excludes Koiter's own mode.)

Also: a one-wavelength cell, which admits only modes with the axial period of
Koiter's trial mode (Budiansky & Hutchinson's 'k = 1' class) and should
therefore sit just below Koiter's upper bound.
"""
import json
import pathlib

from rung3_cylinder import ELL, NS, Q0, XIS, bifurcation_load, koiter_upper_bound

RES = pathlib.Path(__file__).parent / "results"
out = {"four_wavelength_cell_negative_xi": [], "one_wavelength_cell": []}
for xi in XIS:
    row = {"xi": xi, "by_mesh": []}
    for per in (8, 16, 32):
        r, *_ = bifurcation_load(4 * per, 4 * ELL, -xi, "cell", NS)
        row["by_mesh"].append(r["lam_c"])
        if per == 32:
            row["rho"], row["n_crit"] = r["lam_c"], r.get("n_crit")
    out["four_wavelength_cell_negative_xi"].append(row)
    print("4-cell -xi", row, flush=True)
    ref, g = koiter_upper_bound(xi)
    a, *_ = bifurcation_load(32, ELL, xi, "cell", NS)
    b, *_ = bifurcation_load(32, ELL, -xi, "cell", NS)
    b2, *_ = bifurcation_load(64, ELL, -xi, "cell", NS)
    row1 = {"xi": xi, "rho_pos": a["lam_c"], "n_pos": a.get("n_crit"), "rho_neg": b["lam_c"], "n_neg": b.get("n_crit"),
            "rho_neg_fine": b2["lam_c"], "koiter": ref, "koiter_n": g * Q0, "koiter_minus_rho": ref - min(a["lam_c"], b["lam_c"])}
    out["one_wavelength_cell"].append(row1)
    print("1-cell", row1, flush=True)
(RES / "r3_cell_phase.json").write_text(json.dumps(out, indent=1, default=float))
