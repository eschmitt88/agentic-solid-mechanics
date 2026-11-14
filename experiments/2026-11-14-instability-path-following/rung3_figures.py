"""Rung 3 — collect own-solver and CalculiX results, grade, and draw figures."""
import glob
import json
import pathlib

import numpy as np

from plotstyle import C, setup
from rung3_cylinder import XIS, R, T, bh_longwave, koiter_asymptotic_memory, koiter_upper_bound, nasa_sp8007

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"


def ccx_curve(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    s, P = d[:, 0], d[:, 1]
    k = np.diff(P) / np.diff(s)
    k0 = np.median(k[: max(3, len(k) // 10)])
    knee = None
    for i in range(len(k)):
        if k[i] < 0.25 * k0:           # tangent stiffness has dropped to a quarter: the shell has buckled
            knee = i
            break
    return s, P, k, knee


def ccx_buckling_load(path, completed):
    """Operational definition (written 21:20, after the xi = 0.1 and 0.3 runs with seeds 1e-3 and 1e-2 had
    been read, before the xi = 0, the extended xi = 1.0 and the seed 5e-2 runs were read):
    Newton failure before the end of the step -> last converged (maximum) load = collapse;
    otherwise the first local maximum of the load; otherwise the knee where the tangent
    stiffness has dropped below a quarter of its initial value."""
    s, P, k, knee = ccx_curve(path)
    if not completed:
        i = int(np.argmax(P))
        return float(P[i]), float(s[i]), "collapse (Newton failed, load at last converged increment)"
    for i in range(1, len(P) - 1):
        if P[i] > P[i + 1] and P[i] >= P[i - 1]:
            return float(P[i]), float(s[i]), "first local load maximum, then stable buckled state"
    if knee is not None:
        return float(P[knee]), float(s[knee]), "knee (stiffness < 1/4 of initial), load keeps rising"
    return float(P[-1]), float(s[-1]), "no buckling detected in the range run"


RUNS = [  # (xi, tag, seed)  -- first entry per xi is the run planned in config.yaml (seed 1e-3 t)
    (0.0, "_a", 1e-3), (0.1, "_a", 1e-3), (0.3, "_long", 1e-3), (1.0, "_b", 1e-3),
    (0.1, "_b", 1e-2), (0.1, "_c", 5e-2), (0.3, "_t1e-2", 1e-2),
    (0.0, "_full", 1e-2), (0.1, "_full", 1e-2), (0.3, "_full", 1e-2), (1.0, "_full", 1e-2),   # full 360 deg model
]


def main():
    own = json.loads((RES / "r3_cylinder.json").read_text())
    buck = json.loads((RES / "r3_ccx_buckle.json").read_text())
    ss = {r["xi"]: r for r in own["knockdown_ss"]}
    cell = {r["xi"]: r for r in own["knockdown_cell"]}
    # the infinitely long shell admits modes even about crests (xi > 0) and about troughs (xi < 0): take the lower
    ph = json.loads((RES / "r3_cell_phase.json").read_text())
    neg = {r["xi"]: r for r in ph["four_wavelength_cell_negative_xi"]}
    for xi in XIS:
        c = cell[xi]
        c["rho_pos"], c["rho_neg"], c["n_pos"], c["n_neg"] = c["rho"], neg[xi]["rho"], c["n_crit"], neg[xi]["n_crit"]
        c["rho_by_mesh_min"] = [min(a["lam_c"], b) for a, b in zip(c["by_mesh"], neg[xi]["by_mesh"])]
        if c["rho_neg"] < c["rho_pos"]:
            c["rho"], c["n_crit"] = c["rho_neg"], c["n_neg"]
        c["abs_diff"] = abs(c["rho"] - c["koiter_upper_bound"])
    ccx = []
    for xi, tag, seed in RUNS:
        f = RES / f"r3_ccx_xi{xi}_p8{tag}.csv"
        jf = RES / f"r3_ccx_collapse_xi{xi}_p8{tag}.json"
        if not f.exists() or not jf.exists():
            continue
        meta = json.loads(jf.read_text())
        Pb, sb, how = ccx_buckling_load(f, meta["completed_step"])
        ref = ss[xi]["rho_even_n"]
        ccx.append({"xi": xi, "seed_over_t": seed, "file": f.name, "P_buckle": Pb, "shortening": sb, "how": how,
                    "own_ss_even_n": ref, "own_ss_all_n": ss[xi]["rho"], "rel_diff": Pb / ref - 1,
                    "completed_step": meta["completed_step"], "n_increments": meta["n_increments"], "seconds": meta["seconds"]})
    primary = [c for c in ccx if c["seed_over_t"] == 1e-3]
    fullm = [c for c in ccx if "_full" in c["file"]]
    for c in fullm:                      # the full model admits every n: compare with the all-n value
        c["rel_diff_all_n"] = c["P_buckle"] / c["own_ss_all_n"] - 1
    grade = {
        "3a_own_linear": own["linear_buckling"][-1]["lam_cell"],
        "3a_own_linear_by_mesh": [(l["el_per_wavelength"], l["lam_cell"]) for l in own["linear_buckling"]],
        "3a_own_linear_ss": own["linear_buckling"][-1]["lam_ss"],
        "3b_ccx_linear_by_mesh": [(b["per"], b["n_elements"], b["N_cr_over_N_cl"][0]) for b in buck],
        "3c": [{"xi": xi, "rho": cell[xi]["rho"], "rho_by_mesh": cell[xi]["rho_by_mesh_min"],
                "rho_ends_on_crest": cell[xi]["rho_pos"], "rho_ends_on_trough": cell[xi]["rho_neg"],
                "koiter": cell[xi]["koiter_upper_bound"], "abs_diff": cell[xi]["abs_diff"],
                "bh_longwave": cell[xi]["bh_longwave"], "n_crit": cell[xi]["n_crit"],
                "memory_formula": cell[xi]["koiter_asymptotic_from_memory"]} for xi in XIS],
        "3d_runs": ccx, "3d_primary_max_abs_rel_diff": max(abs(c["rel_diff"]) for c in primary) if primary else None,
        "nasa_gamma": float(nasa_sp8007(R / T)),
        "ss": [{"xi": x, "rho": ss[x]["rho"], "n": ss[x]["n_crit"], "rho_even_n": ss[x]["rho_even_n"]} for x in [0.0] + XIS],
        "long_cell": own["knockdown_long_cell"], "one_wavelength_cell": ph["one_wavelength_cell"],
    }
    grade["3c_max_abs_diff"] = max(g["abs_diff"] for g in grade["3c"])

    def crossing(xs, ys, y0):
        for i in range(len(xs) - 1):
            if (ys[i] - y0) * (ys[i + 1] - y0) <= 0:
                return float(xs[i] + (xs[i + 1] - xs[i]) * (y0 - ys[i]) / (ys[i + 1] - ys[i]))
        return None
    grade["xi_at_nasa_cell"] = crossing(XIS, [cell[x]["rho"] for x in XIS], grade["nasa_gamma"])
    grade["xi_at_nasa_ss"] = crossing([0.0] + XIS, [ss[x]["rho"] for x in [0.0] + XIS], grade["nasa_gamma"])
    (RES / "r3_grade.json").write_text(json.dumps(grade, indent=1))
    print(json.dumps({k: v for k, v in grade.items() if k not in ("3c", "ss")}, indent=1))
    for g in grade["3c"]:
        print(g)

    plt = setup()
    # ---------------- knockdown curve
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    xf = np.concatenate([[1e-4], np.linspace(0.005, 1.0, 120)])
    ax.plot(xf, [koiter_upper_bound(x)[0] for x in xf], color=C["ref"], lw=1.6, label="Koiter 1963 upper bound (theory)")
    ax.plot(xf, [bh_longwave(x) for x in xf], color=C["mute"], lw=1.2, ls="--", label="Budiansky–Hutchinson long-wave limit (theory)")
    ax.plot([0.0] + XIS, [1.0] + [cell[x]["rho"] for x in XIS], marker="o", color=C["blue"], label="own solver, infinitely long shell")
    ax.plot([0.0] + XIS, [ss[x]["rho"] for x in [0.0] + XIS], marker="s", color=C["orange"], label="own solver, simply supported ends")
    ax.plot([c["xi"] for c in primary], [c["P_buckle"] for c in primary], ls="none", marker="D", ms=9, mfc="none", mew=2,
            color=C["aqua"], label="CalculiX as planned (quarter model, tiny seed)")
    if fullm:
        ax.plot([c["xi"] for c in fullm], [c["P_buckle"] for c in fullm], ls="none", marker="D", ms=8,
                color=C["aqua"], label="CalculiX corrected (full model, seed 0.01 t)")
    g = grade["nasa_gamma"]
    ax.axhline(g, color=C["bad"], lw=1.3, ls=":")
    ax.text(0.99, g + 0.012, f"NASA SP-8007 design factor for R/t = 200: {g:.2f}", ha="right", fontsize=10)
    ax.set_xlabel("imperfection amplitude / wall thickness"); ax.set_ylabel("buckling load / classical load")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05)
    ax.set_title("Knockdown curve, axially compressed cylinder, R/t = 200", loc="left"); ax.legend(loc="upper right", fontsize=9.5)
    fig.savefig(RES / "r3_knockdown.png")

    # ---------------- CalculiX load-shortening curves
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    cols = {0.0: C["violet"], 0.1: C["blue"], 0.3: C["orange"], 1.0: C["aqua"]}
    for c in primary:
        s, P, k, knee = ccx_curve(RES / c["file"])
        ax.plot(s, P, color=cols[c["xi"]], label=f"CalculiX, imperfection {c['xi']:g} t")
        ax.plot([0, 1.3], [c["own_ss_even_n"]] * 2, color=cols[c["xi"]], lw=1, ls="--")
        ax.plot(c["shortening"], c["P_buckle"], marker="D", ms=8, mfc="none", mew=2, color=cols[c["xi"]])
    ax.set_xlim(0, 1.3); ax.set_ylim(0, 1.0)
    ax.set_xlabel("end shortening / classical end shortening"); ax.set_ylabel("axial load / classical load")
    ax.set_title("CalculiX, model as planned; dashed = own solver", loc="left", fontsize=12); ax.legend(loc="upper left", fontsize=9.5)
    ax = axes[1]
    axp = RES / "r3_axisym_path_ss_xi0.1.csv"
    if axp.exists():
        a = np.loadtxt(axp, delimiter=",", skiprows=1)
        ax.plot(a[:, 0], a[:, 1], color=C["mute"], lw=5, alpha=0.4, label="own solver: axisymmetric path (unstable above 0.58)")
    for c, col in zip([c for c in ccx if c["xi"] == 0.1 and "_full" not in c["file"]], [C["blue"], C["orange"], C["aqua"]]):
        s, P, k, knee = ccx_curve(RES / c["file"])
        ax.plot(s, P, color=col, label=f"CalculiX, seed {c['seed_over_t']:g} t → {c['P_buckle']:.3f}")
        ax.plot(c["shortening"], c["P_buckle"], marker="D", ms=8, mfc="none", mew=2, color=col)
    ax.axhline(ss[0.1]["rho_even_n"], color=C["ref"], lw=1, ls="--"); ax.text(0.42, ss[0.1]["rho_even_n"] + 0.008, f"own bifurcation load {ss[0.1]['rho_even_n']:.3f}", fontsize=9.5)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.38, 0.7)
    ax.set_xlabel("end shortening / classical end shortening"); ax.set_ylabel("axial load / classical load")
    ax.set_title("Imperfection 0.1 t: Newton can step over the bifurcation", loc="left", fontsize=12); ax.legend(loc="lower right", fontsize=9)
    fig.savefig(RES / "r3_ccx_paths.png")


if __name__ == "__main__":
    main()
