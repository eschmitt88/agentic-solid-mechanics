"""Assemble metrics.json from the per-rung result files and apply the
pre-registered tolerances (README.md, 'Pre-registered pass criteria')."""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"
J = lambda n: json.loads((RES / n).read_text())


def main():
    tr, bm, cl, bi = J("r1_truss.json"), J("r1_beams.json"), J("r1_ccx_lee.json"), J("r2_biot.json")
    g3 = J("r3_grade.json")
    m = {}

    def crit(cid, name, value, ref, err, tol, passed, note=""):
        m[cid] = {"name": name, "value": value, "reference": ref, "error": err, "tolerance": tol,
                  "pass": bool(passed), "note": note}

    s = tr["summary"]
    crit("1a", "von Mises truss limit load / closed form", s["1a_limit_load"], s["1a_ref"], s["1a_rel_err"], 1e-3, s["1a_rel_err"] <= 1e-3)
    crit("1b", "von Mises truss bifurcation load / closed form (+ branch switch)", s["1b_bif_load"], s["1b_ref"], s["1b_rel_err"], 1e-3,
         s["1b_rel_err"] <= 1e-3 and s["1b_branch_circle_err"] < 1e-6 and s["1b_branch_max_sideways_over_h"] > 0.1,
         note=f"switched branch follows closed-form circle to {s['1b_branch_circle_err']:.1e}")
    L = bm["lee"]
    crit("1c", "Lee frame first limit load P L^2/EI (160 elements)", L["P1_finest"], 18.552, L["P1_rel_err_finest"], 0.01,
         L["P1_rel_err_finest"] <= 0.01, note=f"Richardson {L['P1_extrap']:.4f}, order {L['P1_order']:.2f}")
    path = np.loadtxt(RES / "r1_lee_path.csv", delimiter=",", skiprows=1)
    v, P = path[:, 0], path[:, 2]
    k = [i for i in range(len(P) - 1) if (P[i] - 31.887) * (P[i + 1] - 31.887) <= 0][-1]
    v_at_ref = float(v[k] + (v[k + 1] - v[k]) * (31.887 - P[k]) / (P[k + 1] - P[k]))
    crit("1d", "Lee frame load at u_y/L = 0.784", L["P784_by_mesh"][-1], 31.887, L["P784_rel_err_finest"], 0.02,
         L["P784_rel_err_finest"] <= 0.02,
         note=f"ill-conditioned criterion (curve nearly vertical); post-hoc: deflection at P = 31.887 is {v_at_ref:.4f} vs 0.784 "
              f"({(v_at_ref / 0.784 - 1) * 100:+.2f} %)")
    m["1d"]["posthoc_deflection_at_reference_load"] = v_at_ref
    A = bm["arch"]
    crit("1e", "deep arch maximum load P R^2/EI (320 elements)", A["Pmax_finest"], 8.972922, A["rel_err_finest"], 0.01,
         A["rel_err_finest"] <= 0.01, note=f"Richardson {A['Pmax_extrap']:.5f}, order {A['order']:.2f}")
    ccx = cl["lee_b32r_8"]["Pmax"]
    d = abs(ccx / L["P1_extrap"] - 1)
    crit("1f", "CalculiX vs own solver, Lee frame limit load", ccx, L["P1_extrap"], d, 0.02, d <= 0.02,
         note="CalculiX B32R, displacement control; by mesh " + ", ".join(f"{v_['Pmax']:.3f}" for v_ in cl.values()))

    conv = bi["stream_function"]["mesh_convergence"]
    f = conv[-1]
    crit("2a", "Biot critical strain, finest mesh (stream-function elements)", f["eps_c"], bi["reference"]["eps_W"],
         abs(f["rel_err_vs_biot"]), 0.01, abs(f["rel_err_vs_biot"]) <= 0.01,
         note="first attempt with Taylor-Hood mixed elements failed: detected 0.3216 (shift-invert) / 0.03 (inertia count)")
    m["2a"]["attempt1_taylor_hood"] = {"naive": bi["taylor_hood"]["naive_shift_invert"],
                                       "inertia_first_unstable_strain": [t["first_unstable_strain_on_grid"] for t in bi["taylor_hood"]["inertia"]]}
    crit("2b", "Biot threshold mesh convergence monotone (6 levels)", [c["eps_c"] for c in conv], None,
         [abs(c["rel_err_vs_biot"]) for c in conv], "monotone", bi["stream_function"]["monotone"],
         note=f"observed order {bi['stream_function']['observed_order_last']:.2f}")

    crit("3a", "classical load, own linear eigenvalue analysis / formula", g3["3a_own_linear"], 1.0, abs(g3["3a_own_linear"] - 1), 0.03,
         abs(g3["3a_own_linear"] - 1) <= 0.03, note=f"simply supported ends: {g3['3a_own_linear_ss']:.5f}")
    b = g3["3b_ccx_linear_by_mesh"]
    crit("3b", "classical load, CalculiX *BUCKLE (finest mesh) / formula", b[-1][2], 1.0, abs(b[-1][2] - 1), 0.05, abs(b[-1][2] - 1) <= 0.05,
         note="by mesh: " + ", ".join(f"{x[2]:.4f}" for x in b))
    crit("3c", "knockdown curve vs Koiter upper bound, max |difference| over 7 amplitudes", [(r["xi"], r["rho"]) for r in g3["3c"]],
         [(r["xi"], r["koiter"]) for r in g3["3c"]], g3["3c_max_abs_diff"], 0.05, g3["3c_max_abs_diff"] <= 0.05,
         note="own solver always below the bound, above the Budiansky-Hutchinson long-wave limit")
    planned = [r for r in g3["3d_runs"] if r["seed_over_t"] == 1e-3 and "_full" not in r["file"]]
    worst = max(abs(r["rel_diff"]) for r in planned)
    crit("3d", "CalculiX nonlinear buckling load vs own solver (as planned: quarter model, seed 1e-3 t)",
         [(r["xi"], r["P_buckle"]) for r in planned], [(r["xi"], r["own_ss_even_n"]) for r in planned],
         [(r["xi"], r["rel_diff"]) for r in planned], 0.10, worst <= 0.10,
         note="misses traced to Newton stepping over the bifurcation (small seed) and to rotational symmetry constraints stiffening the quarter model")
    full = [r for r in g3["3d_runs"] if "_full" in r["file"]]
    if full:
        m["3d"]["corrected_full_model"] = [{"xi": r["xi"], "seed_over_t": r["seed_over_t"], "ccx": r["P_buckle"], "own": r["own_ss_all_n"],
                                            "rel_diff": r["P_buckle"] / r["own_ss_all_n"] - 1, "how": r["how"]} for r in full]
    m["informational"] = {
        "nasa_sp8007_gamma_R_over_t_200": g3["nasa_gamma"],
        "imperfection_amplitude_at_nasa_factor_infinite_shell": g3["xi_at_nasa_cell"],
        "imperfection_amplitude_at_nasa_factor_simply_supported": g3["xi_at_nasa_ss"],
        "perfect_shell_nonlinear_bifurcation_simply_supported": g3["ss"][0]["rho"],
        "lee_frame_second_limit_load": L["P2_finest"],
    }
    m["summary"] = {"n_criteria": sum(1 for v_ in m.values() if isinstance(v_, dict) and "pass" in v_), "n_pass": sum(1 for k_, v_ in m.items() if isinstance(v_, dict) and v_.get("pass") is True),
                    "failed": [k_ for k_, v_ in m.items() if isinstance(v_, dict) and v_.get("pass") is False]}
    (HERE / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    for k_, v_ in m.items():
        if isinstance(v_, dict) and "pass" in v_:
            print(k_, "PASS" if v_["pass"] else "FAIL", v_["name"], "| err", v_["error"] if not isinstance(v_["error"], list) else "...", "| tol", v_["tolerance"])
    print(m["summary"])


if __name__ == "__main__":
    main()
