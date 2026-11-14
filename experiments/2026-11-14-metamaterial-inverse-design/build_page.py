"""Assemble metrics.json and the review page docs/qa/trial-9-metamaterial/index.html from results/*.json."""
import json
import shutil
from pathlib import Path

import numpy as np

import qiu
from validate_qiu_lib import pts
import fem

R = Path("results")
PAGE = Path("../../docs/qa/trial-9-metamaterial")
PAGE.mkdir(parents=True, exist_ok=True)
J = lambda f: json.load(open(R / f))

r0 = J("rung0_qiu_validation.json")
r12 = J("rung0_qiu_q12_supplementary.json")
D, Fs = qiu.series_curve(6.0)
c = fem.E_PS / 12 * 6 / 1e6
ser = pts(D * 6, Fs * c, 6.0)
jx = r0["jax_100x2"]
ref = r0["qiu_eq_2_47_2_48"]
rung0 = dict(
    source="Qiu, PhD thesis MIT 2003, ch.2 eqs 2.9-2.48 (https://pergatory.mit.edu/research/jqiu/thesisscreen.pdf); doi:10.1109/JMEMS.2004.825308",
    case="l=100 mm, t=1 mm, h=6 mm, Q=6, plane strain",
    jax=dict(f_top=jx["f_top"], f_bot=jx["f_bot"], d_top=jx["d_top"], d_bot=jx["d_bot"], d_end=jx["d_end"]),
    qiu_closed_form=ref,
    qiu_series_Q6={k: r0["qiu_series_Q6"][k] for k in ("f_top", "f_bot", "d_top", "d_bot", "d_end")},
    rel_err_vs_closed_form=r0["rel_err_vs_eq_2_47"],
    pos_err_over_h_vs_closed_form={k: r0["abs_err_over_h_vs_eq_2_48"][k] for k in ("d_top", "d_bot", "d_end")},
    rel_err_vs_series=r0["rel_err_vs_series"],
    pos_err_over_h_vs_series={k: r0["abs_err_over_h_vs_series"][k] for k in ("d_top", "d_bot", "d_end")},
    nrmse_jax_vs_calculix=r0["nrmse_jax_vs_ccx_imperfect"],
    supplementary_Q12_rel_err_vs_closed_form=r12["rel_err_vs_eq_2_47"],
    supplementary_Q12_pos_err_over_h={k: r12["abs_err_over_h_vs_eq_2_48"][k] for k in ("d_top", "d_bot", "d_end")},
)
c0 = rung0["rel_err_vs_closed_form"]; pp = rung0["pos_err_over_h_vs_closed_form"]
rung0["pass_preregistered"] = dict(
    f_top=abs(c0["f_top"]) <= 0.05, f_bot=abs(c0["f_bot"]) <= 0.05, d_top=abs(pp["d_top"]) <= 0.05, d_bot=abs(pp["d_bot"]) <= 0.05,
    d_end=abs(pp["d_end"]) <= 0.05, jax_vs_calculix=rung0["nrmse_jax_vs_calculix"] <= 0.02)
rung0["pass_all"] = all(rung0["pass_preregistered"].values())

metrics = dict(rung0=rung0, rung1={}, material_check_max_rel_diff=max(x["max_rel_diff"] for x in J("material_check.json")))
sway = J("sway_check.json")
for T in ("T1", "T2", "T3"):
    v = J(f"verify_{T}.json"); o = J(f"opt_{T}.json"); rb = J(f"robustness_{T}.json")
    metrics["rung1"][T] = dict(
        optimizer_nrmse=v["optimizer_nrmse_jax40x2"], calculix_nrmse=v["verdict"]["nrmse"], tolerance=v["verdict"]["tol"], passed=v["verdict"]["pass"],
        secondary={k: v["verdict"][k] for k in v["verdict"] if k not in ("nrmse", "tol", "pass")},
        calculix_fine_mesh_nrmse=v["verdict_fine_mesh"]["nrmse"], solver_gap_jax_vs_calculix=v["gap_jax40x2_vs_ccx_t3"],
        mesh_conv_jax_40x2_vs_80x4=v["mesh_conv_jax_40x2_vs_80x4"], mesh_conv_calculix_t3_vs_t6=v["mesh_conv_ccx_t3_vs_t6"],
        all_starts_optimizer_nrmse={s: r["nrmse"] for s, r in o.items()}, p=v["p"],
        sway_2pct_side_load=dict(max_drift_mm=sway[T]["max_abs_ux_mm"], curve_change=sway[T]["change_vs_fixed"]),
        tolerance_sensitivity=dict(median_nrmse=rb["median"], max_nrmse=rb["max"], frac_within_tol=rb["frac_within_tol"]))
sn = J("sanity_recovery.json")
metrics["sanity_check_in_space_recovery"] = {k: dict(nrmse=r["nrmse"], max_abs_param_err=r["max_abs_param_err"]) for k, r in sn["runs"].items()}
g = J("gradcheck.json")["fine"]
d = np.array(g["d"]); Jm = np.array(g["J"]); dc = g["d_c"]
errs = {}
for j, nm in ((0, "a1"), (4, "log_t0"), (6, "b2"), (1, "a2")):
    fd = np.array(g["fd"][str(j)]["1e-06"]); e = np.abs(fd - Jm[:, j]) / np.max(np.abs(Jm[:, j]))
    errs[nm] = dict(max_rel_err_excluding_dc=float(e[np.abs(d - dc) > 1e-9].max()), rel_err_at_dc=float(e[np.abs(d - dc) <= 1e-9].max()))
i_c = int(np.argmin(np.abs(d - dc)))
metrics["gradient_check"] = dict(d_c_mm=dc, adjoint_vs_same_branch_fd=errs, lam_min_at_dc=g["lam_min"][i_c],
                                 jump_across_dc={nm: [float(Jm[i_c - 1, j]), float(Jm[i_c + 1, j])] for j, nm in ((0, "a1"), (4, "log_t0"), (6, "b2"))},
                                 dF_da2={"at_dc": float(Jm[i_c, 1]), "dc+0.02": float(Jm[i_c + 1, 1]), "dc+0.3": float(Jm[i_c + 15, 1]), "dc+4": float(Jm[-1, 1]),
                                         "rerun_central_fd_everywhere": float(np.max(np.abs(g["fd_restart"]["1"])))})
bi = None
if (R / "verify_bistable.json").exists():
    bi = J("verify_bistable.json"); db = J("design_bistable.json")
    fj = bi["jax_40x2"]["features"]
    ex = bi.get("extra", {})
    # pre-registered verifier: CPE6 with element size <= t_min/3.  The t_min/3 run aborted (see log.md); the finer
    # t_min/4 and t_min/6 runs completed.  Graded on the coarsest completed as-designed mesh.
    cands = [("t_min/3", bi["ccx_t3"]["features"]), ("t_min/4", ex.get("ccx_t4", {}).get("features")), ("t_min/6", bi["ccx_t6"]["features"])]
    vmesh, fc = next(((n, f) for n, f in cands if f), (None, None))
    rel = bi.get("ccx_release", {})
    need = [k for k in rel if k != "above_state_2"]
    both = bool(fc and all(rel[k]["all_eig_positive"] and rel[k]["completed"] for k in need)
                and all(abs(rel[k]["d_release"] - fc["d_2"]) < 0.05 for k in need if "state_2" in k) and abs(rel["near_state_1"]["d_release"]) < 0.05)
    metrics["rung2_bistable"] = dict(target=bi["target"], optimizer=fj, calculix=fc, calculix_mesh_used=vmesh,
                                     calculix_t3_as_designed=dict(completed=bi["ccx_t3"]["info"]["completed"], d_reached_mm=bi["ccx_t3"]["d_reached"]),
                                     calculix_t3_imperfection_plus=ex.get("ccx_t3_imp_plus", {}).get("features"),
                                     calculix_t3_imperfection_minus=ex.get("ccx_t3_imp_minus", {}).get("features"),
                                     calculix_t4=ex.get("ccx_t4", {}).get("features"), calculix_t6=bi["ccx_t6"]["features"], jax_80x2=bi["jax_80x2"]["features"],
                                     design_solver_min_tangent_eig_near_abort=dict(d=ex.get("design_solver_near_abort", {}).get("d_at_min_lam"), lam=ex.get("design_solver_near_abort", {}).get("min_lam")),
                                     jax_stability=bi.get("jax_stability"), calculix_release=rel, both_states_stable_in_calculix=both,
                                     release_from_plus0p6_failed=not rel.get("above_state_2", {}).get("completed", True),
                                     loop_forward_backward_max_diff=bi["jax_40x2_backward"]["max_abs_diff_vs_forward"],
                                     passed=bool(fc and fc["within_5pct"] and both), start=db.get("start"), p=bi["p"],
                                     attempts=db.get("attempts"))
json.dump(metrics, open("metrics.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ page
figs = ["cell_schematic.png", "targets.png", "rung0_qiu_validation.png", "gradcheck.png", "opt_history.png", "mesh_convergence.png"] + \
       [f"{k}_{T}.png" for T in ("T1", "T2", "T3") for k in ("curves", "shapes")] + (["bistable_curves.png", "bistable_shapes.png", "bistable_stall_diagnostic.png"] if bi else [])
for f in figs:
    shutil.copyfile(R / f, PAGE / f)

chip = lambda ok: '<span class="chip ok">PASS</span>' if ok else '<span class="chip bad">FAIL</span>'
pc = lambda x, n=2: f"{100*x:.{n}f} %"
m1 = metrics["rung1"]
names = {"T1": "T1 — stiffening curve", "T2": "T2 — plateau", "T3": "T3 — snap-through"}
rows = ""
for T in ("T1", "T2", "T3"):
    m = m1[T]
    sec = {"T1": f"monotonic: {'yes' if m['secondary'].get('monotonic') else 'no'}",
           "T2": f"plateau stays within {m['secondary'].get('plateau_max_dev_kPa', float('nan')):.3f} kPa of 0.40 (limit 0.06)",
           "T3": "peak then valley present" if m["secondary"].get("has_peak_then_valley") else "no peak/valley"}[T]
    rows += f"<tr><td><b>{names[T]}</b></td><td class='metric'>{pc(m['optimizer_nrmse'])}</td><td class='metric'><b>{pc(m['calculix_nrmse'])}</b></td><td class='metric'>≤ {pc(m['tolerance'],0)}</td><td>{sec}</td><td>{chip(m['passed'])}</td></tr>"
p0 = rung0["pass_preregistered"]
r0rows = "".join(
    f"<tr><td>{lab}</td><td class='metric'>{val}</td><td class='metric'>{tol}</td><td>{chip(ok)}</td></tr>" for lab, val, tol, ok in [
        ("snap-through force vs closed form (740 EIh/l³)", f"{100*c0['f_top']:+.1f} %", "± 5 %", p0["f_top"]),
        ("snap-back force vs closed form (370 EIh/l³)", f"{100*c0['f_bot']:+.1f} %", "± 5 %", p0["f_bot"]),
        ("position of force peak", f"{pp['d_top']:+.3f} h", "± 0.05 h", p0["d_top"]),
        ("position of force minimum", f"{pp['d_bot']:+.3f} h", "± 0.05 h", p0["d_bot"]),
        ("second stable position", f"{pp['d_end']:+.3f} h", "± 0.05 h", p0["d_end"]),
        ("whole curve, design solver vs CalculiX", pc(rung0["nrmse_jax_vs_calculix"]), "≤ 2 %", p0["jax_vs_calculix"])])
sr = rung0["rel_err_vs_series"]; q12 = rung0["supplementary_Q12_rel_err_vs_closed_form"]
gc = metrics["gradient_check"]

def tsec(T, blurb):
    m = m1[T]
    return f"""<h2>{names[T]}</h2><div class='card'><p>{blurb}</p><img class='fig' src='curves_{T}.png' alt='{T} curves'>
<p class='hint'>Dashed black = the prescribed target. Blue = the optimised design in the design solver. Orange dots = the same design re-meshed and re-solved in CalculiX — this is the curve that is graded. Grey = where the optimiser started.</p>
<table class='leg'><tr><td>optimiser's own error</td><td class='metric'>{pc(m['optimizer_nrmse'])} of the target range</td></tr>
<tr><td>CalculiX-verified error</td><td class='metric'><b>{pc(m['calculix_nrmse'])}</b> (tolerance {pc(m['tolerance'],0)}) &nbsp; {chip(m['passed'])}</td></tr>
<tr><td>gap between the two solvers</td><td class='metric'>{pc(m['solver_gap_jax_vs_calculix'])}</td></tr>
<tr><td>finer meshes change the curve by</td><td class='metric'>design solver {pc(m['mesh_conv_jax_40x2_vs_80x4'])}, CalculiX {pc(m['mesh_conv_calculix_t3_vs_t6'])}</td></tr>
<tr><td>if made slightly wrong (±0.05 mm, ±3 % thickness)</td><td class='metric'>typical error {pc(m['tolerance_sensitivity']['median_nrmse'],1)}, worst of 16 {pc(m['tolerance_sensitivity']['max_nrmse'],1)}; {100*m['tolerance_sensitivity']['frac_within_tol']:.0f} % still inside tolerance</td></tr></table>
<img class='fig' src='shapes_{T}.png' alt='{T} shapes' style='max-width:520px;margin-top:12px'></div>"""

bis = ""
if bi:
    b = metrics["rung2_bistable"]; fc = b["calculix"]; fj = b["optimizer"]; tg = b["target"]
    def brow(lab, key, unit, nd):
        cv = fc[key] if fc else None; jv = fj[key] if fj else None
        e = (cv / tg[key] - 1) if cv is not None else None
        return (f"<tr><td>{lab}</td><td class='metric'>{tg[key]:.{nd}f} {unit}</td><td class='metric'>{'—' if jv is None else f'{jv:.{nd}f}'}</td>"
                f"<td class='metric'><b>{'—' if cv is None else f'{cv:.{nd}f}'}</b></td><td class='metric'>{'—' if e is None else f'{100*e:+.1f} %'}</td><td>{chip(e is not None and abs(e) <= 0.05)}</td></tr>")
    rel = b["calculix_release"]
    relrows = "".join(f"<tr><td>held at {x['d_hold']:.2f} mm, then let go</td><td class='metric'>" + (f"settles at {abs(x['d_release']):.3f} mm; vibration eigenvalues all positive: {'yes' if x['all_eig_positive'] else 'NO'}" if x["completed"] else f"CalculiX static solve did not converge (stopped at {x['d_release']:.2f} mm) — reported, see note") + "</td></tr>" for x in sorted(rel.values(), key=lambda x: x["d_hold"]))
    bis = f"""<h2>Stretch rung — a cell that stays snapped</h2><div class='card'><p>Target: a cell with two resting positions. Pushing it must take a peak force of {tg['F_pk']*1e3:.0f} mN per mm depth, it must come to rest {tg['d_2']:.0f} mm lower, and pushing it back must cost {tg['E_back']*1e3:.0f} µJ per mm depth. Each number must be within 5 % in CalculiX and both resting positions must be shown stable.</p>
<img class='fig' src='bistable_curves.png' alt='bistable curves'>
<table class='leg' style='margin-top:10px'><tr><td><b>quantity</b></td><td><b>target</b></td><td><b>design solver</b></td><td><b>CalculiX ({b['calculix_mesh_used']})</b></td><td><b>error</b></td><td></td></tr>
{brow('snap-through peak force (N/mm)', 'F_pk', '', 4)}{brow('second stable position (mm)', 'd_2', '', 2)}{brow('return energy barrier (N·mm/mm)', 'E_back', '', 4)}</table>
<table class='leg' style='margin-top:10px'>{relrows}<tr><td>push-down vs come-back curves</td><td class='metric'>differ by at most {b['loop_forward_backward_max_diff']*1e3:.2f} mN/mm ({100*b['loop_forward_backward_max_diff']/tg['F_pk']:.1f} % of the peak force; a small loop near the force corner at 4 mm)</td></tr>
<tr><td>both resting positions stable in CalculiX</td><td>{chip(b['both_states_stable_in_calculix'])}</td></tr><tr><td><b>rung 2 overall</b></td><td>{chip(b['passed'])}</td></tr></table>
<img class='fig' src='bistable_shapes.png' alt='bistable shapes' style='max-width:520px;margin-top:12px'>
<p style='margin-top:14px'><b>It took six attempts, and the reason is the main finding of this rung.</b> Attempts 3–5 all stopped 8–16 % short. At the stalled design no limit was active and the gradient predicted a step that would hit all three targets. Taking that step fixed two of them and made the third worse:</p>
<img class='fig' src='bistable_stall_diagnostic.png' alt='stall diagnostic'>
<p class='hint'>Left: the stalled design has a small secondary snap on its negative-force branch (a jump in force at 7.4 mm). Right: the return barrier (orange) moves opposite to the gradient's prediction. The barrier is an area under the curve; when a jump in the curve moves or disappears, that area changes in a way no per-point gradient can see. The optimiser had climbed onto the ridge where the snap appears. Starting attempt 6 one step beyond the ridge converged in 26 evaluations.</p>
<p class='hint'>{{BIS_NOTE}}</p></div>"""

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial 9 — Inverse design of a snap-through metamaterial cell</title><style>
:root{--fg:#1c2128;--mut:#636c76;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--bg:#fff;--card:#f6f8fa}
*{box-sizing:border-box}body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}
.wrap{max-width:1040px;margin:0 auto;padding:28px 22px 80px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:25px;margin:.2em 0 .1em}h2{font-size:18px;margin:1.7em 0 .5em;border-bottom:1px solid var(--bd);padding-bottom:.25em}
.sub{color:var(--mut);margin:0 0 1.4em}
.crumb{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0;overflow-x:auto}
img.fig{max-width:100%;border:1px solid var(--bd);border-radius:8px;display:block}
.hint{color:var(--mut);font-size:12.5px;margin:.4em 0 0}
table.leg{border-collapse:collapse;width:100%;font-size:14px}
table.leg td{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top}
table.leg td:first-child{color:var(--mut);min-width:150px}
.chip{display:inline-block;font-size:12px;font-weight:600;padding:1px 9px;border-radius:20px;white-space:nowrap}
.chip.ok{background:#dafbe1;color:var(--ok)}.chip.bad{background:#ffebe9;color:var(--bad)}
code{background:#eaeef2;padding:1px 5px;border-radius:5px;font-size:12.5px}
.metric{font-variant-numeric:tabular-nums}
details summary{cursor:pointer;color:var(--accent);font-size:13.5px}
ul{padding-left:1.2em}li{margin:.35em 0}
</style></head><body><div class="wrap">
<div class="crumb"><a href="../">QA dashboard</a> · <a href="../../">project</a></div>
<h1>Trial 9 — Inverse design of a snap-through metamaterial cell</h1><p class="sub">Shape a rubber beam by gradient descent so that squeezing the cell follows a prescribed stress–strain curve, including one that buckles and snaps. Every design is graded by a second, independent solver.</p>
"""

BIS_NOTE = ""
if bi:
    BIS_NOTE = (Path("results/bistable_note.txt").read_text().strip() if Path("results/bistable_note.txt").exists() else "")
    bis = bis.replace("{BIS_NOTE}", BIS_NOTE)

b2 = metrics.get("rung2_bistable")
summary_rows = f"""<tr><td>Rung 0 — is the solver right?</td><td>Matches CalculiX to {pc(rung0['nrmse_jax_vs_calculix'])} and Qiu's theory at the tested geometry to {abs(100*sr['f_top']):.1f} % / {abs(100*sr['f_bot']):.1f} %. Misses two of six pre-registered checks against the rounded textbook constants (snap-back force {100*c0['f_bot']:+.1f} %).</td><td>{chip(rung0['pass_all'])}</td></tr>
<tr><td>T1 stiffening</td><td>CalculiX error {pc(m1['T1']['calculix_nrmse'])} (limit 5 %)</td><td>{chip(m1['T1']['passed'])}</td></tr>
<tr><td>T2 plateau</td><td>CalculiX error {pc(m1['T2']['calculix_nrmse'])} (limit 5 %)</td><td>{chip(m1['T2']['passed'])}</td></tr>
<tr><td>T3 snap-through</td><td>CalculiX error {pc(m1['T3']['calculix_nrmse'])} (limit 8 %)</td><td>{chip(m1['T3']['passed'])}</td></tr>"""
if b2:
    fc = b2["calculix"]
    txt = ("CalculiX: " + ", ".join(f"{lab} {100*(fc[k]/b2['target'][k]-1):+.1f} %" for lab, k in (("peak force", "F_pk"), ("second position", "d_2"), ("return barrier", "E_back"))) + " (limit ± 5 % each). Needed six optimisation attempts; the coarsest verifier mesh aborted at a buckling point and one of five release tests did not converge.") if fc else "not bistable in CalculiX"
    summary_rows += f"<tr><td>Stretch — programmed bistability</td><td>{txt}</td><td>{chip(b2['passed'])}</td></tr>"

HTML += f"""<h2>Result at a glance</h2><div class='card'><table class='leg'>{summary_rows}</table>
<p class='hint'>Tolerances were written down before any optimisation (<code>PREREGISTRATION.md</code>) and not changed. "Error" is the root-mean-square gap between the CalculiX curve and the target over 50 strain values, as a fraction of the target's stress range.</p></div>
<h2>Objective</h2><div class='card'><p>A mechanical metamaterial gets its behaviour from geometry. Here one repeating cell is a rubber beam held between two stiff walls and pushed at its middle. Depending on how the beam is curved and how its thickness varies, the cell can stiffen steadily, hold a constant force, or buckle and snap through.</p>
<p>The task is the <b>inverse</b> problem: given a stress–strain curve we want, find the beam. The search uses gradients computed <i>through</i> a large-deformation finite-element solve, which is delicate because gradients misbehave exactly where structures buckle. The design only counts if a different solver, with a different mesh and element type, reproduces the curve.</p>
<img class='fig' src='cell_schematic.png' alt='unit cell' style='max-width:640px'>
<table class='leg' style='margin-top:10px'><tr><td>Material</td><td>soft rubber (neo-Hookean, shear modulus 1 MPa, Poisson ratio 0.4), large-deformation, plane strain</td></tr>
<tr><td>Design freedom</td><td>9 numbers: 4 for the beam's centre-line, 5 for its thickness profile (0.4–3 mm), mirror-symmetric</td></tr>
<tr><td>Design solver</td><td>hand-written differentiable finite elements in JAX (9-node quads, float64); gradients by implicit differentiation (adjoint)</td></tr>
<tr><td>Optimiser</td><td>least-squares trust region on 50 stress residuals, 4 generic starting shapes per target</td></tr>
<tr><td>Independent verifier</td><td>CalculiX 2.23, 6-node triangles on a finer body-fitted Gmsh mesh, whole beam, no symmetry assumed</td></tr></table></div>
<h2>Rung 0 — can the solver be trusted?</h2><div class='card'><p>Before designing anything, the solver was tested on a beam with a known answer: the cosine-shaped bistable beam of Qiu, Lang &amp; Slocum (2004), which has closed-form snap-through and snap-back forces.</p>
<img class='fig' src='rung0_qiu_validation.png' alt='rung 0'>
<p class='hint'>Blue (design solver), solid orange (CalculiX) and dashed black (Qiu's theory) lie on top of each other. The dotted orange curve is CalculiX on the perfectly symmetric beam: it rides an unstable equilibrium path and then fails. A 1 %-of-thickness bump fixes it.</p>
<table class='leg' style='margin-top:10px'><tr><td><b>pre-registered check</b></td><td><b>measured</b></td><td><b>tolerance</b></td><td></td></tr>{r0rows}</table>
<p style='margin-top:10px'><b>Two misses, reported as misses.</b> The constants 740 and 370 are the theory's limit for very slender arches. For the tested beam (height/thickness = 6) the same theory, evaluated in full, gives 714 and 340; against those the solver is off by {100*sr['f_top']:+.1f} % and {100*sr['f_bot']:+.1f} %. A follow-up run on a twice-as-slender beam (height/thickness = 12) matches the closed form to {100*q12['f_top']:+.1f} % and {100*q12['f_bot']:+.1f} %. The solver agrees with CalculiX to {pc(rung0['nrmse_jax_vs_calculix'])}, and the two codes' material laws agree to 7 digits on a one-element test.</p></div>
<h2>The three targets</h2><div class='card'><img class='fig' src='targets.png' alt='targets' style='max-width:640px'><p class='hint'>Defined from generic formulas, not from any design in the search space, so whether the cell could reach them was not known in advance.</p></div>
{tsec('T1', 'A curve that gets steadily stiffer, like soft tissue (an exponential). The optimiser turned a plain arch into a wavy beam that unfolds and then goes taut.')}
{tsec('T2', 'Rise, then hold a constant 0.40 kPa from 10 % to 35 % strain, then stiffen. The optimiser kept an arch and reshaped its thickness (up to the 3 mm cap) until the push-through force stayed level.')}
{tsec('T3', 'Rise to a peak, lose three quarters of the load (negative stiffness), then recover. This is snap-through behaviour; under displacement control both solvers can follow it.')}
<h2>Pass / fail</h2><div class='card'><table class='leg'><tr><td><b>target</b></td><td><b>optimiser's loss</b></td><td><b>CalculiX error</b></td><td><b>tolerance</b></td><td><b>second criterion</b></td><td></td></tr>{rows}</table>
<img class='fig' src='opt_history.png' alt='history' style='margin-top:12px'><p class='hint'>Best error so far against solver evaluations, per starting shape. The starting shape matters: for the plateau and snap-through targets only arch-shaped starts get there; flat or sagging starts stall far above tolerance.</p>
<img class='fig' src='mesh_convergence.png' alt='mesh convergence' style='margin-top:12px'><p class='hint'>Mesh convergence: every mesh in both solvers stays within about 0.3 % of the finest CalculiX mesh.</p></div>
<h2>Are the gradients right — and where do they stop meaning anything?</h2><div class='card'><img class='fig' src='gradcheck.png' alt='gradient check'>
<ul><li><b>Right almost everywhere.</b> Adjoint gradients match finite differences to better than {max(v['max_rel_err_excluding_dc'] for v in gc['adjoint_vs_same_branch_fd'].values()):.0e} (relative), even 0.02 mm past a buckling point.</li>
<li><b>They jump at a buckling point.</b> On a perfectly symmetric arch the force curve has a corner where the beam buckles (d = {gc['d_c_mm']:.2f} mm). The sensitivity to thickness jumps from {gc['jump_across_dc']['log_t0'][0]:.3f} to {gc['jump_across_dc']['log_t0'][1]:.3f} across it, so the loss has a kink whenever that corner crosses a sampled strain.</li>
<li><b>They are meaningless for symmetry-breaking changes.</b> The sensitivity to a lopsided shape change is zero before buckling, then {gc['dF_da2']['at_dc']:.2f} at the buckling point, decaying like one over the square root of the distance ({gc['dF_da2']['dc+0.02']:.3f} at +0.02 mm, {gc['dF_da2']['dc+4']:.4f} at +4 mm). Its sign is whichever way the beam happened to buckle; re-running the simulation gives a finite-difference slope of exactly zero. The gradient describes one branch, not the function the optimiser evaluates.</li>
<li><b>Consistent with what happened in practice.</b> The first bistability run started on perfectly symmetric arches and made no progress at all.</li></ul></div>
{bis}
<h2>What failed along the way</h2><div class='card'><ul>
<li><b>CalculiX followed an unstable path</b> on the symmetric validation beam (force peak 27 % too high, then a crash). Its Newton solver has no stability check. Seeding a tiny imperfection is the documented fix; the optimised designs are not symmetric-degenerate and needed none.</li>
<li><b>The first plateau and snap-through optimisations hung.</b> Weak bound penalties let the optimiser fold the beam onto itself; the solver ground on a nonsense mesh. Fixed with stronger penalties and two added shape-validity limits (slope, curvature × thickness).</li>
<li><b>Solver round-off flaw.</b> Near convergence the energy line search failed and declared a good step bad, aborting one mesh-convergence run. Fixed by accepting steps that reduce the residual.</li>
<li><b>Bistability, attempts 1–2.</b> Stalled first on symmetric starts (consistent with the gradient finding above), then on a solver stall at a second instability where parameter changes of 1e-7 decided whether the run "failed". Fixed with a saddle-free Newton step.</li>
{{BIS_FAIL}}</ul></div>
<h2>Honest caveats</h2><div class='card'><ul>
<li><b>Simulation only.</b> Nothing was printed or tested. The independent check is a second solver, not an experiment.</li>
<li><b>The two solvers share modelling assumptions</b>: 2-D plane strain, the same rubber law, perfectly rigid walls and shuttle, no self-contact, no viscoelasticity. CalculiX verifies the numerics and the geometry hand-off, not those assumptions. A compliant frame would change the response; it was not modelled.</li>
<li><b>The targets turned out easy for this cell.</b> Verified errors are 4–20 times inside tolerance. The tolerances were fixed in advance, so they were not tuned to pass, but they were not a hard test either.</li>
<li><b>Designs are sensitive to manufacturing error.</b> With ±0.05 mm shape error and ±3 % thickness, only {100*m1['T2']['tolerance_sensitivity']['frac_within_tol']:.0f}–{100*m1['T1']['tolerance_sensitivity']['frac_within_tol']:.0f} % of perturbed copies stay inside tolerance.</li>
<li><b>Many shapes give the same curve.</b> In a labelled sanity check the optimiser reproduced a known design's curve to better than 0.001 % with parameters up to 1.9 mm away from the true ones.</li>
<li><b>Rung 0 has two reported misses</b> against the rounded closed-form constants (see above).</li>
<li><b>The sideways-sway check is light.</b> A 2 % sideways load on a laterally free shuttle moved it by at most {max(m1[T]['sway_2pct_side_load']['max_drift_mm'] for T in m1):.2f} mm and left the curves unchanged, but shuttle rotation was never released.</li></ul></div>
<details><summary>Terms used on this page</summary><div class='card'><table class='leg'>
<tr><td><b>metamaterial</b></td><td>A material whose behaviour comes from the shape of its repeating cells rather than from its chemistry.</td></tr>
<tr><td><b>unit cell</b></td><td>The single repeating building block; here one curved rubber beam in a stiff frame.</td></tr>
<tr><td><b>stress–strain curve</b></td><td>How hard the cell pushes back (force per area) against how far it has been squeezed (as a fraction of its height).</td></tr>
<tr><td><b>inverse design</b></td><td>Starting from the behaviour you want and solving for the shape that produces it.</td></tr>
<tr><td><b>buckling</b></td><td>A sudden change of deformation pattern when a compressed part gives way sideways.</td></tr>
<tr><td><b>snap-through</b></td><td>A curved part popping to the other side; the force drops as it travels, which is negative stiffness.</td></tr>
<tr><td><b>bistable</b></td><td>Having two different rest positions, each stable with no force applied.</td></tr>
<tr><td><b>energy barrier</b></td><td>The work needed to push the cell out of one rest position over the tipping point.</td></tr>
<tr><td><b>bifurcation</b></td><td>A point where the equilibrium path splits into branches; the structure must "choose" one.</td></tr>
<tr><td><b>geometrically nonlinear</b></td><td>Deformations large enough that the changing shape itself alters the stiffness.</td></tr>
<tr><td><b>neo-Hookean</b></td><td>The simplest large-strain rubber model.</td></tr>
<tr><td><b>plane strain</b></td><td>A 2-D idealisation of a long extruded part: no strain out of the page.</td></tr>
<tr><td><b>adjoint / implicit differentiation</b></td><td>A way to get the gradient of a simulation result with respect to all design parameters for about the cost of one extra linear solve.</td></tr>
<tr><td><b>finite difference</b></td><td>The brute-force gradient check: nudge a parameter, re-solve, divide.</td></tr>
<tr><td><b>NRMSE</b></td><td>Root-mean-square gap between two curves divided by the target's stress range; 0 is a perfect match.</td></tr>
<tr><td><b>mesh convergence</b></td><td>Checking that a finer mesh does not change the answer.</td></tr>
<tr><td><b>CalculiX</b></td><td>An open-source finite-element solver, used here only as the independent referee.</td></tr>
<tr><td><b>pre-registration</b></td><td>Writing the pass criteria down before running, so they cannot drift toward the result.</td></tr></table></div></details>
<h2>Artifacts</h2><div class='card'><p class='hint'><a href='https://github.com/eschmitt88/agentic-solid-mechanics/tree/master/experiments/2026-11-14-metamaterial-inverse-design'>experiment folder on GitHub</a> — README, PREREGISTRATION.md, log.md (chronological, including dead ends), config.yaml, metrics.json, code and results/. Reference: J. Qiu, PhD thesis, MIT 2003, ch. 2 (<a href='https://pergatory.mit.edu/research/jqiu/thesisscreen.pdf'>pdf</a>); Qiu, Lang &amp; Slocum, JMEMS 13(2), 2004, doi:10.1109/JMEMS.2004.825308.</p></div>
</div></body></html>"""
BIS_FAIL = Path("results/bistable_fail_items.html").read_text() if Path("results/bistable_fail_items.html").exists() else ""
HTML = HTML.replace("{BIS_FAIL}", BIS_FAIL)
(PAGE / "index.html").write_text(HTML)
print("page written", len(HTML), "bytes; rung0 pass_all:", rung0["pass_all"], {T: m1[T]["passed"] for T in m1}, "rung2:", b2 and b2["passed"])
