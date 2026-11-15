"""Assemble metrics.json and the review page docs/qa/trial-12-inverse-design-hard/index.html from results/*.json."""
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

import design

R = Path("results")
PAGE = Path("../../docs/qa/trial-12-inverse-design-hard")
PAGE.mkdir(parents=True, exist_ok=True)
J = lambda f: json.load(open(R / f))
ex = lambda f: (R / f).exists()
pc = lambda x, n=1: "—" if x is None else f"{100*x:.{n}f} %"

M = dict(trial=12, date="2026-11-15", targets={})

# ---- curve targets ---------------------------------------------------------------------------------------------
for t, vname in (("H1", "H1"), ("H2", "H2"), ("H4", "H4")):
    starts = {json.load(open(f))["start"]: json.load(open(f)) for f in sorted(glob.glob(str(R / f"opt/{t}__*.json")))}
    rec = dict(tolerance_nrmse=design.SPEC[t]["tol"], optimizer_nrmse_per_start={s: r["nrmse"] for s, r in starts.items()},
               nfev_per_start={s: r["nfev"] for s, r in starts.items()}, proj_grad_norm_per_start={s: r["proj_grad_norm"] for s, r in starts.items()})
    if starts:
        b = min(starts.values(), key=lambda r: r["nrmse"])
        rec.update(best_start=b["start"], optimizer_nrmse=b["nrmse"], optimizer_status=b["message"], design_solver_bands=b["design_grade"]["bands"])
    if ex(f"verify_{vname}.json"):
        v = J(f"verify_{vname}.json")
        rec.update(verdict=v["verdict"]["verdict"], calculix_nrmse=v["verdict"].get("nrmse"), calculix_bands=v["verdict"].get("bands"), graded_mesh=v.get("graded_mesh") or v["verdict"].get("mesh"),
                   ladder={k: dict(completed=r["completed"], d_reached=r["d_reached"], nrmse=r["grade"]["nrmse"]) for k, r in v["ccx"].items()},
                   solver_gap=v.get("solver_gap_jax40x2_vs_ccx"), ladder_spread=v.get("ladder_spread"), design_mesh_conv_40x2_vs_80x2=v.get("mesh_conv_jax_40x2_vs_80x2"), p=v["p"])
        if "imperfection_runs" in v:
            rec["imperfection_runs"] = {k: dict(completed=r["completed"], d_reached=r["d_reached"], nrmse=r["grade"]["nrmse"]) for k, r in v["imperfection_runs"].items()}
    else:
        rec["verdict"] = "not verified"
    M["targets"][t] = rec
if ex("certificate.json"):
    c = J("certificate.json")
    M["infeasibility_certificate"] = {t: dict(lower_bound_nrmse=c[t]["lower_bound_nrmse"], tolerance=c[t]["tolerance"], certified_infeasible=c[t]["certified_infeasible"]) for t in c}

# ---- robust design -----------------------------------------------------------------------------------------------
rob = None
if ex("robust_T2.json"):
    r = J("robust_T2.json")
    rob = dict(target="T2 (trial 9)", n_design_solver=64, optimisation_start="nominal optimum (pre-registered)")
    for key in ("nominal", "robust"):
        cp = r[key + "_design"]["copies"]
        cc = [json.load(open(f)) for f in sorted(glob.glob(str(R / f"robust_ccx/T2_{key}_*.json")))]
        rob[key] = dict(own_nrmse_design_solver=r[key + "_design"]["grade"]["nrmse"], pass_fraction_design_solver=cp["pass_fraction"], median_nrmse=cp["median_nrmse"], p90_nrmse=cp["p90_nrmse"],
                        n_calculix=len(cc), pass_fraction_calculix=float(np.mean([c["passed"] for c in cc])) if cc else None,
                        calculix_aborted=int(sum(c["nrmse"] is None for c in cc)), calculix_fallback_mesh=int(sum(c.get("lc_div") == 4.0 for c in cc)),
                        agreement_calculix_vs_design_solver=float(np.mean([c["passed"] == cp["samples"][c["k"]]["passed"] for c in cc])) if cc else None)
    rob["rms_over_optimisation_copies"] = dict(start=r["opt"]["history"][0]["mean_sq"] ** 0.5, end=r["opt"]["rms_over_copies"], nfev=r["opt"]["nfev"], proj_grad_norm=r["opt"]["proj_grad_norm"])
    d_ds = rob["robust"]["pass_fraction_design_solver"] - rob["nominal"]["pass_fraction_design_solver"]
    d_cc = None if rob["robust"]["pass_fraction_calculix"] is None else rob["robust"]["pass_fraction_calculix"] - rob["nominal"]["pass_fraction_calculix"]
    rob["gain_design_solver"], rob["gain_calculix"] = d_ds, d_cc
    rob["claim_robust_design_helps"] = bool(d_ds >= 0.15 and d_cc is not None and d_cc >= 0.15)
    rob["other_starts_not_preregistered"] = {}
    for f in sorted(glob.glob(str(R / "robust_T2_from_*.json"))):
        q = json.load(open(f))
        rob["other_starts_not_preregistered"][q["start"]] = dict(own_nrmse=q["robust_design"]["grade"]["nrmse"], pass_fraction_design_solver=q["robust_design"]["copies"]["pass_fraction"],
                                                                 rms_over_optimisation_copies=q["opt"]["rms_over_copies"])
    if ex("robust_decompose_T2.json"):
        rob["decomposition_nominal"] = J("robust_decompose_T2.json")
    if ex("verify_T2_robust.json"):
        v = J("verify_T2_robust.json")
        rob["robust_design_calculix"] = dict(verdict=v["verdict"]["verdict"], nrmse=v["verdict"].get("nrmse"), graded_mesh=v.get("graded_mesh"), bands=v["verdict"].get("bands"))
    M["robust_design"] = rob

# ---- snap test ---------------------------------------------------------------------------------------------------
snp = J("snap_summary.json") if ex("snap_summary.json") else None
if snp:
    M["snap_test"] = snp["arms"]
    M["snap_test_per_run"] = snp["per_run"]
if ex("gradcheck_fold.json"):
    g = J("gradcheck_fold.json")
    dds, loss, feat_ok, feat_jump = [], [], [], 0
    for name, rec in g.items():
        for k, v in rec.get("fd_dds", {}).items():
            if k.endswith("1e-05") and np.isfinite(v["fd"]):
                dds.append(v["rel"]); loss.append(v["rel_loss"])
        for k, v in rec["fd_feat"].items():
            if "rel" in v:
                if max(v["rel"]) < 1e-3:
                    feat_ok.append(max(v["rel"]))
                else:
                    feat_jump += 1
    M["gradient_check_fold"] = dict(threshold=1e-3, snap_position_sensitivity_max_rel_err=max(dds), n_snap_position_checks=len(dds), energy_drop_sensitivity_max_rel_err=max(loss),
                                    whole_pipeline_max_rel_err_where_continuous=max(feat_ok), n_whole_pipeline_probes_matching=len(feat_ok), n_probes_straddling_branch_flip=feat_jump,
                                    passed=bool(max(dds) <= 1e-3 and max(feat_ok) <= 1e-3))
if ex("branch_flip.json"):
    b = J("branch_flip.json")
    M["branch_flip"] = dict(jump_in_E_back=float(max(abs(np.diff(b["E_new"])))), relative=float(max(abs(np.diff(b["E_new"]))) / np.mean(b["E_new"])))
bi = None
if ex("verify_bistable.json"):
    bi = J("verify_bistable.json")
    M["targets"]["B"] = dict(target=dict(F_pk=0.040, d_2=10.0, E_back=0.045), tolerance_rel=0.05, design=J("designs/bistable.json") if ex("designs/bistable.json") else None,
                             design_solver=bi["jax_40x2"]["features"], design_solver_80x2=bi["jax_80x2"]["features"], verdict=bi["verdict"]["verdict"], graded_mesh=bi.get("graded_mesh") or bi["verdict"].get("mesh"),
                             calculix=bi["verdict"].get("features"), ladder={k: dict(completed=r["completed"], d_reached=r["d_reached"], features=r["features"]) for k, r in bi["ccx"].items()},
                             imperfection_runs={k: dict(completed=r["completed"], features=r["features"]) for k, r in bi.get("imperfection_runs", {}).items()},
                             forward_backward_max_diff=bi["jax_backward_max_abs_diff"])

# ---- relaxed idealisation ------------------------------------------------------------------------------------------
rel = {}
for f in sorted(glob.glob(str(R / "relaxed_*.json"))):
    q = json.load(open(f))
    e = dict(kind=q["kind"])
    for key in ("ps", "3d", "ef"):
        if key in q:
            x = q[key]
            e[key] = {k: x.get(k) for k in ("completed", "d_reached", "shift_nrmse_vs_plane_strain", "median_force_ratio", "feature_change_vs_plane_strain", "nnodes", "error") if x.get(k) is not None}
            if "grade" in x:
                e[key].update(nrmse=x["grade"]["nrmse"], passed=x["grade"]["passed"])
            if x.get("features"):
                e[key].update(features=x["features"])
    rel[q["name"]] = e
M["relaxed_idealisation"] = rel
json.dump(M, open("metrics.json", "w"), indent=1, default=float)

# ====================================================================================================================
figs = ["cell_schematic.png", "targets.png", "design_H1.png", "design_H2.png", "design_H4.png", "opt_starts.png", "certificate_H4.png", "robust_T2.png", "robust_T2_shapes.png",
        "snap_matrix.png", "snap_traces.png", "gradcheck_fold.png", "branch_flip.png", "bistable.png", "bistable_margin.png", "relaxed.png", "design_T2_robust.png"]
have = set()
for f in figs:
    if ex(f):
        shutil.copyfile(R / f, PAGE / f)
        have.add(f)
img = lambda f, alt, style="": f"<img class='fig' src='{f}' alt='{alt}' style='{style}'>" if f in have else f"<p class='hint'>(figure {f} not produced)</p>"
CH = {"PASS": "ok", "MISS": "bad", "FAIL (unverifiable)": "bad"}
chip = lambda v: f"<span class='chip {CH.get(v, 'na')}'>{v}</span>"
T = M["targets"]


def trow(cells, labels):
    return "<tr>" + "".join(f"<td data-l='{l}'>{c}</td>" for c, l in zip(cells, labels)) + "</tr>"


LAB = ["target", "optimiser's own error", "CalculiX-verified", "tolerance", "second criterion", "verdict"]
rows = ""
names = {"H1": "H1 — published foam plateau", "H2": "H2 — two-step staircase (two cells)", "H4": "H4 — expected infeasible"}
for t in ("H1", "H2", "H4"):
    m = T[t]
    bands = m.get("calculix_bands") or []
    sec = "; ".join(f"band {b['lo']:.2f}–{b['hi']:.2f}: off by {b['max_dev']:.3f} kPa (limit {b['tol']:.3f})" for b in bands) or ("—" if t != "H4" else "energy floor " + pc(M.get("infeasibility_certificate", {}).get("H4", {}).get("lower_bound_nrmse")))
    cal = pc(m.get("calculix_nrmse")) if m.get("calculix_nrmse") is not None else ("no completed run" if m["verdict"].startswith("FAIL") else "—")
    rows += trow([f"<b>{names[t]}</b>", pc(m.get("optimizer_nrmse")), f"<b>{cal}</b>" + (f" <span class='hint'>(mesh {m['graded_mesh'].replace('t', 't_min/', 1)})</span>" if m.get("graded_mesh") else ""), "≤ " + pc(m["tolerance_nrmse"], 0), sec, chip(m["verdict"])], LAB)
if "B" in T:
    b = T["B"]
    fc = b["calculix"] if isinstance(b["calculix"], dict) else None
    fj = b["design_solver"]
    e_j = max(abs(np.array(fj["rel_err"]))) if fj else None
    e_c = max(abs(np.array(fc["rel_err"]))) if fc else None
    rows += trow(["<b>B — bistable cell (trial 9's target)</b>", "largest feature error " + pc(e_j, 2), "<b>" + ("largest feature error " + pc(e_c, 2) if e_c is not None else "no completed run") + "</b>" + f" <span class='hint'>(mesh {b['graded_mesh'].replace('t', 't_min/', 1)})</span>",
                  "each feature ≤ 5 %", "peak force, second position, return barrier", chip(b["verdict"])], LAB)
if rob:
    rd = rob.get("robust_design_calculix", {})
    rows += trow(["<b>R — robust version of trial 9's plateau</b>", pc(rob["robust"]["own_nrmse_design_solver"]), "<b>" + pc(rd.get("nrmse")) + "</b>", "≤ 5 %",
                  f"share of imperfect copies inside tolerance: {pc(rob['robust']['pass_fraction_design_solver'],0)} vs {pc(rob['nominal']['pass_fraction_design_solver'],0)} for the nominal design (needed: +15 points)",
                  chip(rd.get("verdict", "not verified")) + " nominal; " + ("<span class='chip ok'>robustness gain</span>" if rob["claim_robust_design_helps"] else "<span class='chip bad'>no robustness gain</span>")], LAB)

# snap table
ARMS = [("B0-legacy", "Trial 9's method exactly as run yesterday"), ("B0", "Trial 9's gradient, repaired path-following"), ("B0+O", "… plus automatic restarts"),
        ("A", "New fold-aware gradient"), ("A+O", "Fold-aware gradient plus automatic restarts (pre-registered remedy)"), ("A+O2", "Same, giving up on a stall sooner (added during the run)"),
        ("AM+O2", "Same, plus a margin that keeps the design off the buckling surface (added during the run)"), ("AM+O2@120", "Margin variant with 120 evaluations, only on starts it had missed (added during the run)")]
srows = ""
if snp:
    for a, lab in ARMS:
        g = snp["arms"].get(a)
        if not g:
            srows += trow([f"<b>{lab}</b>", "not run", "—", "—"], ["method", "the 11 pre-registered starts", "6 fresh starts", "typical evaluations"])
            continue
        p, f = g.get("preregistered"), g.get("fresh")
        srows += trow([f"<b>{lab}</b>", f"<b>{p['success']} of {p['n']}</b> reach all three targets within 1 % ({p['within_5pct']} within 5 %)" if p else "not run",
                       f"<b>{f['success']} of {f['n']}</b> ({f['within_5pct']} within 5 %)" if f else "not run", f"{p['median_nfev']:.0f}" if p else "—"],
                      ["method", "the 11 pre-registered starts", "6 fresh starts", "typical evaluations"])

rrows = ""
RL = ["design", "3D body, 10 mm deep", "elastic frame", "still inside its tolerance?"]
for name, e in rel.items():
    def cell(k):
        x = e.get(k)
        if not x:
            return "not run"
        if not x.get("completed"):
            return f"CalculiX stopped at {x.get('d_reached', 0):.1f} mm" + (f"; up to there: curve moves {pc(x.get('shift_nrmse_vs_plane_strain'))}" if x.get("shift_nrmse_vs_plane_strain") is not None else "")
        if e["kind"] == "curve":
            return f"curve moves by {pc(x.get('shift_nrmse_vs_plane_strain'))} of the target range; forces × {x.get('median_force_ratio', float('nan')):.2f}"
        f_ = x.get("features")
        return "not bistable any more" if not f_ else f"against the target: peak force {100*f_['rel_err'][0]:+.0f} %, second position {100*f_['rel_err'][1]:+.1f} %, return barrier {100*f_['rel_err'][2]:+.0f} %"
    def ok(k):
        x = e.get(k)
        if not x:
            return "not run"
        if not x.get("completed"):
            return "run aborted"
        if e["kind"] == "curve":
            return "yes" if x.get("passed") else "no"
        f_ = x.get("features")
        return "yes" if f_ and f_["within_5pct"] else "no"
    nice = {"H1": "H1 foam plateau (a miss to begin with)", "T2_nominal": "Trial 9's plateau design", "T2_robust": "Robust plateau design", "bistable": "Bistable design (on the buckling surface)", "bistable_margin": "Bistable design (margin variant)"}
    rrows += trow([f"<b>{nice.get(name, name)}</b>", cell("3d"), cell("ef"), f"plane strain (same mesh): {ok('ps')} · 3D: {ok('3d')} · elastic frame: {ok('ef')}"], RL)

gc = M.get("gradient_check_fold", {})
cert = M.get("infeasibility_certificate", {}).get("H4", {})
TEXT = json.load(open("page_text.json")) if os.path.exists("page_text.json") else {}
tx = lambda k: TEXT.get(k, "")

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial 12 — Inverse design, the hard version</title><style>
:root{{--fg:#1c2128;--mut:#636c76;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--bg:#fff;--card:#f6f8fa}}
*{{box-sizing:border-box}}body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}}
.wrap{{max-width:1040px;margin:0 auto;padding:28px 22px 80px}}
a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}
h1{{font-size:25px;margin:.2em 0 .1em}}h2{{font-size:18px;margin:1.7em 0 .5em;border-bottom:1px solid var(--bd);padding-bottom:.25em}}h3{{font-size:15.5px;margin:1.2em 0 .3em}}
.sub{{color:var(--mut);margin:0 0 1.4em}}.crumb{{color:var(--mut);font-size:13px;margin-bottom:18px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0}}
img.fig{{max-width:100%;height:auto;border:1px solid var(--bd);border-radius:8px;display:block;margin:10px 0}}
.hint{{color:var(--mut);font-size:12.5px;margin:.4em 0 0}}
table.t{{border-collapse:collapse;width:100%;font-size:14px}}
table.t td,table.t th{{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top;text-align:left}}
table.t th{{color:var(--mut);font-weight:600;font-size:12.5px}}
.chip{{display:inline-block;font-size:12px;font-weight:600;padding:1px 9px;border-radius:20px;white-space:nowrap}}
.chip.ok{{background:#dafbe1;color:var(--ok)}}.chip.bad{{background:#ffebe9;color:var(--bad)}}.chip.na{{background:#eaeef2;color:var(--mut)}}
code{{background:#eaeef2;padding:1px 5px;border-radius:5px;font-size:12.5px;word-break:break-word}}
ul{{padding-left:1.2em}}li{{margin:.35em 0}}
dl{{margin:0}}dt{{font-weight:600;margin-top:.6em}}dd{{margin:0 0 0 0;color:var(--fg)}}
@media (max-width:640px){{.wrap{{padding:18px 12px 60px}}h1{{font-size:21px}}
 table.t thead{{display:none}}table.t tr{{display:block;border-bottom:2px solid var(--bd);padding:6px 0}}
 table.t td{{display:block;border:0;padding:3px 2px}}table.t td::before{{content:attr(data-l) ": ";color:var(--mut);font-size:12px}}table.t td:first-child::before{{content:""}}}}
</style></head><body><div class="wrap">
<div class="crumb"><a href="../">QA dashboard</a> · <a href="../trial-9-metamaterial/">trial 9 (yesterday's run this one follows)</a></div>
<h1>Trial 12 — Inverse design, the hard version</h1>
<p class="sub">Yesterday an AI agent shaped a rubber beam so that squeezing it followed a prescribed force curve, and passed easily. Today the same agent attacks the four weak points of that pass: harder targets, designs that survive manufacturing error, a gradient that goes blind when the structure snaps, and assumptions both solvers shared. Every number below was fixed as a pass rule before any optimisation ran and is graded by an independent solver (CalculiX), not by the optimiser.</p>

<div class="card">{tx('headline')}</div>

<h2>The cell and the targets</h2>
<div class="card"><p>One cell is a curved rubber beam between two stiff walls; a stiff block in the middle is pushed down and the force is recorded. The optimiser changes the beam's centre line and its thickness along the span (13 numbers per beam today, 9 yesterday). For the staircase target two cells are stacked, so that one gives way first and the other later.</p>
{img('cell_schematic.png', 'cell schematic')}{img('targets.png', 'targets')}
<p class="hint">Black = the curve asked for. Grey bars = bands the curve must stay inside. H1's numbers come from the Gibson–Ashby foam formulas (source below). H4 asks for something no elastic object can do, to check that the method reports a miss instead of a pass.</p></div>

<h2>Pass / fail, number against tolerance</h2>
<div class="card"><table class="t"><thead><tr>{''.join(f'<th>{l}</th>' for l in LAB)}</tr></thead><tbody>{rows}</tbody></table>
<p class="hint">"Error" is the root-mean-square distance between curve and target as a share of the target's range. The verdict uses the CalculiX column only: a fresh, finer, body-fitted mesh, graded on the finest of three mesh sizes that runs to the end (the rule written down beforehand).</p></div>

<h2>H1 — a foam-like plateau taken from the literature</h2>
<div class="card">{tx('H1')}{img('design_H1.png', 'H1 design and curves')}{img('opt_starts.png', 'optimiser loss per start')}</div>

<h2>H2 — a two-step staircase from two stacked cells</h2>
<div class="card">{tx('H2')}{img('design_H2.png', 'H2 design and curves')}</div>

<h2>H4 — the target that cannot be met</h2>
<div class="card">{tx('H4')}{img('certificate_H4.png', 'H4 certificate')}{img('design_H4.png', 'H4 best design')}</div>

<h2>Designing for manufacturing error</h2>
<div class="card">{tx('robust')}{img('robust_T2.png', 'robustness comparison')}{img('robust_T2_shapes.png', 'nominal and robust shapes', 'max-width:560px')}</div>

<h2>Past the snap</h2>
<div class="card">{tx('snap_intro')}{img('branch_flip.png', 'branch flip')}{tx('snap_grad')}{img('gradcheck_fold.png', 'gradient check')}
<h3>Success rates from the same starts that stalled yesterday</h3>
<table class="t"><thead><tr><th>method</th><th>the 11 pre-registered starts</th><th>6 fresh starts</th><th>typical evaluations</th></tr></thead><tbody>{srows}</tbody></table>
{tx('snap_result')}{img('snap_matrix.png', 'snap test matrix')}{img('snap_traces.png', 'snap test traces')}
<h3>The bistable design, re-solved in CalculiX</h3>{tx('bistable')}{img('bistable.png', 'bistable verification')}{img('bistable_margin.png', 'bistable verification, margin design') if 'bistable_margin.png' in have else ''}</div>

<h2>Dropping an assumption both solvers shared</h2>
<div class="card">{tx('relaxed')}<table class="t"><thead><tr>{''.join(f'<th>{l}</th>' for l in RL)}</tr></thead><tbody>{rrows}</tbody></table>{img('relaxed.png', 'relaxed idealisation curves')}</div>

<h2>Honest caveats</h2><div class="card">{tx('caveats')}</div>

<h2>What changed after the plan was written down</h2><div class="card">{tx('deviations')}</div>

<h2>Glossary</h2><div class="card"><dl>
<dt>Design solver</dt><dd>The agent's own finite-element code (JAX), which also returns exact gradients of the force with respect to the shape. Used to optimise.</dd>
<dt>CalculiX</dt><dd>An independent open-source finite-element program. Every design is re-meshed and re-solved in it; only its numbers decide pass or miss.</dd>
<dt>NRMSE</dt><dd>Root-mean-square gap between a curve and its target, divided by the target's range. 0.03 means "3 % of the range, on average".</dd>
<dt>Snap / fold point</dt><dd>A position where the structure can no longer follow the push smoothly and jumps to another shape. Mathematically the stiffness matrix becomes singular there.</dd>
<dt>Branch</dt><dd>One continuous family of shapes the structure can take as it is pushed. Near a buckling point there are two, and a tiny design change can decide which one is followed.</dd>
<dt>Bistable</dt><dd>Has two resting positions; once pushed past a tipping point it stays snapped.</dd>
<dt>Return barrier</dt><dd>The work needed to push a snapped cell back over its tipping point.</dd>
<dt>Plane strain</dt><dd>The assumption that the part is infinitely deep, so nothing moves out of the drawing plane. A real 10 mm deep part is softer.</dd>
<dt>Pass fraction</dt><dd>Of many copies made with random manufacturing error, the share that still meets the tolerance.</dd>
<dt>Pre-registration</dt><dd>Targets, tolerances and test starts written to a file before any optimisation, and not edited afterwards.</dd></dl></div>

<h2>Sources and files</h2><div class="card"><ul>
<li>H1 target: L. J. Gibson, MIT OpenCourseWare 3.054 <i>Cellular Solids</i>, Lecture 7 "Foams" notes (E* = E<sub>s</sub>(ρ*/ρ<sub>s</sub>)², σ*<sub>el</sub> = 0.05 E<sub>s</sub>(ρ*/ρ<sub>s</sub>)², ε<sub>D</sub> = 1 − 1.4 ρ*/ρ<sub>s</sub>), <a href="https://ocw.mit.edu/courses/3-054-cellular-solids-structure-properties-and-applications-spring-2015/a7c6b7bd4d16424aeeff92f1bb277c59_MIT3_054S15_L7_foams_trans.pdf">PDF opened 2026-11-15</a>; after Gibson &amp; Ashby, <i>Cellular Solids</i>, 2nd ed., Cambridge University Press 1997.</li>
<li>Experiment folder: <code>experiments/2026-11-15-metamaterial-inverse-design-hard/</code> — <code>PREREGISTRATION.md</code>, <code>log.md</code> (chronological, with every stall and miss), <code>metrics.json</code>, <code>results/</code>.</li></ul></div>
</div></body></html>"""
(PAGE / "index.html").write_text(html)
print("page written; figures:", sorted(have))
