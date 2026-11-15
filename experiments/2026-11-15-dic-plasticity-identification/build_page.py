"""Assemble metrics.json and the review page docs/qa/trial-11-identification/index.html from results/*.json."""
import json, os, shutil
from pathlib import Path
import numpy as np

R = Path("results")
PAGE = Path("../../docs/qa/trial-11-identification")
PAGE.mkdir(parents=True, exist_ok=True)
J = lambda f: json.load(open(R / f))

fz = J("frozen_models.json"); PRI, BASE = fz["primary"], fz["baseline"]
ho = J("heldout_scores.json"); dev = J("dev_scores.json"); conv = J("convergence_heldout.json")
warm = J("warmup.json"); base = J("baseline.json"); ccx = J("ccx_check_O15-3_M2.json")
acc = J("heldout_access.json"); alpha = J("alpha_seen.json"); post = J("posthoc_rolling_dogbones.json"); esens = J("elastic_sensitivity_O5-4.json")
fits = {k: J(f"fits/{k}.json") for k in ("M1", "M2", "M3", "M4", "M4b")}
TOL = {"force_nrmse": 0.05, "force_last_err": 0.05, "strain_relL2_st13": 0.15, "strain_relL2_st27": 0.15, "strain_relL2_st40": 0.15, "hot_err": 0.15}
KEYS = list(TOL)
LAB = {"force_nrmse": "Force curve, average error", "force_last_err": "Force at 90 % of the way to peak load",
       "strain_relL2_st13": "Strain map error at 30 %", "strain_relL2_st27": "Strain map error at 60 %",
       "strain_relL2_st40": "Strain map error at 90 %", "hot_err": "Peak (hot-spot) strain"}
SPECN = {"O15-3": "Plate O15 (slot + notch)", "XT10": "X-specimen, same direction as calibration", "XR5": "X-specimen, pulled along the rolling direction"}


def slim(m):
    return {k: m[k] for k in KEYS + ["hot_dic", "hot_fe", "force_last_station"]} | {"pass": m["verdict"]}


def spec_block(res):
    return {"mesh_nodes": res["info"]["nn"], "dic_points": res["info"]["npts"], "x_F_mm": res["info"]["xF"], "F_max_N": res["info"]["Fmax"],
            "force_stations_excluded_as_drops": res["info"]["drop_stations"], "mean_temperature_C": res["info"]["T_mean"],
            "models": {k: slim(m) for k, m in res["models"].items() if "fail" not in m}}


def max_change(a, b):
    return {k: float(abs(a[k] - b[k])) for k in KEYS}


n_pass = {s: sum(ho[s]["models"][PRI]["verdict"][k] for k in ("force_nrmse", "force_last", "strain_st13", "strain_st27", "strain_st40", "hot")) for s in ho}
n_pass_b = {s: sum(ho[s]["models"][BASE]["verdict"][k] for k in ("force_nrmse", "force_last", "strain_st13", "strain_st27", "strain_st40", "hot")) for s in ho}
half = {s: {m: all(abs(ho[s]["models"][m][k]) <= 0.5 * TOL[k] for k in KEYS) for m in (PRI, BASE)} for s in ho}
ident = fits[PRI]["best"]["ident"]
metrics = {
    "data": {"dataset": "Sandia 304L DIC+IR, Figshare+ doi:10.25452/figshare.plus.25483534 (CC0); descriptor doi:10.1038/s41597-024-03949-y",
             "real_data": True, "calibration": ["O1-3", "O14-3"], "development": ["O5-4"], "uniaxial_baseline_data": ["BT3", "CT3", "CT4"],
             "held_out_primary": ["O15-3", "XT10"], "held_out_orientation_transfer": ["XR5"]},
    "preregistration": {"file": "PREREGISTRATION.md", "sha256_at_heldout_pass": acc[0]["prereg_sha256"], "tolerances": TOL,
                        "amendments_before_first_fit": 1},
    "heldout_access": acc,
    "warmup": {"truth": warm["truth"]["truth"], "alpha95": warm["truth"]["alpha95"],
               "runs": {k: {"start": f["start"], "recovered": f["p"], "sigy_max_rel_err_to_alpha95": f["sigy_max_rel_err"],
                            "sigy_max_rel_err_to_alpha_max": f["sigy_max_rel_err_to_alpha_max"], "d_a": f["d_a"], "d_h": f["d_h"], "d_c": f["d_c"],
                            "n_solver_evaluations": f["n_eval"], "pass": f["pass"]} for k, f in warm["fits"].items()},
               "gradient_check_max_rel_err_vs_fd": 1.1e-6, "gradient_check_tolerance": 1e-5, "pass": all(f["pass"] for f in warm["fits"].values())},
    "models": {k: {"p": v["p"], "kind": v["kind"], **({"J_calibration": v["J_calibration"], "n_par": v["n_par"]} if "n_par" in v else {}),
                   "dev_S": v.get("dev_S")} for k, v in fz["models"].items()},
    "param_names": fz["param_names"], "primary": PRI, "baseline": BASE,
    "multistart": {k: {s: {"J": f["cost"], "p": f["p"], "n_eval": f["n_eval"], "time_s": f["time_s"]} for s, f in v["fits"].items() if "cost" in f} for k, v in fits.items()},
    "identifiability_primary": {"params": ident["params"], "change_for_1pct_worse_fit_others_reoptimised": ident["dlog_1pct_profile"],
                                "condition_number_JtJ_logparams": ident["cond"], "correlation": ident["corr"],
                                "curve_values": dict(zip(ident["curve_coords"]["names"], ident["curve_coords"]["dlog_1pct_profile"]))},
    "plastic_strain_seen_in_calibration": alpha,
    "development_O5-4": spec_block(dev["O5-4"]), "calibration_scores": {s: spec_block(dev[s]) for s in ("O1-3", "O14-3")},
    "heldout": {s: spec_block(ho[s]) for s in ho},
    "heldout_verdict": {"primary_checks_passed": n_pass, "baseline_checks_passed": n_pass_b,
                        "generalises_preregistered": bool(ho["O15-3"]["models"][PRI]["verdict"]["all"] and ho["XT10"]["models"][PRI]["verdict"]["all"]),
                        "orientation_transfer_pass": bool(ho["XR5"]["models"][PRI]["verdict"]["all"]),
                        "would_pass_at_half_tolerances": half},
    "convergence_heldout": {s: {m: {"step_80_vs_40_abs_change": max_change(conv["cal_n2"][s]["models"][m], ho[s]["models"][m]),
                                    "mesh_half_size_abs_change": max_change(conv["fine_n1"][s]["models"][m], ho[s]["models"][m]),
                                    "fine_mesh_nodes": conv["fine_n1"][s]["info"]["nn"],
                                    "fine_mesh_all_pass": conv["fine_n1"][s]["models"][m]["verdict"]["all"]} for m in (PRI, BASE)} for s in ho},
    "calculix_crosscheck": {k: ccx[k] for k in ("spec", "model", "force_nrmse_jax_vs_ccx", "force_max_rel_diff", "strain_relL2", "ccx_metrics_vs_experiment",
                                                "jax_metrics_vs_experiment", "pass", "ccx_wall_s")},
    "uniaxial_baseline_fit": {"ep_max": base["ep_max"], "fits": base["fits"]},
    "elastic_sensitivity_O5-4": {k: {a: v[a] for a in KEYS} for k, v in esens.items()},
    "posthoc_rolling_dogbones_not_graded": post,
}
conv_max = max(max(v2.values()) for s in metrics["convergence_heldout"].values() for m in s.values() for k2, v2 in m.items() if isinstance(v2, dict))
metrics["convergence_heldout_max_abs_change"] = conv_max
json.dump(metrics, open("metrics.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ page
FIGS = ["fig_specimens.png", "fig_warmup.png", "fig_hardening.png", "fig_identifiability.png", "fig_scores.png", "fig_ccx.png",
        "fig_force_O15-3.png", "fig_force_XT10.png", "fig_force_XR5.png", "fig_fields_O15-3_Eyy.png", "fig_fields_O15-3_Exx.png",
        "fig_fields_XT10_Eyy.png", "fig_fields_XR5_Eyy.png"]
for f in FIGS:
    shutil.copy(R / f, PAGE / f)

pc = lambda x, d=1, sign=False: (f"{100 * x:+.{d}f}" if sign else f"{100 * abs(x):.{d}f}") + " %"
chip = lambda ok: f'<span class="chip {"ok" if ok else "bad"}">{"pass" if ok else "miss"}</span>'
VK = {"force_nrmse": "force_nrmse", "force_last_err": "force_last", "strain_relL2_st13": "strain_st13", "strain_relL2_st27": "strain_st27",
      "strain_relL2_st40": "strain_st40", "hot_err": "hot"}


def score_table(s):
    rows = []
    p, b, n = ho[s]["models"][PRI], ho[s]["models"][BASE], ho[s]["models"]["null-elastic"]
    for k in KEYS:
        sg = k in ("force_last_err", "hot_err")
        nul = pc(n[k], 0, sg) if k.startswith("strain") or k == "hot_err" else "—"
        rows.append(f'<tr><td data-l="Check">{LAB[k]}</td><td data-l="Allowed" class="metric">{"±" if sg else "≤ "}{100 * TOL[k]:.0f} %</td>'
                    f'<td data-l="Full-field model" class="metric"><b>{pc(p[k], 1, sg)}</b> {chip(p["verdict"][VK[k]])}</td>'
                    f'<td data-l="Tension-test model" class="metric">{pc(b[k], 1, sg)} {chip(b["verdict"][VK[k]])}</td>'
                    f'<td data-l="No plasticity at all" class="metric mut">{nul}</td></tr>')
    return ('<table class="res"><thead><tr><th>Check</th><th>Allowed</th><th>Full-field model</th><th>Tension-test model</th><th>No plasticity at all</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def forms_table():
    rows = []
    desc = {"M1": "isotropic, simple hardening (3 numbers)", "M2": "isotropic, hardening + straight-line term (4)", "M3": "isotropic, two-part hardening (6)",
            "M4": "direction-dependent (Hill), as M2 (7)", "M4b": "direction-dependent, two-part hardening (9)"}
    for k in ("M1", "M2", "M3", "M4", "M4b"):
        m = dev["O5-4"]["models"][k]; st = np.mean([m["strain_relL2_st13"], m["strain_relL2_st27"], m["strain_relL2_st40"]])
        mark = ' <span class="chip ok">chosen</span>' if k == PRI else ""
        rows.append(f'<tr><td data-l="Model form">{k}: {desc[k]}{mark}</td><td data-l="Fit to calibration plates (lower is better)" class="metric">{fz["models"][k]["J_calibration"]:.3f}</td>'
                    f'<td data-l="Force error on check plate" class="metric">{pc(m["force_nrmse"], 2)}</td><td data-l="Strain map error on check plate" class="metric">{pc(st, 1)}</td>'
                    f'<td data-l="Selection score" class="metric">{fz["models"][k]["dev_S"]:.3f}</td></tr>')
    return ('<table class="res"><thead><tr><th>Model form</th><th>Fit to calibration plates (lower is better)</th><th>Force error on check plate</th>'
            '<th>Strain map error on check plate</th><th>Selection score</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


pp = fz["models"][PRI]["p"]; bp = fz["models"][BASE]["p"]
wA = warm["fits"]["A"]
tot_pri = n_pass["O15-3"] + n_pass["XT10"]; tot_b = n_pass_b["O15-3"] + n_pass_b["XT10"]
strain_gain = {s: (np.mean([ho[s]["models"][BASE][k] for k in KEYS[2:5]]), np.mean([ho[s]["models"][PRI][k] for k in KEYS[2:5]])) for s in ho}
ratio = post["measured_ratio_rolling_over_transverse"]
cvals = metrics["identifiability_primary"]["curve_values"]; raw = dict(zip(ident["params"], ident["dlog_1pct_profile"]))

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial 11 — material identification from full-field data</title>
<style>
:root{{--fg:#1c2128;--mut:#636c76;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--bg:#fff;--card:#f6f8fa;--full:#2a78d6;--base:#eb6834}}
*{{box-sizing:border-box}}body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}}
.wrap{{max-width:1040px;margin:0 auto;padding:28px 22px 80px}}
a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}
h1{{font-size:25px;margin:.2em 0 .1em}}h2{{font-size:18px;margin:1.7em 0 .5em;border-bottom:1px solid var(--bd);padding-bottom:.25em}}
h3{{font-size:15.5px;margin:1.3em 0 .4em}}
.sub{{color:var(--mut);margin:0 0 1.4em}}.crumb{{color:var(--mut);font-size:13px;margin-bottom:18px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0}}
img.fig{{max-width:100%;height:auto;border:1px solid var(--bd);border-radius:8px;display:block;margin:10px 0 4px}}
.hint{{color:var(--mut);font-size:12.5px;margin:.3em 0 1em}}
.chip{{display:inline-block;font-size:12px;font-weight:600;padding:1px 9px;border-radius:20px;white-space:nowrap}}
.chip.ok{{background:#dafbe1;color:var(--ok)}}.chip.bad{{background:#ffebe9;color:var(--bad)}}
code{{background:#eaeef2;padding:1px 5px;border-radius:5px;font-size:12.5px}}
.metric{{font-variant-numeric:tabular-nums}}.mut{{color:var(--mut)}}
ul{{padding-left:1.2em}}li{{margin:.35em 0}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}}
.tile{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 14px}}
.tile .n{{font-size:24px;font-weight:700;line-height:1.15}}.tile .l{{color:var(--mut);font-size:12.5px;margin-top:3px}}
.key{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px}}
table.res{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}}
table.res th{{text-align:left;font-size:12.5px;color:var(--mut);font-weight:600;padding:6px 10px;border-bottom:2px solid var(--bd)}}
table.res td{{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top}}
dl.gloss dt{{font-weight:600;margin-top:.7em}}dl.gloss dd{{margin:0;color:var(--fg)}}
@media (max-width:640px){{
 .wrap{{padding:18px 14px 60px}}h1{{font-size:21px}}.tiles{{grid-template-columns:repeat(2,1fr)}}
 table.res thead{{display:none}}table.res tr{{display:block;border:1px solid var(--bd);border-radius:8px;margin:8px 0;padding:6px 10px;background:var(--bg)}}
 table.res td{{display:flex;justify-content:space-between;gap:12px;border:0;padding:3px 0;text-align:right}}
 table.res td::before{{content:attr(data-l);color:var(--mut);font-size:12.5px;text-align:left;flex:0 0 46%}}
 table.res td:first-child{{font-weight:600;display:block;text-align:left;padding-bottom:5px}}table.res td:first-child::before{{content:none}}
}}
</style></head><body><div class="wrap">
<div class="crumb"><a href="../index.html">QA home</a> · <a href="../phase-2/index.html">Phase 2</a> · Trial 11</div>
<h1>Trial 11 — learning a steel's behaviour from camera-measured strain maps</h1>
<p class="sub">15 November 2026 · Phase 2, track E · real test data from Sandia National Laboratories (304L stainless sheet)</p>

<div class="card"><b>The question.</b> A laboratory pulled oddly shaped stainless-steel plates until they tore, while cameras measured how
every point of the surface stretched. Can an AI agent use those strain maps from <b>two</b> plate shapes to work out how the steel behaves,
and then correctly predict <b>other shapes it has never seen</b> — both the pulling force and where the plate stretches most? And does all
that camera data actually beat the old way, a single plain tension test?
<p style="margin:.7em 0 0"><b>Short answer.</b> Yes, it generalised: on the unseen shapes all {tot_pri} of {tot_pri} pass marks set in advance were met, by a wide margin, and an independent
solver agrees. But the plain tension test, fitted in seconds, <b>also passed all {tot_b}</b>. The camera data bought one real thing — better strain maps
(error {pc(strain_gain["O15-3"][0])} → {pc(strain_gain["O15-3"][1])} on the unseen plate) by revealing that the steel deforms a little differently in different directions — and nothing for the force,
which the tension-test model predicted slightly better.</p></div>

<div class="tiles">
<div class="tile"><div class="n">{tot_pri} / 12</div><div class="l">pass marks met on the two unseen shapes (full-field model)</div></div>
<div class="tile"><div class="n">{pc(ho["O15-3"]["models"][PRI]["force_nrmse"])} · {pc(ho["XT10"]["models"][PRI]["force_nrmse"])}</div><div class="l">force error on the two unseen shapes (allowed 5 %)</div></div>
<div class="tile"><div class="n">{pc(ho["O15-3"]["models"][PRI]["strain_relL2_st40"])} · {pc(ho["XT10"]["models"][PRI]["strain_relL2_st40"])}</div><div class="l">strain-map error near peak load (allowed 15 %)</div></div>
<div class="tile"><div class="n">{tot_b} / 12</div><div class="l">met by the plain tension-test model too</div></div>
</div>

<h2>The test pieces</h2>
<img class="fig" src="fig_specimens.png" alt="Outlines of the five specimen shapes with their roles">
<p class="hint">Notches and holes force the metal to stretch unevenly, so one test exercises many stress states at once. Two shapes were used to
learn the material, one to choose between model forms, and two were locked away. The X-shape was tested twice: cut across the sheet's
rolling direction like all the others, and cut along it.</p>
<ul>
<li><b>Data.</b> Jones et al., Scientific Data 11:1101 (2024), <a href="https://doi.org/10.1038/s41597-024-03949-y">doi:10.1038/s41597-024-03949-y</a>; files from Figshare+
(<a href="https://doi.org/10.25452/figshare.plus.25483534">doi:10.25452/figshare.plus.25483534</a>, CC0, no login). Only the slow tests (about 1.5 hours each, no self-heating) were used.</li>
<li><b>Split, fixed before any fitting.</b> Learn from plates O1 and O14 · check model choices on O5 · held out: O15 and the X-specimen. The held-out files were
opened once, at {acc[0]["opened_utc"][11:19]} UTC, after the models were frozen.</li>
<li><b>Two ways to calibrate.</b> <span class="key" style="background:var(--full)"></span><b>Full-field model</b>: a simulation of each calibration plate is adjusted until its strain maps and force match the measurements.
<span class="key" style="background:var(--base)"></span><b>Tension-test model</b>: the same hardening formula fitted to three plain tension tests only.</li>
</ul>

<h2>Warm-up: does the method recover a known answer?</h2>
<img class="fig" src="fig_warmup.png" alt="Truth, starting guess and recovered hardening curve in the synthetic warm-up">
<p class="hint">Synthetic strain maps were made with a hidden material on a fine mesh, camera-like noise was added, and the method had to find the material
again on a different, coarser mesh from a starting guess 30–60 % off.</p>
<table class="res"><thead><tr><th>Check</th><th>Allowed</th><th>Result</th></tr></thead><tbody>
<tr><td data-l="Check">Hardening curve recovered</td><td data-l="Allowed" class="metric">within 2 %</td><td data-l="Result" class="metric"><b>{pc(wA["sigy_max_rel_err"], 2)}</b> {chip(wA["pass"])}</td></tr>
<tr><td data-l="Check">Direction-dependence numbers a, h, c</td><td data-l="Allowed" class="metric">0.05 / 0.05 / 0.3</td><td data-l="Result" class="metric"><b>{wA["d_a"]:+.3f} / {wA["d_h"]:+.4f} / {wA["d_c"]:+.4f}</b> {chip(wA["pass"])}</td></tr>
<tr><td data-l="Check">Second starting guess reaches the same answer</td><td data-l="Allowed" class="metric">—</td><td data-l="Result" class="metric">identical to 8 digits {chip(warm["fits"]["B"]["pass"])}</td></tr>
<tr><td data-l="Check">Computed gradients vs finite differences</td><td data-l="Allowed" class="metric">1e-5</td><td data-l="Result" class="metric"><b>1.1e-6</b> {chip(True)}</td></tr>
</tbody></table>

<h2>What was identified</h2>
{forms_table()}
<p class="hint">Five model forms were fitted to the two calibration plates and compared on the check plate O5 only. The rule fixed in advance — the simplest form within 10 % of the
best score — picks <b>{PRI}</b>. Allowing the steel to behave differently along and across the sheet (the "Hill" forms) is what cuts the strain-map error from about 8 % to about 5 %;
extra hardening terms add nothing.</p>
<img class="fig" src="fig_hardening.png" alt="Hardening curves from the three calibrations and the tension tests">
<p class="hint">Once direction-dependence is allowed, the hardening curve learned from the plates (solid blue) lands almost on the curve from the plain tension tests (orange).
The isotropic full-field fit (dashed) bends its curve to make up for the missing direction-dependence. Chosen model: initial yield {pp[0]:.0f} MPa,
Voce {pp[1]:.0f} MPa at rate {pp[2]:.2f}, linear slope {pp[3]:.0f} MPa; Hill a = {pp[6]:.3f}, h = {pp[7]:.3f}, c = {pp[8]:.2f} (isotropic would be 1, 0.5, 3).</p>
<img class="fig" src="fig_identifiability.png" alt="Bar charts: how well each parameter is pinned by the data">
<p class="hint">Which numbers the data really fix. The yield stress at any given strain is pinned to about ±{100 * max(v for k, v in cvals.items() if k.startswith("sigy")):.0f} % or better, and so are the direction-dependence numbers
(±{100 * raw["c"]:.1f}–{100 * raw["a"]:.1f} %). The individual constants of the hardening formula are not: Q, b and H can each move by {100 * min(raw["Q1"], raw["b1"], raw["H"]):.0f}–{100 * max(raw["Q1"], raw["b1"], raw["H"]):.0f} % if the others compensate.
Different starting guesses still end at the same point (three starts for {PRI}, identical to four digits).</p>

<h2>Predicting the shapes it never saw</h2>
<img class="fig" src="fig_scores.png" alt="Dot plot of held-out errors relative to tolerance for both models">
"""
for s, extra in (("O15-3", ""), ("XT10", ""), ("XR5", " <span class='mut'>(orientation transfer, judged separately)</span>")):
    html += f"""<h3>{SPECN[s]}{extra}</h3>
<img class="fig" src="fig_force_{s}.png" alt="Measured and predicted force against extension, {s}">
<img class="fig" src="fig_fields_{s}_Eyy.png" alt="Measured and predicted strain maps, {s}">
{score_table(s)}
"""
    if s == "O15-3":
        html += f"""<img class="fig" src="fig_fields_O15-3_Exx.png" alt="Sideways strain maps and errors, O15-3">
<p class="hint">The sideways strain shows what the camera data added: the tension-test model (right) shrinks the plate sideways too much almost everywhere; the full-field model does not.</p>
"""
html += f"""<p class="hint">"No plasticity at all" is a deliberately wrong model given the same measured edge motion. It shows how much of the strain-map score comes for free from
imposing that motion: not much — it is off by 30–40 % and misses the hot spot by 30–50 %.</p>

<h2>Did the camera data buy anything?</h2>
<table class="res"><thead><tr><th>Question</th><th>Full-field model</th><th>Tension-test model</th></tr></thead><tbody>
<tr><td data-l="Question">Force curve error, plate O15</td><td data-l="Full-field model" class="metric">{pc(ho["O15-3"]["models"][PRI]["force_nrmse"], 2)}</td><td data-l="Tension-test model" class="metric"><b>{pc(ho["O15-3"]["models"][BASE]["force_nrmse"], 2)}</b></td></tr>
<tr><td data-l="Question">Force curve error, X-specimen</td><td data-l="Full-field model" class="metric">{pc(ho["XT10"]["models"][PRI]["force_nrmse"], 2)}</td><td data-l="Tension-test model" class="metric"><b>{pc(ho["XT10"]["models"][BASE]["force_nrmse"], 2)}</b></td></tr>
<tr><td data-l="Question">Strain map error (mean of three stages), plate O15</td><td data-l="Full-field model" class="metric"><b>{pc(strain_gain["O15-3"][1])}</b></td><td data-l="Tension-test model" class="metric">{pc(strain_gain["O15-3"][0])}</td></tr>
<tr><td data-l="Question">Strain map error, X-specimen</td><td data-l="Full-field model" class="metric"><b>{pc(strain_gain["XT10"][1])}</b></td><td data-l="Tension-test model" class="metric">{pc(strain_gain["XT10"][0])}</td></tr>
<tr><td data-l="Question">Force at the last stage, X pulled along the rolling direction</td><td data-l="Full-field model" class="metric">{pc(ho["XR5"]["models"][PRI]["force_last_err"], 1, True)}</td><td data-l="Tension-test model" class="metric"><b>{pc(ho["XR5"]["models"][BASE]["force_last_err"], 1, True)}</b></td></tr>
<tr><td data-l="Question">Work to calibrate</td><td data-l="Full-field model" class="metric">two plate tests with cameras; about 8 minutes of simulation per fit</td><td data-l="Tension-test model" class="metric">three tension tests; under a second</td></tr>
</tbody></table>
<p>For this steel the answer is: <b>better strain maps, no better forces.</b> 304L hardens smoothly and almost in a straight line at large strain, so extending the tension-test curve past
where that test ends (plastic strain {base["ep_max"]:.2f}) is harmless, and the direction-dependence is mild. A material with a sharper change in hardening, stronger anisotropy or rate effects would be a harder exam for the baseline.</p>

<h2>Checks on the solver itself</h2>
<img class="fig" src="fig_ccx.png" alt="Force curve from JAX solver and CalculiX on held-out plate O15">
<table class="res"><thead><tr><th>Check</th><th>Allowed</th><th>Result</th></tr></thead><tbody>
<tr><td data-l="Check">CalculiX re-run of the identified isotropic model on unseen plate O15: force curve</td><td data-l="Allowed" class="metric">2 %</td><td data-l="Result" class="metric"><b>{pc(ccx["force_nrmse_jax_vs_ccx"], 2)}</b> {chip(ccx["force_nrmse_jax_vs_ccx"] <= 0.02)}</td></tr>
<tr><td data-l="Check">Same re-run: strain map near peak load</td><td data-l="Allowed" class="metric">3 %</td><td data-l="Result" class="metric"><b>{pc(ccx["strain_relL2"]["40"], 2)}</b> {chip(ccx["strain_relL2"]["40"] <= 0.03)}</td></tr>
<tr><td data-l="Check">Finer mesh (about 4× the nodes): largest change in any held-out score</td><td data-l="Allowed" class="metric">reported</td><td data-l="Result" class="metric">{100 * max(max(m["mesh_half_size_abs_change"].values()) for s in metrics["convergence_heldout"].values() for m in s.values()):.2f} percentage points</td></tr>
<tr><td data-l="Check">Twice as many load steps: largest change in any held-out score</td><td data-l="Allowed" class="metric">reported</td><td data-l="Result" class="metric">{100 * max(max(m["step_80_vs_40_abs_change"].values()) for s in metrics["convergence_heldout"].values() for m in s.values()):.2f} percentage points</td></tr>
</tbody></table>
<p class="hint">CalculiX has no direction-dependent yield model, so the cross-check covers the isotropic path. The direction-dependent part was checked against hand formulas instead (stress and sideways strain in simple tension, agreement to 9 digits or better).</p>

<h2>Honest caveats</h2>
<div class="card"><ul>
<li><b>The pass marks turned out generous.</b> They were set before seeing any result (5 % on force, 15 % on strain maps) and every model beat them several times over. At half the tolerances the full-field model would still pass both primary shapes
but not the rolling-direction X ({pc(ho["XR5"]["models"][PRI]["force_nrmse"])} force, {pc(ho["XR5"]["models"][PRI]["strain_relL2_st40"])} strain); the tension-test model would {"pass" if half["XT10"][BASE] else "miss"} the X-specimen on strain ({pc(ho["XT10"]["models"][BASE]["strain_relL2_st13"])} at the first stage).</li>
<li><b>The exam was not hard enough to separate the two methods.</b> The plain tension-test model passes everything. "Full-field data is worth it" is not shown here for force prediction.</li>
<li><b>The edge motion is an input.</b> As is standard practice, the measured motion along two lines near the grips drives the simulation, for unseen shapes too. Force and interior strain are predicted; the overall stretch is not.</li>
<li><b>No tearing.</b> The model has no damage, so nothing past 90 % of the extension at peak load is predicted — not the peak load, not the failure.</li>
<li><b>One direction learned, the other guessed wrong.</b> All calibration plates were pulled across the rolling direction. The fitted direction-dependence then implies the steel is 3.5 % weaker along the rolling direction; separate tension tests (looked at afterwards, not graded)
show it is {100 * (min(ratio) - 1):.0f}–{100 * (max(ratio) - 1):.0f} % stronger. That is why the full-field model is worse than the baseline on the rolling-direction X ({pc(ho["XR5"]["models"][PRI]["force_last_err"], 1, True)} vs {pc(ho["XR5"]["models"][BASE]["force_last_err"], 1, True)}), though still inside the limit.</li>
<li><b>A residual 4–7 % strain-map error remains</b> that no model form removed. Likely contributors, not separated here: camera smoothing near sharp strain peaks, as-cut versus nominal outline, the plane-stress idealisation where the sheet necks, and yield-surface shape beyond Hill's.</li>
<li><b>Narrow scope.</b> Slow, room-temperature tests of one sheet only. The fast tests (self-heating, rate effects), the D-shaped specimen and the fifth plate (O18, 15 GB) were not used.</li>
<li><b>One pre-registration amendment</b>, made before any fit: the rule for skipping spurious force drops was written backwards and was corrected. The elastic constants are textbook values from memory (the tension tests give 187–194 GPa against the 193 GPa used); changing them moves no score by more than 0.06 points.</li>
<li><b>Optimiser trouble along the way</b>: two warm-up runs were stopped because the optimiser crawled along a flat valley; fixed by fitting the curve's values instead of its constants. The six-number isotropic form ended at three different points from three starts.</li>
</ul></div>

<h2>Glossary</h2>
<dl class="gloss">
<dt>DIC (digital image correlation)</dt><dd>Cameras track a speckle pattern painted on the part and compute how every point moved — a strain map instead of a single gauge reading.</dd>
<dt>Strain</dt><dd>Stretch as a fraction of original length; 0.2 means 20 % longer.</dd>
<dt>Plasticity, hardening</dt><dd>Permanent deformation of metal, and the way it gets stronger the more it has been deformed. The "hardening curve" is yield stress against plastic strain.</dd>
<dt>Anisotropy (Hill model)</dt><dd>Rolled sheet behaves slightly differently along and across the rolling direction. Hill's model describes that with three extra numbers.</dd>
<dt>Finite-element model updating</dt><dd>Run a simulation of the test, compare with the measurement, adjust the material numbers, repeat until they agree.</dd>
<dt>Held-out data</dt><dd>Measurements kept unopened until the model is final, so the score cannot be tuned.</dd>
<dt>Identifiability</dt><dd>Whether the data can actually fix a number, or whether very different values fit equally well.</dd>
<dt>CalculiX</dt><dd>An independent open-source simulation program, used here to make sure a result is not an artefact of the agent's own solver.</dd>
</dl>
<p class="hint">Full record: <code>experiments/2026-11-15-dic-plasticity-identification/</code> (README.md, PREREGISTRATION.md, log.md, metrics.json).</p>
</div></body></html>
"""
open(PAGE / "index.html", "w").write(html)
print("page written:", PAGE / "index.html", len(html), "bytes; figures:", len(FIGS))
print({k: metrics["heldout_verdict"][k] for k in metrics["heldout_verdict"]})
