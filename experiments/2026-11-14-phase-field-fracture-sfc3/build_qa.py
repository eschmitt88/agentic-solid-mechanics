"""Build docs/qa/trial-8-fracture/index.html (self-contained, trial-5 house style) from the result JSONs."""
import json, os, shutil, html

EXP = os.path.dirname(os.path.abspath(__file__))
QA = os.path.normpath(os.path.join(EXP, "..", "..", "docs", "qa", "trial-8-fracture"))
os.makedirs(QA, exist_ok=True)
G = json.load(open(f"{EXP}/results/rung1_grade.json"))
M = json.load(open(f"{EXP}/results/pf_run_metrics.json"))
P = json.load(open(f"{EXP}/results/dmc_prediction.json"))
FIGS = ["fig_pf_load_displacement.png", "fig_pf_crack_paths.png", "fig_pf_convergence.png", "fig_pf_traps.png",
        "fig_dmc_calibration.png", "fig_dmc_crack_path_validation.png", "fig_dmc_prediction.png", "fig_dmc_geometry.png"]
for f in FIGS:
    if os.path.exists(f"{EXP}/results/{f}"):
        shutil.copyfile(f"{EXP}/results/{f}", f"{QA}/{f}")

CSS = """
:root{--fg:#1c2128;--mut:#636c76;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--bg:#fff;--card:#f6f8fa;--warn:#9a6700}
*{box-sizing:border-box}body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}
.wrap{max-width:1040px;margin:0 auto;padding:28px 22px 80px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:25px;margin:.2em 0 .1em}h2{font-size:18px;margin:1.7em 0 .5em;border-bottom:1px solid var(--bd);padding-bottom:.25em}
h3{font-size:15.5px;margin:1.1em 0 .3em}
.sub{color:var(--mut);margin:0 0 1.4em}
.crumb{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0}
.card.await{background:#fff8c5;border-color:#d4a72c}
.card.warn{background:#fff1e5;border-color:#f0b37e}
img.fig{max-width:100%;border:1px solid var(--bd);border-radius:8px;display:block;background:#fff}
.hint{color:var(--mut);font-size:12.5px;margin:.4em 0 0}
.tw{overflow-x:auto}
table.leg{border-collapse:collapse;width:100%;font-size:14px}
table.leg td,table.leg th{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top;text-align:left}
table.leg th{font-size:12.5px;color:var(--mut);font-weight:600}
table.kv td:first-child{color:var(--mut);width:230px}
.chip{display:inline-block;font-size:12px;font-weight:600;padding:1px 9px;border-radius:20px;white-space:nowrap}
.chip.ok{background:#dafbe1;color:var(--ok)}.chip.bad{background:#ffebe9;color:var(--bad)}.chip.wait{background:#fff8c5;color:var(--warn)}
code{background:#eaeef2;padding:1px 5px;border-radius:5px;font-size:12.5px}
.metric{font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}@media(max-width:760px){.tiles{grid-template-columns:1fr 1fr}}
.tile{background:#fff;border:1px solid var(--bd);border-radius:10px;padding:12px 14px}
.tile .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}.tile .l{font-size:12.5px;color:var(--mut)}
details summary{cursor:pointer;color:var(--accent);font-size:13.5px}
ul{padding-left:20px}li{margin:.25em 0}
"""


def chip(ok):
    return '<span class="chip ok">PASS</span>' if ok else '<span class="chip bad">FAIL</span>'


def fmt(v, unit):
    if v is None:
        return "—"
    if unit == "mm" and abs(v) < 0.1:
        return f"{v:.5f} mm" if v > 2e-3 else f"{v:.2e} mm"
    if unit == "kN":
        return f"{v:.3f} kN"
    if unit == "deg":
        return f"{v:.1f}°"
    if unit == "%":
        return f"{v:.2f} %"
    return f"{v:.3f} {unit}"


def fig(name, alt, cap):
    return f"<div class='card'><img class='fig' src='{name}' alt='{html.escape(alt)}'><p class='hint'>{cap}</p></div>"

rows = ""
for r in G["rows"]:
    ref = "—" if r["id"] in ("C1", "T5") else fmt(r["reference"], r["unit"])
    if r["id"] == "T5":
        ref = "straight, y = 0.5"
    rows += (f"<tr><td>{r['id']}</td><td>{html.escape(r['quantity'])}</td><td class='metric'>{fmt(r['value'], r['unit'])}</td>"
             f"<td class='metric'>{ref}</td><td>{html.escape(r['tolerance'])}</td><td>{chip(r['passed'])}</td><td class='hint'>{html.escape(r['note'])}</td></tr>")
npass = sum(r["passed"] for r in G["rows"]); ntot = len(G["rows"])
q1 = P["Q1_load_displacement"]; q2 = P["Q2_crack_path"]; unc = P["uncertainty"]; mdl = P["model"]
calrows = "".join(f"<tr><td>{g}</td><td class='metric'>{v['measured_mean_N']:.0f} N</td><td class='metric'>{v['model_N']:.0f} N</td><td class='metric'>{v['error_pct']:+.1f} %</td></tr>"
                  for g, v in P["calibration_fit"].items())
sens = "".join(f"<tr><td>{html.escape(k)}</td><td class='metric'>{v:.0f} N</td></tr>" for k, v in P["sensitivity_peak_load_N"].items())
xt = q2["crack_x_mm_near_top_surface"]
extra = json.load(open(f"{EXP}/results/qa_text.json")) if os.path.exists(f"{EXP}/results/qa_text.json") else {}

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial 8 — Fracture: phase-field benchmarks and a blind challenge prediction</title><style>{CSS}</style></head><body><div class="wrap">
<div class="crumb"><a href="../">QA dashboard</a> · <a href="../../">project</a></div>
<h1>Trial 8 — Fracture: phase-field benchmarks, then a blind prediction</h1>
<p class="sub">Can the agent write a fracture solver that reproduces the standard published benchmarks, and then predict a real experiment it has not seen?</p>

<h2>Objective</h2><div class='card'>
<p><b>Rung 1.</b> Write a phase-field fracture solver from scratch and reproduce the two standard benchmarks (a notched square plate pulled apart, and the same plate sheared), graded against curves published by other groups, with the pass tolerance fixed before the first run.</p>
<p><b>Rung 2.</b> Take a real blind challenge: calibrate a model on the organisers' calibration tests only, predict how an unseen specimen breaks, freeze the prediction, and let someone else grade it against the experiment.</p>
<p class='hint'>The brief named the Third Sandia Fracture Challenge (3D-printed steel). Its data package sits behind a Globus login and the only open copy of the problem statement was firewall-blocked, so the fallback named in the brief was used instead: the <b>Damage Mechanics Challenge</b> (3D-printed gypsum beams, organised by Purdue, LLNL and Sandia). Same protocol: calibration data in, blind prediction out.</p></div>

<div class="tiles">
<div class="tile"><div class="v">{npass} / {ntot}</div><div class="l">rung-1 checks passed (tolerances fixed in advance)</div></div>
<div class="tile"><div class="v">{extra.get('tile_tension','')}</div><div class="l">tension peak load vs published mean</div></div>
<div class="tile"><div class="v">{extra.get('tile_shear','')}</div><div class="l">shear peak load vs published mean</div></div>
<div class="tile"><div class="v">{q1['peak_load_N']:.0f} N</div><div class="l">blind prediction of the challenge peak load <span class="chip wait">awaiting grading</span></div></div>
</div>

<h2>Rung 1 — the benchmark and how it is held</h2>
<div class='card'><div class='tw'><table class='leg kv'>
<tr><td>Specimen</td><td>1 mm × 1 mm plate with a horizontal notch from the left edge to the centre (Miehe, Hofacker &amp; Welschinger 2010)</td></tr>
<tr><td>Material</td><td>λ = 121.15, μ = 80.77 kN/mm², fracture energy G<sub>c</sub> = 2.7·10⁻³ kN/mm, length scale l = 0.015 and 0.0075 mm</td></tr>
<tr><td>Tension test</td><td>bottom edge clamped, top edge pulled upward</td></tr>
<tr><td>Shear test</td><td>bottom edge clamped, top edge pushed sideways with no vertical motion; the side edges are also held vertically (see trap 1 below)</td></tr>
<tr><td>Solver</td><td>hand-written: bilinear quads, JAX automatic differentiation on the GPU for residual and tangent, sparse direct solves; staggered displacement / phase-field scheme with a history field for irreversibility and the spectral tension–compression split</td></tr>
<tr><td>Graded configuration</td><td>single-pass staggered scheme with Miehe's load steps on the finest mesh of each series — fixed in <code>PREREGISTRATION.md</code> before the first run</td></tr>
</table></div></div>

<h2>Crack paths</h2>
{fig('fig_pf_crack_paths.png', 'crack paths', "Phase field (dark = broken). Tension: the crack runs straight across. Shear: one crack leaves the notch tip downward and curves to the bottom edge — the pattern every published result shows.")}

<h2>Load–displacement against published results</h2>
{fig('fig_pf_load_displacement.png', 'load displacement', "Lines: this work on successively finer meshes (h = element size, l = length scale). Open symbols: peak points read from published figures. Dashed: Miehe's curve as digitized by a third party. The published values themselves scatter by 10–15 %.")}

<h2>Rung 1 — pass / fail</h2>
<div class='card'><div class='tw'><table class='leg'>
<tr><th>#</th><th>Quantity</th><th>This work</th><th>Reference</th><th>Tolerance</th><th></th><th>Note</th></tr>{rows}</table></div>
<p class='hint'>References are means of values read this session from: Molnár &amp; Gravouil 2017 (HAL hal-02132509), Zhou, Rabczuk &amp; Zhuang (arXiv 1902.05922), Zhang et al. (arXiv 1706.05449, 1805.07357) and Miehe's curve as digitized in the phasefieldx repository. Miehe's own papers are paywalled and were not opened; nothing is graded against memory. {extra.get('grade_note','')}</p></div>

<h2>How the answer moves with mesh and load step</h2>
{fig('fig_pf_convergence.png', 'convergence', extra.get('conv_caption', ''))}

<h2>The known traps, shown rather than avoided</h2>
{fig('fig_pf_traps.png', 'traps', extra.get('traps_caption', ''))}
<div class='card'>{extra.get('traps_list','')}</div>

<h2>Rung 2 — the blind challenge</h2>
<div class='card'><div class='tw'><table class='leg kv'>
<tr><td>Challenge</td><td>Damage Mechanics Challenge: 3D-printed gypsum beams (76.2 × 25.4 × 12.7 mm) broken in three-point bending</td></tr>
<tr><td>Given</td><td>load–displacement curves and fracture surfaces of 16 calibration beams with four other notch shapes (HC, HB, HA, H45), compression and split-cylinder tests, ultrasonic velocities</td></tr>
<tr><td>Asked</td><td>for a beam with an off-centre, skewed, tapered notch: the load–displacement curve, and where the crack goes</td></tr>
<tr><td>Model</td><td>the same phase-field method in 3D (GPU, {mdl['elements']:,} elements): l = {mdl['l_mm']} mm, G<sub>c</sub> = {mdl['Gc_N_per_mm']} N/mm fitted to the four calibration geometries; plus an empirical contact-compliance law for the loading rods</td></tr>
<tr><td>Blindness</td><td>only the organisers' information packet and calibration files were read; the outcome paper and all participant papers were not opened. What the agent remembered beforehand is written in <code>log.md</code> (no outcome numbers).</td></tr>
</table></div></div>
{fig('fig_dmc_geometry.png', 'geometries', "The four calibration geometries and the challenge geometry (notch shown in orange; front face, and seen from below).") if os.path.exists(f"{QA}/fig_dmc_geometry.png") else ''}

<h3>Calibration (not blind)</h3>
{fig('fig_dmc_calibration.png', 'calibration', "Grey: the lab tests. Blue: the model with one fracture energy for all four geometries. Peak loads agree within " + extra.get('cal_range','±9 %') + "; the misfit pattern (too strong for the far-offset notch, too weak for the rotated notch) is carried into the uncertainty.")}
<div class='card'><div class='tw'><table class='leg'><tr><th>Geometry</th><th>Measured mean</th><th>Model</th><th>Error</th></tr>{calrows}</table></div></div>
{fig('fig_dmc_crack_path_validation.png', 'crack path validation', extra.get('path_caption', ''))}

<h3>The prediction</h3>
<div class='card await'><p><b>Awaiting grading.</b> The numbers below were frozen on 2026-11-14 before any outcome data was seen. The orchestrator grades them against the experiment; this section will be replaced by the comparison.</p></div>
{fig('fig_dmc_prediction.png', 'prediction', "Left: predicted load–displacement record with the 80 % band for a single specimen. Middle: predicted crack on the front and back faces. Right: the crack surface seen from above — it starts along the skewed notch and twists into a plane across the beam.")}
<div class='card'><div class='tw'><table class='leg kv'>
<tr><td>Peak load</td><td class='metric'><b>{q1['peak_load_N']:.0f} N</b> &nbsp;(1σ ±{q1['peak_load_1sigma_N']:.0f} N; 80 % interval {q1['peak_load_80pct_interval_mean_of_specimens_N'][0]:.0f}–{q1['peak_load_80pct_interval_mean_of_specimens_N'][1]:.0f} N for the mean of the test series, {q1['peak_load_80pct_interval_single_specimen_N'][0]:.0f}–{q1['peak_load_80pct_interval_single_specimen_N'][1]:.0f} N for one specimen) <span class="chip wait">awaiting grading</span></td></tr>
<tr><td>Displacement at peak</td><td class='metric'>{q1['displacement_at_peak_mm']:.2f} mm (1σ ±{q1['displacement_at_peak_1sigma_mm']:.2f} mm) — mostly rod-contact compliance, only {q1['structural_only_displacement_at_peak_mm']:.2f} mm is beam deflection <span class="chip wait">awaiting grading</span></td></tr>
<tr><td>Loading stiffness</td><td class='metric'>{q1['loading_stiffness_30_80pct_N_per_mm']:.0f} N/mm between 30 % and 80 % of peak <span class="chip wait">awaiting grading</span></td></tr>
<tr><td>After the peak</td><td>{html.escape(q1['post_peak'])}</td></tr>
<tr><td>Crack path</td><td>{html.escape(q2['initiation'])}; {html.escape(q2['description'])}. Near the top surface the crack sits at x ≈ {xt['front (y=-6.35)']:.1f} mm (front), {xt['mid-width (y=0)']:.1f} mm (mid), {xt['back (y=+6.35)']:.1f} mm (back), measured from the loading rod. <span class="chip wait">awaiting grading</span></td></tr>
<tr><td>Not predicted</td><td>surface displacement fields (challenge question 3) and fracture-surface roughness</td></tr>
</table></div>
<h3>Sensitivity of the predicted peak load</h3><div class='tw'><table class='leg'><tr><th>Model variant (each re-calibrated)</th><th>Predicted peak</th></tr>{sens}</table></div>
<p class='hint'>Uncertainty budget (relative, 1σ): model form {100*unc['model_form_rel']:.1f} % (scatter of the four calibration geometries about the fit), numerical / length-scale choice {100*unc['numerical_rel']:.1f} %, specimen-to-specimen {100*unc['specimen_scatter_cov']:.1f} %.</p></div>

<h2>Honest caveats</h2><div class='card warn'>{extra.get('caveats','')}</div>

<details><summary>Terms used on this page</summary><div class='card'><div class='tw'><table class='leg kv'>
<tr><td><b>phase-field fracture</b></td><td>A way to simulate cracks without tracking them explicitly: a smooth field d runs from 0 (intact) to 1 (broken) and the crack is the narrow band where d ≈ 1.</td></tr>
<tr><td><b>length scale l</b></td><td>The width of that band. In this model it also sets the material's strength, so it is a material parameter, not just a numerical one.</td></tr>
<tr><td><b>fracture energy G<sub>c</sub></b></td><td>Energy needed to create a unit area of crack.</td></tr>
<tr><td><b>mesh size h</b></td><td>Element edge length. The band must be resolved (h well below l) or the crack looks tougher than it is.</td></tr>
<tr><td><b>staggered scheme</b></td><td>Solving for displacements and for the crack field alternately. With one pass per load step the crack lags the load; iterating to convergence removes the lag but can take hundreds of passes when the crack runs.</td></tr>
<tr><td><b>tension/compression split</b></td><td>Only the tensile part of the strain energy is allowed to drive (and be released by) cracking; without it material also "cracks" in compression.</td></tr>
<tr><td><b>irreversibility</b></td><td>A crack cannot heal. Enforced here by driving the crack field with the largest tensile energy seen so far (the history field).</td></tr>
<tr><td><b>three-point bending</b></td><td>A beam resting on two rods and pushed down by a third at mid-span.</td></tr>
<tr><td><b>mode I / II / III</b></td><td>Opening, in-plane sliding and out-of-plane tearing of a crack. An off-centre notch mixes I and II; a skewed notch adds III.</td></tr>
<tr><td><b>blind prediction</b></td><td>A prediction frozen before the experimental outcome is seen by the predictor.</td></tr>
</table></div></div></details>

<h2>Artifacts</h2><div class='card'><p class='hint'><a href='https://github.com/eschmitt88/agentic-solid-mechanics/tree/master/experiments/2026-11-14-phase-field-fracture-sfc3'>experiment folder on GitHub</a> — README, PREREGISTRATION.md, log.md (including dead ends), solver code (<code>pf.py</code>, <code>pf3d.py</code>, <code>dmc.py</code>), <code>results/dmc_prediction.json</code>.</p></div>
</div></body></html>"""
open(f"{QA}/index.html", "w").write(page)
print("wrote", f"{QA}/index.html", len(page), "bytes")
