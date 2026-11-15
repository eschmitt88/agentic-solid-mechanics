"""Assemble README.md, metrics.json, prediction write-ups and the QA page from the result files.
Run after: analyze_pf.py, rung1_grade.py, figs_pf.py, dmc_post.py, dmc_predict.py."""
import json, os, subprocess, sys

G = json.load(open("results/rung1_grade.json")); P = json.load(open("results/dmc_prediction.json")); M = json.load(open("results/pf_run_metrics.json"))
path = json.load(open("results/dmc/crack_path_validation.json"))
q1 = P["Q1_load_displacement"]; q2 = P["Q2_crack_path"]; unc = P["uncertainty"]; xt = q2["crack_x_mm_near_top_surface"]

# ---------------- metrics.json
rows = {f"{r['id']}_{i}": dict(quantity=r["quantity"], value=r["value"], reference=r["reference"], tolerance=r["tolerance"], passed=r["passed"])
        for i, r in enumerate(G["rows"])}
metrics = dict(
    rung1_all_preregistered_checks_pass=G["all_pass"], rung1_checks_passed=sum(r["passed"] for r in G["rows"]), rung1_checks_total=len(G["rows"]),
    rung1_checks=rows,
    tension_l015_peak_kN=M["t_l015_h16"]["peak_F"], tension_l0075_peak_kN=M["t_l0075_h8"]["peak_F"],
    shear_l0075_peak_kN=M["s_l0075_h8"]["peak_F"], shear_l015_peak_kN=M["s_l015_h8"]["peak_F"],
    shear_crack_angle_deg=M["s_l0075_h4"]["crack_angle_deg"],
    rung2_challenge="Damage Mechanics Challenge (SFC3 data unreachable)", rung2_status="frozen blind prediction, awaiting external grading",
    rung2_pred_peak_load_N=q1["peak_load_N"], rung2_pred_peak_load_1sigma_N=q1["peak_load_1sigma_N"],
    rung2_pred_displacement_at_peak_mm=q1["displacement_at_peak_mm"], rung2_pred_stiffness_N_per_mm=q1["loading_stiffness_30_80pct_N_per_mm"],
    rung2_calibration_rms_log_misfit=unc["model_form_rel"])
json.dump(metrics, open("metrics.json", "w"), indent=1)

# ---------------- README
subprocess.run([sys.executable, "report_src/readme_parts.py"], check=True, stdout=subprocess.DEVNULL)
parts = json.load(open("report_src/readme_parts.json")); parts.update(json.load(open("report_src/readme_static.json")))
extra = json.load(open("results/final_text.json"))
txt = open("report_src/README_template.md").read()
for k, v in parts.items():
    for kk, vv in extra.items():
        v = v.replace(kk, vv)
    txt = txt.replace(k, v.replace('"', "'") if k == "RESULT_LINE" else v)
open("README.md", "w").write(txt)

# ---------------- prediction write-up
cal = "\n".join(f"| {g} | {v['measured_mean_N']:.0f} | {v['model_N']:.0f} | {v['error_pct']:+.1f} % |" for g, v in P["calibration_fit"].items())
sens = "\n".join(f"| {k} | {v:.0f} N |" for k, v in P["sensitivity_peak_load_N"].items())
tr = q2["crack_x_mm_at_heights"]
hs = list(next(iter(tr.values())).keys())
trt = "| height | " + " | ".join(tr.keys()) + " |\n|---|" + "---|" * len(tr) + "\n" + "\n".join(
    f"| {h} mm | " + " | ".join(("%.1f" % tr[f][h]) if tr[f][h] is not None else "–" for f in tr) + " |" for h in hs)
md = f"""# Blind prediction — Damage Mechanics Challenge geometry

**Status: FROZEN 2026-11-15 (UTC), not graded.** Made from the organisers' information packet (15 Jan 2022) and
calibration files only (`raw/dmc/`). This replaces the SFC3 prediction the brief asked for: the SFC3 data package
and problem statement could not be obtained (`raw/sfc3/SOURCE.md`), so the named fallback challenge was used.
Machine-readable version: `dmc_prediction.json`. Curve: `dmc_prediction_Q1.txt`. Crack surface: `dmc_prediction_crack_surface.csv`.

## What was asked (information packet, slides 13–18)
- Q1: force–displacement curve (vertical displacement at the central rod) from first loading through post-peak.
- Q2: crack position as a function of load. Q3: displacement fields of the front and back faces.
- Bonus: coordinates of the fracture surfaces. Discussion: model variability.
Coordinates: x along the beam from the loading rod, y through the width (front −6.35, back +6.35 mm), z up from the top face (bottom z = −25.4 mm).
Challenge notch: 1 mm wide, x = −12.7 mm on the front face to −6.35 mm on the back face (26.6° from the cross-section), depth 5.08 mm (front) tapering to 2.54 mm (back).

## Prediction

| Quantity | Value | Uncertainty |
|---|---|---|
| Peak load | **{q1['peak_load_N']:.0f} N** | 1σ ±{q1['peak_load_1sigma_N']:.0f} N; 80 % interval {q1['peak_load_80pct_interval_mean_of_specimens_N'][0]:.0f}–{q1['peak_load_80pct_interval_mean_of_specimens_N'][1]:.0f} N (mean of the test series), {q1['peak_load_80pct_interval_single_specimen_N'][0]:.0f}–{q1['peak_load_80pct_interval_single_specimen_N'][1]:.0f} N (one specimen) |
| Displacement at peak (as the LVDT would record it, toe included) | {q1['displacement_at_peak_mm']:.2f} mm | 1σ ±{q1['displacement_at_peak_1sigma_mm']:.2f} mm |
| Loading stiffness between 30 % and 80 % of peak | {q1['loading_stiffness_30_80pct_N_per_mm']:.0f} N/mm | calibration beams scatter ±20 % |
| Beam deflection alone at peak (no contact compliance) | {q1['structural_only_displacement_at_peak_mm']:.2f} mm | – |
| Post-peak | {q1['post_peak']} | – |

Crack path (Q2 / bonus), x-position of the crack [mm] on three planes through the width:

{trt}

- Initiation: {q2['initiation']}.
- Shape: {q2['description']}.
- {extra['PATH_STATEMENT']}

Not predicted: Q3 displacement fields, fracture-surface roughness, crack position as a function of load during the
unstable drop (the model breaks the ligament within one displacement increment).

## Model choices
- Method: {P['model']['method']}. Solver: {P['model']['solver']}; {P['model']['elements']:,} elements, h = {P['model']['h_mm']} mm in the crack region, 26 elements through the width.
- Elasticity: isotropic, E = {P['model']['E_MPa']:.0f} MPa, ν = {P['model']['nu']}. E is a compromise between the compression-test slopes
  (≥ ~1 GPa if the cylinders are 25 mm × 51 mm, which the packet does not state) and the ultrasonic moduli (several GPa). It only affects the
  split of the displacement between beam and contact, not the peak load (which depends on E·Gc).
- Fracture: length scale l = {P['model']['l_mm']} mm, Gc = {P['model']['Gc_N_per_mm']} N/mm, one value fitted to the mean peak loads of the four calibration geometries.
  l ≥ 2 mm was rejected because the far-offset beam (HA) then breaks at mid-span instead of at the notch, contrary to the photographs.
- Loading: line displacement under the central rod, line supports 30.48 mm either side; no damage allowed within 2 mm of the rods.
- Displacement record: {P['model']['contact_compliance']}. The LVDT record of the calibration beams is 2–3× softer than the beam itself;
  this empirical term stands for rod indentation and fixture compliance.
- Not modelled: anisotropy of the printed layers (ultrasonic velocities differ by ~15 % between directions), rate effects, friction at the rods,
  dynamics of the unstable crack run.

## Calibration fit (not blind)
| Geometry | measured mean peak [N] | model [N] | error |
|---|---|---|---|
{cal}

## Sensitivity of the predicted peak load (each variant re-calibrated on the same four geometries)
| Variant | Predicted peak |
|---|---|
{sens}

The point prediction is the mean of these variants.

## Uncertainty estimate (my own)
Relative 1σ contributions to the peak load: model form {100*unc['model_form_rel']:.1f} % (scatter of the four calibration geometries about the
one-parameter fit), numerical / length-scale choice {100*unc['numerical_rel']:.1f} %, uncertainty of the calibration means {100*unc['calibration_mean_rel']:.1f} %,
specimen-to-specimen scatter {100*unc['specimen_scatter_cov']:.1f} % (applies to a single specimen). Combined: {100*unc['total_rel_for_mean']:.1f} % for the mean of the
test series, {100*unc['total_rel_single_specimen']:.1f} % for one specimen.
Known direction of bias: the model over-predicts the two offset-notch beams (+3 %, +8 %) and under-predicts the rotated-notch beam (−9 %).
The challenge notch is both offset and rotated, so the two errors should partly cancel, but if one dominates the truth could sit near either
end of the interval. I would not be surprised by anything between 265 and 335 N; I would be surprised by a value outside 240–360 N.
The crack path is less certain than the peak load: {extra['PATH_UNCERTAINTY']}

## Blindness
No outcome information for the challenge geometry was read, searched for, or opened. Before predicting I recorded what I remembered
(`log.md`, 21:00): that a participant paper using a phase-field model exists and reports good agreement; no numbers.
"""
open("results/dmc_prediction.md", "w").write(md)
stub = dict(status="NOT PRODUCED", reason="SFC3 data package requires a Globus login and the open copy of the problem statement was firewall-blocked (HTTP 429); see raw/sfc3/SOURCE.md",
            fallback="Blind prediction made for the Damage Mechanics Challenge instead, as allowed by the brief", see=["results/dmc_prediction.json", "results/dmc_prediction.md"])
json.dump(stub, open("results/sfc3_prediction.json", "w"), indent=1)
open("results/sfc3_prediction.md", "w").write("# SFC3 prediction — not produced\n\nThe Third Sandia Fracture Challenge data package and problem statement could not be obtained "
     "(`raw/sfc3/SOURCE.md`). The blind prediction was made for the fallback challenge named in the brief instead: see `dmc_prediction.md` and `dmc_prediction.json`.\n")

# ---------------- QA page text
qa = dict(
    tile_tension=f"{100*(M['t_l015_h16']['peak_F']/0.734-1):+.0f} % / {100*(M['t_l0075_h8']['peak_F']/0.720-1):+.0f} %",
    tile_shear=f"{100*(M['s_l0075_h8']['peak_F']/0.598-1):+.0f} %",
    grade_note="The crack-path rows use the finest shear run carried to the end of the loading; the finest mesh (stopped earlier) gives 69°.",
    conv_caption=("Left: peak load against mesh size — it keeps falling as the mesh is refined; the stars are the published means; the shear ones were computed on coarse meshes "
                  "(h ≈ l/2). Middle: tension with one staggered pass per step — the standard step (green) is within 0.3 % of the fully iterated answer (dashed). "
                  "Right: shear is not converged at the standard step: a four-times smaller step, or iterating each step, lowers the peak by 2–3 % and moves it to the first hump."),
    traps_caption="Top: final crack for five set-ups of the same shear test. Bottom: their load–displacement curves with Miehe's digitized curve (dashed), which the reference set-up overlays.",
    traps_list=("<ul><li><b>Boundary conditions nobody prints.</b> With the side edges free (what most papers describe) the plate is 27 % too soft and the peak 22 % too low. "
                "Holding the side edges vertically reproduces the published stiffness to three digits. This was the first run, and it was wrong.</li>"
                "<li><b>No tension/compression split.</b> The crack grows upward into the compressed half and damage smears along the top edge.</li>"
                "<li><b>The other common split</b> (volumetric–deviatoric) sends the crack sideways and the plate fails completely — a different answer from a defensible model choice.</li>"
                "<li><b>Mesh size against length scale.</b> Peak load drops 5–8 % from h = l/2 to h = l/8.</li>"
                "<li><b>Staggered-scheme stagnation.</b> Iterating the shear problem to convergence with coarse steps took 348 passes at the onset of cracking and then failed to finish a single step in 3000 passes (run abandoned).</li>"
                "<li><b>Irreversibility.</b> Switching it off changes almost nothing here (+0.4 %): the loading is monotonic, so this benchmark does not test it.</li></ul>"),
    cal_range="−9 % to +8 %",
    path_caption=extra["PATH_CAPTION"],
    caveats=extra["CAVEATS_HTML"])
json.dump(qa, open("results/qa_text.json", "w"), indent=1)
subprocess.run([sys.executable, "build_qa.py"], check=True)
print("finalized")
