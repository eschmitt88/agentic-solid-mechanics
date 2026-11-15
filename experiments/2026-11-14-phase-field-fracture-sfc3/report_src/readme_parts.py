import json
G=json.load(open("results/rung1_grade.json")); P=json.load(open("results/dmc_prediction.json")); M=json.load(open("results/pf_run_metrics.json"))
q1=P["Q1_load_displacement"]; q2=P["Q2_crack_path"]; unc=P["uncertainty"]; xt=q2["crack_x_mm_near_top_surface"]
path=json.load(open("results/dmc/crack_path_validation.json"))
npass=sum(r["passed"] for r in G["rows"]); ntot=len(G["rows"])
table=open("results/rung1_grade.md").read()
def g(t,k): return M[t][k]
pm=lambda d: ", ".join(f"{k}: {('%.1f' % v) if v is not None else 'n/a'} mm" for k,v in d.items())
RESULT_LINE=(f"Rung 1 PASS: {npass}/{ntot} pre-registered checks met on the finest meshes (tension peak {g('t_l015_h16','peak_F'):.3f} kN at l=0.015 and {g('t_l0075_h8','peak_F'):.3f} kN at l=0.0075 vs published means 0.734 / 0.720; shear peak {g('s_l0075_h8','peak_F'):.3f} kN vs 0.598; crack paths as published), with the caveat that the mesh- and step-converged shear answer sits 7–10 % below the published values, which were computed on coarse meshes with the same lagging scheme. "
 f"Rung 2: SFC3 data unreachable (Globus login; summary paper firewalled), so the named fallback, the Damage Mechanics Challenge, was predicted blind: peak load {q1['peak_load_N']:.0f} N (1σ ±{q1['peak_load_1sigma_N']:.0f} N) at {q1['displacement_at_peak_mm']:.2f} mm, abrupt failure, crack from the notch twisting into a cross-beam plane — frozen in results/dmc_prediction.json, awaiting grading.")
RESULT_BODY=f"""Numbers: `metrics.json`, `results/rung1_grade.json`, `results/pf_run_metrics.json`, `results/dmc_prediction.json`.
Figures: `results/fig_pf_*.png`, `results/fig_dmc_*.png`.

### Rung 1 — phase-field benchmarks: {npass}/{ntot} pre-registered checks pass

{table}
Reference = mean of the matched-l published values listed in `PREREGISTRATION.md`. Graded runs are the finest mesh
of each series (h = l/16 for tension l=0.015, h = l/8 otherwise). S3/S4 come from the finest shear run that was
carried to the end of the load programme (h = l/4); the h = l/8 run stops at u = 0.0125 mm and gives 69.2° there.

**Mesh convergence (single pass, Miehe steps), peak load in kN** (`results/fig_pf_convergence.png`)

| h/l | tension l=0.015 | tension l=0.0075 | shear l=0.015 | shear l=0.0075 |
|---|---|---|---|---|
| 1/2 | {g('t_l015_h2','peak_F'):.4f} | {g('t_l0075_h2','peak_F'):.4f} | {g('s_l015_h2','peak_F'):.4f} | {g('s_l0075_h2','peak_F'):.4f} |
| 1/4 | {g('t_l015_h4','peak_F'):.4f} | {g('t_l0075_h4','peak_F'):.4f} | {g('s_l015_h4','peak_F'):.4f} | {g('s_l0075_h4','peak_F'):.4f} |
| 1/8 | {g('t_l015_h8','peak_F'):.4f} | {g('t_l0075_h8','peak_F'):.4f} | {g('s_l015_h8','peak_F'):.4f} | {g('s_l0075_h8','peak_F'):.4f} |
| 1/16 | {g('t_l015_h16','peak_F'):.4f} | not run (stopped for compute) | – | – |
| published | 0.70 / 0.701 / 0.80 | 0.68–0.74 | 0.54 / 0.531 | 0.595 / 0.60 |

The peak falls monotonically with refinement (≈ linear in h; the usual explanation is that coarse elements over-estimate the
fracture energy — quoted from memory, not used for grading). At h = l/2 — the resolution of the published shear runs (Zhou: h ≈ l/1.9) — this solver matches the
published shear peak to 0.3 % (0.5996 vs 0.598) and Miehe's digitized tension curve almost point for point at
h = l/4 (0.711 kN at 5.55e-3 mm vs 0.701 at 5.55e-3).

**Load-step and staggered-iteration study (l = 0.015, h = l/4)**

| Scheme | tension peak [kN] @ u [mm] | shear peak [kN] @ u [mm] |
|---|---|---|
| single pass, coarse step (1e-5 tension / 4e-5 shear) | {g('t_l015_h4_du1e-5','peak_F'):.4f} @ {g('t_l015_h4_du1e-5','u_peak'):.5f} | {g('s_l015_h4_du4e-5','peak_F'):.4f} @ {g('s_l015_h4_du4e-5','u_peak'):.5f} |
| single pass, Miehe step (1e-6 tension / 1e-5 shear) — graded | {g('t_l015_h4','peak_F'):.4f} @ {g('t_l015_h4','u_peak'):.5f} | {g('s_l015_h4','peak_F'):.4f} @ {g('s_l015_h4','u_peak'):.5f} |
| single pass, 4× finer (2.5e-7 / 2.5e-6) | {g('t_l015_h4_du2.5e-7','peak_F'):.4f} @ {g('t_l015_h4_du2.5e-7','u_peak'):.5f} | {g('s_l015_h4_du2.5e-6','peak_F'):.4f} @ {g('s_l015_h4_du2.5e-6','u_peak'):.5f} |
| staggered iterated to max\\|Δd\\| < 1e-5 | {g('t_l015_h4_stag_tol1e-5','peak_F'):.4f} @ {g('t_l015_h4_stag_tol1e-5','u_peak'):.5f} | SHEAR_STAG |

Tension is step-converged at Miehe's step (0.3 % above the converged value). **Shear is not**: the load still
drops 2.4 % when the step is cut four-fold, and because the shear curve has two humps of almost equal height the
"displacement at peak" jumps from the second hump (0.0098 mm) to the first (0.0086 mm). The published curves were
produced with the same single-pass scheme and carry the same lag.

**Traps (shear, l = 0.015, h = l/4; `results/fig_pf_traps.png`)**

| Variant | peak [kN] | what happens |
|---|---|---|
| reference set-up | {g('s_l015_h4','peak_F'):.3f} | single crack curving down to the bottom edge; overlays Miehe's digitized curve |
| side edges left free | {g('s_l015_h4_freesides','peak_F'):.3f} | elastic stiffness 46 instead of 63.5 kN/mm, peak 22 % low — the first thing I got wrong |
| no energy split | {g('s_l015_h4_nosplit','peak_F'):.3f} | the crack grows upward into the compressed half and damage spreads along the top edge |
| volumetric–deviatoric split | {g('s_l015_h4_amor','peak_F'):.3f} | the crack runs almost horizontally and the plate fails completely at u = 0.0109 mm |
| no irreversibility | {g('s_l015_h4_noirrev','peak_F'):.3f} | nearly unchanged (+0.4 %): loading is monotonic, so healing never gets a chance; the trap is real but this benchmark does not exercise it |

Other checks: the 3D GPU solver used for rung 2 reproduces the 2D tension benchmark (0.7087 kN at 5.58e-3 mm,
h = l/2, iterated; `verify3d_sent.py`). Crack surface energy at full tension failure is 1.20–1.35 × Gc × crack length
(h = l/4), i.e. the model dissipates more than Gc per unit crack, which I recall being a known property of this formulation (memory, not graded).

### Rung 2 — blind prediction (frozen, **not graded here**)

**SFC3 could not be attempted.** The data package needs an interactive Globus login; the only open copy of the
challenge description (MIT DSpace author manuscript of the summary paper) returned HTTP 429 from a web firewall on
four polite tries; every OSTI item is a post-deadline results document, which the blind protocol forbids opening.
Details: `raw/sfc3/SOURCE.md`. The fallback named in the brief was used: the **Damage Mechanics Challenge**
(3D-printed gypsum beams in three-point bending; calibration package downloaded from PURR without login).

Calibration (not blind; `results/fig_dmc_calibration.png`): one fracture energy (Gc = {P['model']['Gc_N_per_mm']} N/mm at
l = 1 mm, E = 1.4 GPa) fitted in the least-squares sense to the mean peak loads of the four calibration geometries:
{", ".join(f"{k} {v['model_N']:.0f} vs {v['measured_mean_N']:.0f} N ({v['error_pct']:+.1f} %)" for k,v in P['calibration_fit'].items())}.
The 7 % scatter of those four errors is the model-form uncertainty carried into the prediction.

Prediction for the challenge geometry (`results/dmc_prediction.json`, `results/dmc_prediction.md`,
`results/fig_dmc_prediction.png`):

| Quantity | Prediction | 1σ |
|---|---|---|
| Peak load | **{q1['peak_load_N']:.0f} N** | ±{q1['peak_load_1sigma_N']:.0f} N (80 % interval {q1['peak_load_80pct_interval_mean_of_specimens_N'][0]:.0f}–{q1['peak_load_80pct_interval_mean_of_specimens_N'][1]:.0f} N for the series mean, {q1['peak_load_80pct_interval_single_specimen_N'][0]:.0f}–{q1['peak_load_80pct_interval_single_specimen_N'][1]:.0f} N for one specimen) |
| Displacement at peak | {q1['displacement_at_peak_mm']:.2f} mm | ±{q1['displacement_at_peak_1sigma_mm']:.2f} mm (dominated by rod-contact compliance fitted to the calibration beams) |
| Loading stiffness, 30–80 % of peak | {q1['loading_stiffness_30_80pct_N_per_mm']:.0f} N/mm | – |
| Failure | abrupt load drop at the peak, low tail | – |
| Crack path | starts at the notch root on the deep (front) side; twists out of the 26.6° notch plane into a plane across the beam; CRACK_TOP | see caveat |

Sensitivity of the peak (each variant re-calibrated): {", ".join(f"{k}: {v:.0f} N" for k,v in P['sensitivity_peak_load_N'].items())}.
The point prediction is the mean of these variants.

Crack-path check on the calibration beams (`results/fig_dmc_crack_path_validation.png`): the lower half of the HB
and HA paths matches the laser scans to ≲0.5 mm, but the upper half is biased towards vertical — sideways shift at
22.5 mm height, HB: measured {path['HB']['measured_lateral_shift_at_22p5mm']:.1f} mm, model {pm(path['HB']['model'])}; HA: measured
{path['HA']['measured_lateral_shift_at_22p5mm']:.1f} mm, model {pm(path['HA']['model'])}. Refinement closes most of the gap for HB
(mesh bias) but not for HA (model form). The predicted top exit of the challenge crack therefore carries an
empirical correction of {q2['top_exit_bias_correction_mm_towards_load']:.1f} mm towards the loading rod.

Blindness: only the organisers' information packet and calibration files were read. The outcome paper (Morris et
al. 2024) and all participant papers were not searched for or opened. What I remembered beforehand is in `log.md`
(21:00 entry): that a phase-field participant paper exists and claims good agreement — no outcome numbers.
"""
json.dump(dict(RESULT_LINE=RESULT_LINE, RESULT_BODY=RESULT_BODY), open("report_src/readme_parts.json","w"))
print(RESULT_LINE)
