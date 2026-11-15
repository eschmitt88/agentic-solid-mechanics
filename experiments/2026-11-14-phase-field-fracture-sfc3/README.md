---
kind: experiment
slug: "phase-field-fracture-sfc3"
date: "2026-11-14"
status: done
hypothesis: "An LLM agent can (1) write a phase-field brittle-fracture solver from scratch that reproduces the Miehe 2010 single-edge-notched tension and shear benchmarks (peak load, displacement at peak, crack path) within ±10 % of published values, with mesh and load-step convergence shown, and (2) use it to make a defensible blind prediction for a real fracture challenge from the calibration data alone."
result: "Rung 1 PASS: 12/12 pre-registered checks met on the finest meshes (tension peak 0.696 kN at l=0.015 and 0.729 kN at l=0.0075 vs published means 0.734 / 0.720; shear peak 0.555 kN vs 0.598; crack paths as published), with the caveat that the mesh- and step-converged shear answer sits 7–10 % below the published values, which were computed on coarse meshes with the same lagging scheme. Rung 2: SFC3 data unreachable (Globus login; summary paper firewalled), so the named fallback, the Damage Mechanics Challenge, was predicted blind: peak load 298 N (1σ ±24 N) at 0.41 mm, abrupt failure, crack from the notch twisting into a cross-beam plane — frozen in results/dmc_prediction.json, awaiting grading. GRADED 2026-11-15 (results/dmc_grading.json, figure read-offs from Morris et al. 2024): blind DMC peak load 298 N vs measured mean ~327 N (about 9 % low, below all four expert teams at ~320-335 N, just inside the stated 80 % interval); displacement at peak, abrupt failure mode and crack path (within ~2 mm) correct."
related_concepts: ["physics-grounded-evaluation"]
related_literature: []
tags: ["fracture", "phase-field", "jax", "gpu", "benchmark", "blind-prediction", "damage-mechanics-challenge", "sfc3-unreachable", "trial-8"]
---

# phase-field-fracture-sfc3

Track B of the phase-2 shortlist (`docs/2026-11-14-hard-problem-shortlist.md`): fracture, ending in a blind
prediction. Review page: `docs/qa/trial-8-fracture/index.html`. Chronological record, including dead ends: `log.md`.

## Hypothesis

1. **Rung 1.** A hand-written phase-field solver reproduces the single-edge-notched tension and shear tests of
   Miehe, Hofacker & Welschinger (2010): peak load and displacement at peak within ±10 % of the mean of published
   values opened this session, the published crack paths, and mesh convergence ≤3 % between the two finest meshes.
2. **Rung 2.** The same method, calibrated only on a challenge's calibration data, gives a blind prediction of
   load–displacement and crack path with a stated uncertainty. (Graded later by the orchestrator, not here.)

## Setup

- Config: `config.yaml`. Pass criteria: `PREREGISTRATION.md` (written 20:42 UTC, before the first solver run).
- **Rung 1 code:** `pf.py` (solver), `batch_tension.sh`, `batch_shear.sh`, `batch_shear_b.sh`, `run_s_l0075_h8.sh`
  (run matrix), `analyze_pf.py`, `rung1_grade.py`, `figs_pf.py`.
  AT2 phase field, spectral (Miehe) tension/compression split, history-field irreversibility, plane strain, bilinear
  quads on graded tensor-product meshes, geometric slit. Residual and tangent by JAX autodiff on the GPU, PARDISO
  sparse solves. Graded configuration: single-pass staggered scheme, Miehe's load steps, finest mesh.
- **Rung 1 references** (all opened this session; PDFs in `refs_pdf/`, not for git): Molnár & Gravouil 2017
  (HAL hal-02132509), Zhou, Rabczuk & Zhuang (arXiv 1902.05922), Zhang et al. (arXiv 1706.05449, 1805.07357),
  Miehe's curves as digitized in github.com/CastillonMiguel/phasefieldx. Miehe 2010 and Ambati 2015 are paywalled
  and were **not** opened; nothing is graded against memory.
- **Rung 2 code:** `pf3d.py` (3D GPU solver, hybrid AT2, Jacobi-PCG), `verify3d_sent.py`, `dmc.py`, `dmc_run3d.py`,
  `dmc_calib2d.py`, `dmc_data.py`, `dmc_contact.py`, `dmc_post.py`, `dmc_predict.py`, `fig_dmc_geometry.py`.
- **Rung 2 data:** `raw/dmc/` (Damage Mechanics Challenge calibration package, PURR doi:10.4231/QF39-Q924, CC BY 4.0).
  `raw/sfc3/SOURCE.md` records why the Third Sandia Fracture Challenge package could not be obtained.
- Extra package: `pypardiso` (via `uv pip install`).
- Reproduce: rung 1 `./batch_tension.sh; ./batch_shear.sh; ./batch_shear_b.sh; ./run_s_l0075_h8.sh; ./batch_shear_c.sh`
  (GPU + ≤6 threads each, ≈4 h in total on a shared box); rung 2 unzip `raw/dmc/LD.zip` and `FractureSurface.zip`
  somewhere and point `DMC_LD_DIR` / `DMC_SF_DIR` at them (default `/tmp/dmc_work/...`), then `dmc_calib2d.py`,
  `dmc_run3d.py` (arguments in `dmc_queue*.sh`), `dmc_data.py`, `dmc_post.py`, `dmc_contact.py`, `dmc_predict.py`;
  finally `analyze_pf.py`, `rung1_grade.py`, `figs_pf.py`, `finalize.py` (writes this README from `report_src/`).
- Not in git: `results/runs/` (69 MB of field snapshots), `results/dmc/*.npz`, `refs_pdf/` (36 MB of papers).

## Result

Numbers: `metrics.json`, `results/rung1_grade.json`, `results/pf_run_metrics.json`, `results/dmc_prediction.json`.
Figures: `results/fig_pf_*.png`, `results/fig_dmc_*.png`.

### Rung 1 — phase-field benchmarks: 12/12 pre-registered checks pass

| # | Quantity | This work | Reference | Tolerance | Result | Note |
|---|---|---|---|---|---|---|
| T1 | Tension l=0.015 peak load | 0.696 kN | 0.734 | ±10 % | PASS | -5.2 % (t_l015_h16) |
| T2 | Tension l=0.015 displacement at peak | 0.005407 mm | 0.00565 | ±10 % | PASS | -4.3 % (t_l015_h16) |
| T3 | Tension l=0.0075 peak load | 0.7288 kN | 0.72 | ±10 % | PASS | +1.2 % (t_l0075_h8) |
| T4 | Tension l=0.0075 displacement at peak | 0.005419 mm | 0.00528 | ±10 % | PASS | +2.6 % (t_l0075_h8) |
| S1 | Shear l=0.0075 peak load | 0.555 kN | 0.598 | ±10 % | PASS | -7.2 % (s_l0075_h8) |
| S2 | Shear l=0.0075 displacement at peak | 0.01012 mm | 0.0109 | ±10 % | PASS | -7.2 % (s_l0075_h8) |
| T5 | Tension crack path: max deviation from y=0.5 | 0.00094 mm | 0.0 | ≤ l = 0.0075 mm, reaches right edge | PASS | t_l0075_h8 |
| S3 | Shear initial crack angle (fit 0.02–0.10 mm from tip) | 68.6 deg | 61.0 | ±10° | PASS | s_l0075_h4; other fit windows: 67.0°, 68.3° |
| S4 | Shear crack meets bottom edge at x | 0.839 mm | 0.85 | ±0.07 mm, single crack | PASS | s_l0075_h4 |
| C1 | Mesh convergence t_l015: change of peak load between t_l015_h8 and t_l015_h16 | 0.65 % | 0.0 | ≤ 3 % | PASS |  |
| C1 | Mesh convergence t_l0075: change of peak load between t_l0075_h4 and t_l0075_h8 | 1.72 % | 0.0 | ≤ 3 % | PASS |  |
| C1 | Mesh convergence s_l0075: change of peak load between s_l0075_h4 and s_l0075_h8 | 2.35 % | 0.0 | ≤ 3 % | PASS |  |

Reference = mean of the matched-l published values listed in `PREREGISTRATION.md`. Graded runs are the finest mesh
of each series (h = l/16 for tension l=0.015, h = l/8 otherwise). S3/S4 come from the finest shear run that was
carried to the end of the load programme (h = l/4); the h = l/8 run stops at u = 0.0125 mm and gives 69.2° there.

**Mesh convergence (single pass, Miehe steps), peak load in kN** (`results/fig_pf_convergence.png`)

| h/l | tension l=0.015 | tension l=0.0075 | shear l=0.015 | shear l=0.0075 |
|---|---|---|---|---|
| 1/2 | 0.7375 | 0.7694 | 0.5682 | 0.5996 |
| 1/4 | 0.7108 | 0.7413 | 0.5368 | 0.5680 |
| 1/8 | 0.7005 | 0.7288 | 0.5273 | 0.5550 |
| 1/16 | 0.6960 | not run (stopped for compute) | – | – |
| published | 0.70 / 0.701 / 0.80 | 0.68–0.74 | 0.54 / 0.531 | 0.595 / 0.60 |

The peak falls monotonically with refinement (≈ linear in h; the usual explanation is that coarse elements over-estimate the
fracture energy — quoted from memory, not used for grading). At h = l/2 — the resolution of the published shear runs (Zhou: h ≈ l/1.9) — this solver matches the
published shear peak to 0.3 % (0.5996 vs 0.598) and Miehe's digitized tension curve almost point for point at
h = l/4 (0.711 kN at 5.55e-3 mm vs 0.701 at 5.55e-3).

**Load-step and staggered-iteration study (l = 0.015, h = l/4)**

| Scheme | tension peak [kN] @ u [mm] | shear peak [kN] @ u [mm] |
|---|---|---|
| single pass, coarse step (1e-5 tension / 4e-5 shear) | 0.7222 @ 0.00569 | 0.5794 @ 0.01108 |
| single pass, Miehe step (1e-6 tension / 1e-5 shear) — graded | 0.7108 @ 0.00555 | 0.5368 @ 0.00984 |
| single pass, 4× finer (2.5e-7 / 2.5e-6) | 0.7095 @ 0.00553 | 0.5239 @ 0.00859 |
| staggered iterated to max\|Δd\| < 1e-5 | 0.7089 @ 0.00552 | 0.5205 @ 0.00850 (Δu = 1e-5, at most 100 passes per step; the cap is reached once the crack runs, so this is still not fully converged). Iterating to 1e-5 with Δu = 1e-4 stagnated — one step did not converge in 3000 passes — and was abandoned |

Tension is step-converged at Miehe's step (0.3 % above the converged value). **Shear is not**: the load still
drops 2.4 % when the step is cut four-fold, and because the shear curve has two humps of almost equal height the
"displacement at peak" jumps from the second hump (0.0098 mm) to the first (0.0086 mm). The published curves were
produced with the same single-pass scheme and carry the same lag.

**Traps (shear, l = 0.015, h = l/4; `results/fig_pf_traps.png`)**

| Variant | peak [kN] | what happens |
|---|---|---|
| reference set-up | 0.537 | single crack curving down to the bottom edge; overlays Miehe's digitized curve |
| side edges left free | 0.421 | elastic stiffness 46 instead of 63.5 kN/mm, peak 22 % low — the first thing I got wrong |
| no energy split | 0.488 | the crack grows upward into the compressed half and damage spreads along the top edge |
| volumetric–deviatoric split | 0.481 | the crack runs almost horizontally and the plate fails completely at u = 0.0109 mm |
| no irreversibility | 0.539 | nearly unchanged (+0.4 %): loading is monotonic, so healing never gets a chance; the trap is real but this benchmark does not exercise it |

Other checks: the 3D GPU solver used for rung 2 reproduces the 2D tension benchmark (0.7087 kN at 5.58e-3 mm,
h = l/2, iterated; `verify3d_sent.py`). Crack surface energy at full tension failure is 1.20–1.35 × Gc × crack length
(h = l/4), i.e. the model dissipates more than Gc per unit crack, which I recall being a known property of this formulation (memory, not graded).

### Rung 2 — blind prediction (frozen, **not graded here**)

**SFC3 could not be attempted.** The data package needs an interactive Globus login; the only open copy of the
challenge description (MIT DSpace author manuscript of the summary paper) returned HTTP 429 from a web firewall on
four polite tries; every OSTI item is a post-deadline results document, which the blind protocol forbids opening.
Details: `raw/sfc3/SOURCE.md`. The fallback named in the brief was used: the **Damage Mechanics Challenge**
(3D-printed gypsum beams in three-point bending; calibration package downloaded from PURR without login).

Calibration (not blind; `results/fig_dmc_calibration.png`): one fracture energy (Gc = 0.0514 N/mm at
l = 1 mm, E = 1.4 GPa) fitted in the least-squares sense to the mean peak loads of the four calibration geometries:
HC 212 vs 216 N (-2.0 %), HB 250 vs 243 N (+3.1 %), HA 411 vs 379 N (+8.4 %), H45 246 vs 270 N (-8.7 %).
The 7 % scatter of those four errors is the model-form uncertainty carried into the prediction.

Prediction for the challenge geometry (`results/dmc_prediction.json`, `results/dmc_prediction.md`,
`results/fig_dmc_prediction.png`):

| Quantity | Prediction | 1σ |
|---|---|---|
| Peak load | **298 N** | ±24 N (80 % interval 266–329 N for the series mean, 259–337 N for one specimen) |
| Displacement at peak | 0.41 mm | ±0.09 mm (dominated by rod-contact compliance fitted to the calibration beams) |
| Loading stiffness, 30–80 % of peak | 800 N/mm | – |
| Failure | abrupt load drop at the peak, low tail | – |
| Crack path | starts at the notch root on the deep (front) side; twists out of the 26.6° notch plane into a plane across the beam; it reaches the top face about 4 mm from the loading rod on the notch side (x ≈ -3.8 mm, plausible range -6.4 to -2.8 mm; raw model -6.1 mm plus a 2.3 mm bias correction taken from the calibration beams) | see caveat |

Sensitivity of the peak (each variant re-calibrated): base l=1.0 h=0.5: 295 N, finer mesh l=1.0 h=0.35: 303 N, smaller l=0.6 h=0.3: 295 N.
The point prediction is the mean of these variants.

Crack-path check on the calibration beams (`results/fig_dmc_crack_path_validation.png`): the lower half of the HB
and HA paths matches the laser scans to ≲0.5 mm, but the upper half is biased towards vertical — sideways shift at
22.5 mm height, HB: measured -5.3 mm, model l=1.0,h=0.5: -2.5 mm, l=1.0,h=0.25: -3.8 mm, l=1.0,h=0.125: -4.3 mm, l=0.5,h=0.125: -4.1 mm; HA: measured
-9.4 mm, model l=1.0,h=0.5: -7.5 mm, l=1.0,h=0.25: -7.5 mm, l=1.0,h=0.125: -7.7 mm, l=0.5,h=0.125: -7.7 mm. Refinement closes most of the gap for HB
(mesh bias) but not for HA (model form). The predicted top exit of the challenge crack therefore carries an
empirical correction of 2.3 mm towards the loading rod.

Blindness: only the organisers' information packet and calibration files were read. The outcome paper (Morris et
al. 2024) and all participant papers were not searched for or opened. What I remembered beforehand is in `log.md`
(21:00 entry): that a phase-field participant paper exists and claims good agreement — no outcome numbers.


## Interpretation

**Rung 1.** The solver reproduces the benchmark, and the exercise shows why "reproducing the benchmark" is a
soft target. Three things moved the answer by more than the spread I was graded against:

1. *Boundary conditions nobody prints.* The shear test only matches the published stiffness if the side edges are
   held vertically. None of the papers opened says so; I found it by comparing my elastic stiffness (46 kN/mm)
   with the digitized Miehe curve (63.5 kN/mm) and trying the constraint (63.5 kN/mm).
2. *Mesh.* Peak load falls roughly linearly with h/l. The published shear numbers are coarse-mesh results (Zhou: h ≈ l/2);
   refining to l/8 lowers the shear peak by 7 %. I pass the ±10 % tolerance on the fine mesh, but only because the tolerance is wide.
3. *The staggered scheme.* One pass per step lags the crack. Tension is insensitive once the step is 1e-6; shear is
   not converged at Miehe's step (−2.4 % load for a 4× smaller step, and the location of the maximum jumps). Iterating
   the shear problem to convergence with coarse steps stagnated outright (one step > 3000 passes).

So the honest statement is: within ±10 % of the literature on every pre-registered number, exactly on top of the
literature at the literature's own resolution, and about 5–10 % below it when the numerics are converged.

**Rung 2.** The blind prediction stands on a small, checkable chain: one fracture energy explains four calibration
geometries to within ±9 %; the same model in 3D gives the challenge peak. The weakest links are (a) the model is
7–9 % wrong in opposite directions for the far-offset notch (too strong) and the rotated notch (too weak), and the
challenge geometry combines both; (b) the predicted crack path above mid-height is biased towards vertical (mesh for HB, model form for HA);
(c) the displacement axis is mostly an empirical rod-contact law, not mechanics I computed. The peak-load interval
reflects (a); the crack-path caveat reflects (b).

What surprised me: how much of the "ductile" look of the gypsum curves is the fixture (the beam itself is 2–3×
stiffer than the measured record), and that a larger length scale *improved* the calibration fit for the wrong
reason (it moved the crack away from the notch) — only the crack-path photos exposed that.

## Diagnostics

- intended_effect_confirmed: partial — rung 1 yes (12/12 pre-registered checks, `results/rung1_grade.json`); rung 2
  delivered as a frozen prediction for the fallback challenge, not for SFC3, and is ungraded (`results/dmc_prediction.json`).
- leakage_check: rung 1 references were fixed in `PREREGISTRATION.md` before the first run and only sources opened
  this session are used; rung 2 — no outcome source opened, prior memory recorded in `log.md` (21:00). One SFC3
  outcome fragment was seen by accident (`log.md`, 20:35), irrelevant to the challenge actually predicted.
- overfitting_signal: calibration residuals HC −2 %, HB +3 %, HA +8 %, H45 −9 % with one free fracture parameter
  (`results/dmc/calibration_fit.json`) — under-fit rather than over-fit; no held-out calibration geometry was kept back.
- delta_from_prior: first fracture trial; reuses the JAX-autodiff FEM approach of trials 3–5 (`pf.py:115-166`).
- unexpected_findings: published benchmark values are not mesh-converged (shear −7 % at h=l/8,
  `results/pf_run_metrics.json:s_l0075_h8`); shear single-pass scheme not step-converged at the standard step
  (`results/pf_run_metrics.json:s_l015_h4_du2.5e-6`); structured-mesh bias of the 3D crack path
  (`results/dmc/crack_path_validation.json`).
- next_candidates:
  - Get the SFC3 package through a Globus login and run the originally intended ductile-failure prediction in CalculiX.
  - Re-run the challenge geometry on an unstructured or much finer mesh (h ≤ l/8) to remove the crack-path bias before grading the path.
  - Replace the single-pass scheme by a monolithic or over-relaxed staggered solve and repeat the shear benchmark to get a step-converged reference curve.

## Follow-up

- Grade `results/dmc_prediction.json` against Morris et al. 2024 (orchestrator; this agent has not seen the outcome).
- If graded poorly on the crack path, check first whether the finer-mesh path (`results/dmc/crack_path_validation.json`) would have been closer.
- Kalthoff–Winkler / asymmetric bending crack-path benchmarks were not attempted; the calibration-beam crack paths (measured, not blind) served as the crack-path test instead.
