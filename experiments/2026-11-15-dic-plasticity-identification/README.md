---
kind: experiment
slug: "dic-plasticity-identification"
date: "2026-11-15"
status: done
hypothesis: "An elastoplastic model of 304L sheet calibrated by finite-element model updating against real full-field DIC strain maps plus load on two specimen geometries predicts load and strain fields of geometries it never saw within tolerances fixed in advance, and does so better than the same model calibrated from a plain tension curve."
result: "Real data (Sandia 304L, Figshare+, CC0). First half held: the frozen model (Hill-48 + Voce + linear, 7 parameters, chosen on a development plate) met all 12 pre-registered checks on the two held-out geometries (force curve error 0.7 % and 1.5 % vs 5 % allowed; strain-field error 3.8–7.2 % vs 15 %; hot spot within 1 %), and all 6 on a held-out rolling-direction test; mesh/step refinement moves no score by more than 0.09 points and a CalculiX re-run of the isotropic form agrees to 0.24 % in force. Second half did not hold: the uniaxial-only baseline also met all 18 checks and predicted force slightly better (0.3–0.5 %); full-field data improved only the strain fields (6.6 → 3.9 %, 8.4 → 6.5 %) by identifying mild anisotropy, and that anisotropy has the wrong sign for the rolling direction. Tolerances proved generous; the exam did not separate the two calibration routes."
related_concepts: ["differentiable-inverse-design", "implicit-differentiation-through-solve", "physics-grounded-evaluation", "agent-as-solver-operator"]
related_literature: []
tags: ["jax", "plasticity", "femu", "dic", "material-identification", "held-out", "hill48", "calculix", "identifiability", "trial-11", "phase-2-track-E", "real-data"]
---

# dic-plasticity-identification

Phase 2, track E: identify an elastoplastic model from real digital-image-correlation
(DIC) strain maps on some specimen geometries and predict geometries the
calibration never saw. Review page: `docs/qa/trial-11-identification/index.html`.
Chronological record with every dead end: `log.md`. Sequel to
`experiments/2026-06-23-vhb-viscohyperelastic-calibration/`.

## Hypothesis

A finite-strain elastoplastic model calibrated by finite-element model updating
(FEMU) against full-field DIC plus load on two plate geometries (1) predicts the
load curve and strain fields of held-out geometries within tolerances fixed
before fitting, and (2) does so better than the same model calibrated from the
plain tension curve alone.

## Pass criteria (fixed before fitting)

Written 16:39 UTC before any model was fitted; verbatim in `PREREGISTRATION.md`
(one amendment at 16:42, before any fit: the force-drop rule was written with
the window on the wrong side). Per held-out specimen, primary model:

- force-curve NRMSE over stations 5…40 ≤ 5 %; force at the last station within 5 %;
- relative L2 error of the levelled Green–Lagrange strain field ≤ 15 % at 30, 60
  and 90 % of the scored range; top-1 % major strain (hot spot) within 15 %;
- "generalises" only if both primary held-out specimens pass all of these;
- gates: synthetic warm-up (σ_y within 2 %, Hill a, h within 0.05, c within 0.3;
  gradients vs FD ≤ 1e-5), mesh and step convergence reported, CalculiX
  re-run within 2 % (force) and 3 % (strain).

## Setup

- Config: `config.yaml`. Extra package: `h5py` (`uv pip install h5py`).
- **Data** (`raw/sandia-304l-dic/SOURCE.md`): Jones et al., Scientific Data
  11:1101 (2024), doi:10.1038/s41597-024-03949-y; files from Figshare+
  doi:10.25452/figshare.plus.25483534, CC0, no login. 1.5 mm 304L sheet,
  stereo DIC, load cell. Only the slow tests (5·10⁻³ mm/s, ≈ 23 °C throughout)
  are used, so a rate-independent isothermal model is defensible. 26 GB
  downloaded (six processed `*-Data.mat` files), not in git.
- **Split**: calibration O1-3, O14-3 · development O5-4 · uniaxial baseline
  data BT3, CT3, CT4 · held out O15-3, XT10 (primary) and XR5 (rolling
  direction). All but XR5 are transverse to rolling.
- **Forward model** (`fem.py`): JAX float64, T6 plane stress, logarithmic-strain
  finite plasticity, quadratic yield φ² = a T₁₁² − 2h T₁₁T₂₂ + T₂₂² + c T₁₂²
  (von Mises = 1, ½, 3), σ_y = s0 + Q(1 − e^{−bα}) + Hα (+ second Voce term),
  scalar return map, pypardiso. E = 193 GPa, ν = 0.29 fixed (handbook values
  from memory, not graded; the dog bones give 187–194 GPa).
- **Boundary conditions and levelling** (`data.py`, `specimen.py`): the model is
  the nominal outline cut 3 mm inside the DIC cloud at top and bottom; measured
  DIC displacements on the two cut lines (cubic in x per station) drive it, for
  held-out specimens too. Strains are computed from displacements with one
  local-affine operator (1 mm radius) applied identically to DIC and to the FE
  field interpolated at the DIC points.
- **Identification** (`femu.py`, `fit_real.py`): Levenberg–Marquardt on force +
  strain residuals of both calibration plates, exact forward sensitivities
  propagated through the plastic history (`jax.jvp`, one linear solve per
  parameter per step with the factorised tangent), steps taken in
  log σ_y(fixed plastic strains) and mapped back. Forms M1 (3 parameters),
  M2 (4), M3 (6), M4 = Hill + M2 hardening (7), M4b = Hill + M3 hardening (9).
- **Baseline** (`baseline.py`): von Mises + the same hardening law fitted to
  the mean true-stress curve of the three slow transverse dog bones.
- **Scoring** (`score.py`, `freeze.py`): one script scores development and
  held-out sets; `score.py heldout` refuses to run without
  `results/frozen_models.json` and logs when it opened the files.
- **Verifier** (`ccx_check.py`): CalculiX 2.23, CPS6, NLGEOM, tabulated
  hardening, same mesh and boundary displacements, 40 steps.

## Result

All numbers in `metrics.json`; per-run records in `results/*.json`.

### Warm-up with known truth (`warmup.py`, `results/warmup.json`)

Synthetic field on the O1 outline (30 055-node mesh, Hill + Voce + linear),
3.6 µm displacement and 2 N force noise, boundary fits redone from the noisy
field, identified on the 8 880-node mesh.

| check | result | tolerance | |
|---|---|---|---|
| σ_y(α) on [0, α₉₅ = 0.37], start A | 0.15 % | 2 % | pass |
| Hill a / h / c, start A | +0.005 / +0.0014 / −0.0006 | 0.05 / 0.05 / 0.3 | pass |
| start B (other side of the truth) | same optimum to 8 digits | — | pass |
| sensitivities vs central FD, 9 parameters | ≤ 1.1e-6 | 1e-5 | pass |

Individual constants Q, b, H came back 5 %, 3 %, 7 % off while the curve was
right. The first two warm-up runs were stopped: the optimiser crawled along
that valley (`log.md`).

### Identified models (`results/fits/*.json`, `results/frozen_models.json`)

| form | parameters | J (calibration) | O5-4 force NRMSE | O5-4 strain error | score S |
|---|---|---|---|---|---|
| M1 von Mises + Voce | s0 331, Q 1063, b 2.44 | 1.987 | 0.67 % | 8.3 % | 0.687 |
| M2 + linear | s0 323, Q 431, b 5.17, H 701 | 1.903 | 0.56 % | 7.7 % | 0.628 |
| M3 double Voce | (six) | 1.894 | 0.53 % | 7.8 % | 0.624 |
| **M4 Hill + M2 law** | s0 330.6, Q 379.3, b 5.147, H 842.2, a 1.0735, h 0.4460, c 2.928 | 1.310 | 0.49 % | 4.7 % | **0.411** |
| M4b Hill + M3 law | (nine) | 1.299 | 0.46 % | 4.7 % | 0.407 |
| uniaxial-M2 (baseline) | s0 332, Q 370, b 5.31, H 880 | — | 0.96 % | 7.7 % | 0.709 |

Selection rule (fewest parameters within 10 % of the best S on O5-4) → **M4**.
Multi-start: M1 (2 starts), M2 (3), M4 (3), M4b (2) each end at one optimum to
four digits; M3 ends at three different points (J 1.894, 1.910, 4.339).

Identifiability of M4 (`metrics.json:identifiability_primary`; change that
raises J by 1 % with the others re-optimised): s0 ±1.5 %, a ±5.5 %, h ±2.6 %,
c ±1.5 %, but Q ±28 %, b ±28 %, H ±20 %; expressed as curve values, σ_y(0)
±1.5 %, σ_y(0.05) ±0.4 %, σ_y(0.2) ±0.4 %, σ_y(0.6) ±1.4 %. The data fix the
hardening curve and the anisotropy coefficients, not the Voce/linear constants.
95 % of the calibration material stays below α = 0.28; the maximum is 0.9
(`results/alpha_seen.json`).

### Held-out predictions (`results/heldout_scores.json`, opened 19:06:13 UTC)

| specimen | model | force NRMSE (≤ 5 %) | force, last station (±5 %) | strain error 30 / 60 / 90 % (≤ 15 %) | hot spot (±15 %) | |
|---|---|---|---|---|---|---|
| O15-3 | **M4** | 0.74 % | −0.96 % | 4.0 / 3.8 / 3.8 % | +0.8 % | 6/6 pass |
| O15-3 | uniaxial-M2 | 0.53 % | +0.39 % | 6.7 / 6.5 / 6.6 % | −3.0 % | 6/6 pass |
| O15-3 | null (elastic) | — | — | 39 / 32 / 31 % | −47 % | |
| XT10 | **M4** | 1.53 % | −1.53 % | 7.2 / 6.2 / 5.9 % | −1.0 % | 6/6 pass |
| XT10 | uniaxial-M2 | 0.32 % | −0.05 % | 9.6 / 8.1 / 7.6 % | −2.4 % | 6/6 pass |
| XT10 | null (elastic) | — | — | 38 / 31 / 31 % | −31 % | |
| XR5 (rolling) | **M4** | 2.64 % | −3.26 % | 7.2 / 7.6 / 8.6 % | +3.8 % | 6/6 pass |
| XR5 (rolling) | uniaxial-M2 | 1.27 % | −1.81 % | 8.7 / 8.3 / 9.2 % | +2.4 % | 6/6 pass |

M1, M2, M3, M4b and the other baselines also pass every check
(`results/heldout_scores.log`). Pre-registered verdict: **generalises** (both
primary specimens 6/6); orientation transfer: pass.

### Verification gates

- **Step convergence** (80 vs 40 increments) and **mesh convergence** (26–37 k
  nodes vs 7–10 k): largest change of any held-out score 0.09 percentage points
  (`results/convergence_heldout.json`). No pass depends on either.
- **CalculiX** (M2 on held-out O15-3, `results/ccx_check_O15-3_M2.json`): force
  0.24 % NRMSE vs JAX (limit 2 %), strain field 1.18 % at station 40 (limit
  3 %): pass. It covers the von Mises path only; the Hill path is checked
  against closed forms in `test_hill.py` (≤ 7e-10).
- **Elastic constants**: E 193 → 203 GPa or ν 0.29 → 0.33 moves O5-4 scores by
  ≤ 0.06 points (`results/elastic_sensitivity_O5-4.json`).

## Interpretation

- **It generalised, but the exam was easy for this material.** Every plastic
  model form passes every check with margins of 2–10×. At half the tolerances
  M4 would still pass both primary specimens; the baseline would miss XT10 on
  strain (9.6 %) and both would miss XR5.
- **Full-field data bought strain-field accuracy, not force accuracy.** The
  gain (6.6 → 3.9 % on O15-3, 8.4 → 6.5 % on XT10) comes entirely from the Hill
  coefficients: the von Mises forms contract the plate sideways ≈ 10 % too much
  (r = h/(1−h) = 0.80 < 1). On force the dog-bone baseline is as good or better
  on all three held-out tests. With anisotropy allowed, the FEMU hardening curve
  lands within 1–4 % of the dog-bone curve — the plates and the tension test
  tell the same hardening story.
- **Why the baseline survives extrapolation:** 304L at this rate hardens almost
  linearly beyond the uniform-elongation limit (α 0.47), and only a few per
  cent of the plate material goes past it.
- **One loading direction does not identify the other.** M4's a = 1.07 implies
  the rolling direction is 3.5 % weaker than transverse; the rolling dog bones
  (post hoc, not graded, `results/posthoc_rolling_dogbones.json`) show it is
  1–4 % stronger. This is why M4 is worse than the baseline on XR5 force.
- **A 4–7 % strain-field residual remains** for the best model and is spatially
  diffuse (not at edges). Not separated here: DIC subset smoothing (not
  emulated on the FE side), nominal vs as-cut outline, plane stress at necks,
  yield-surface shape beyond Hill-48.
- The identification is well-posed in the right coordinates: values of σ_y at
  fixed strains are pinned to ≤ 1.5 %, while Q, b, H trade off by 20–28 %.
  An optimiser working in the raw constants crawled; in curve coordinates it
  converged in 3–5 iterations.

## Deviations and things to hold against the result

- Amendment 1 to the pre-registration (force-drop window), before any fit.
- M4 was fitted with the M2 hardening law before the development scores that
  were supposed to choose between M2 and M3 existed; M3 is nominally better by
  0.8 %, so M4b was fitted afterwards and the selection rule applied to both.
- Identifiability is reported as 1 %-cost widths and curve-value widths, not
  the "linearised 95 % intervals" named in the pre-registration: residuals are
  dominated by model error, not noise, so a noise-based interval would be
  meaninglessly narrow.
- Held-out boundary displacements are a model input (pre-registered).
- Peak force and failure are not predicted (no damage model); scoring ends at
  90 % of the extension at maximum force, a window defined from the measured
  force curve.
- The post-hoc rolling dog-bone comparison and the "half tolerance" what-if
  were done after the held-out scores were known.
- O18-3 (15 GB) and all fast tests were not used.

## Diagnostics

- intended_effect_confirmed: partial — held-out generalisation confirmed (`metrics.json:heldout_verdict.generalises_preregistered` = true, 12/12 checks); superiority over the uniaxial baseline not confirmed for force (`metrics.json:heldout.XT10.models` force_nrmse 0.0153 vs 0.0032).
- leakage_check: held-out files opened once by `score.py:108-120` after `freeze.py` wrote `results/frozen_models.json`; timestamps and hashes in `results/heldout_access.json` — no fit or model choice after 19:06:13 UTC.
- overfitting_signal: calibration strain error 5.5–6.5 % / force 1.1 % (`metrics.json:calibration_scores`) vs held-out 3.8–7.2 % / 0.7–1.5 % (`metrics.json:heldout`) gap≈0 — no overfitting visible; the 7-parameter form generalises as well as it fits.
- delta_from_prior: vs 2026-06-23-vhb-viscohyperelastic-calibration, first trial with real full-field data, a held-out geometry split and exact history-dependent sensitivities (`fem.py:381-439`); not numerically comparable.
- unexpected_findings: the uniaxial baseline passes every held-out check (`results/heldout_scores.json`); the identified Hill coefficient a implies the wrong sign of rolling-direction strength (`results/posthoc_rolling_dogbones.json`).
- next_candidates:
  - Repeat on the fast tests of the same dataset with a rate- and temperature-dependent law, where a room-temperature tension curve cannot supply the answer.
  - Add one rolling-direction specimen (XR5 or a rolling dog bone) to the calibration set and test whether the Hill coefficient a flips sign and XT10/O15 scores hold.
  - Emulate the DIC subset (23 px Gaussian) on the FE field or reprocess the raw images to see how much of the 4–7 % residual is measurement smoothing.
  - Tighten the exam: predict the extension at maximum force with a damage or localisation criterion.

## Follow-up

- `_meta/index.md` was not touched (orchestrator's instruction: work only in
  this folder, `raw/sandia-304l-dic/` and the review page).
