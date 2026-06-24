---
kind: experiment
slug: "vhb-viscohyperelastic-calibration"
date: "2026-06-23"
status: done
hypothesis: "A finite-strain viscohyperelastic model (Ogden equilibrium network + a Bergström–Boyce viscous branch) implemented as a differentiable JAX material-point model can be calibrated to real cyclic VHB 4910 stress–stretch data by gradient descent, and the identified parameters generalise — predicting held-out experiments (an unseen strain rate, and amplitudes far beyond the calibration range)."
result: "Confirmed. The 7-parameter model calibrates to all 11 cyclic curves at R²=0.86, and generalises: trained on rates 0.01 & 0.05 s⁻¹ it predicts the held-out 0.03 s⁻¹ tests at R²=0.94; trained only on λ≤4 it predicts λ→6.25 well (R²≈0.82–0.98) and λ→8.9 reasonably (R²≈0.73–0.79), the degradation at λ→8.9 being honest extrapolation past the strong large-stretch stiffening upturn. Calibration runs in ~11 s on CPU via JAX autodiff (jax.scipy BFGS + Adam)."
related_concepts: ["differentiable-inverse-design", "physics-grounded-evaluation"]
related_literature: []
tags: ["jax", "inverse-problem", "material-calibration", "viscoelastic", "bergstrom-boyce", "ogden", "rubber", "trial-5"]
---

# vhb-viscohyperelastic-calibration

Trial 5 / a different use of the differentiable solver: **inverse parameter
identification** (material-model calibration) rather than design. Given real
experimental stress–stretch data, recover the constitutive parameters by
gradient descent on a differentiable simulation. Demonstrates that JAX's autodiff
is as good at *calibration / digital-twin* problems as at design optimisation.

## Hypothesis

A finite-strain **viscohyperelastic** model — an **Ogden** hyperelastic
equilibrium network in parallel with one **Bergström–Boyce** (BB) rate-dependent
viscous branch — written as a differentiable JAX material-point model, can be
calibrated to cyclic VHB 4910 data and **generalise** to held-out experiments
(an unseen strain rate; amplitudes beyond the calibration range).

## Setup

- **Material / data:** VHB 4910 acrylic elastomer, 11 uniaxial **cyclic
  loading–unloading** stress–stretch histories — 4 max-stretch amplitudes
  (λ≈2.25, 4.0, 6.25, 8.9) × 3 stretch rates (0.01, 0.03, 0.05 s⁻¹). Strongly
  rate-dependent with clear hysteresis. Source + license (CC-BY-4.0) and
  citations in `../../raw/data/vhb4910-hossain2012/SOURCE.md` (iCANN Zenodo
  10066805; data digitised from Hossain–Vu–Steinmann 2012).
- **Forward model (`model.py`):** single material point, incompressible uniaxial
  (no mesh). Ogden N=2 equilibrium + one BB viscous branch (multiplicative split
  λ = λ_e·λ_v; neo-Hookean overstress spring; BB power-law flow). The viscous
  internal variable is integrated with a differentiable semi-implicit `lax.scan`.
  7 free parameters. Stress reported as **nominal** (engineering) stress, per
  Hossain et al.
- **Inverse problem (`calibrate.py`):** minimise the normalised mean-square
  stress error over training curves by gradient descent (Adam → `jax.scipy`
  BFGS), `vmap`-ing the forward model over curves. Parameters kept positive via a
  softplus reparameterisation.
- No mesh, no DVC — the "data" is the calibration target itself.

## Result

Points at `metrics.json` and `results/calibration.json`; plots in `results/`.

- **Fit to all 11 curves:** mean **R² = 0.86** with 7 physically-meaningful
  parameters (`results/fit_all.png` — rate-dependence at one amplitude + the
  amplitude range at one rate).
- **Validation A — amplitude extrapolation** (`results/study_A.png`): trained on
  λ≤4 (all rates), **predicts** the large-stretch cyclic tests. λ→6.25:
  R²≈0.82–0.98; λ→8.9: R²≈0.73–0.79. The model reproduces the loop and the
  rate ordering far outside its calibration range; it under-predicts only the
  steep stiffening upturn near λ≈9 (honest limit of extrapolation).
- **Validation B — held-out rate** (`results/study_B.png`): trained on 0.01 &
  0.05 s⁻¹ (all amplitudes), **predicts** the unseen 0.03 s⁻¹ tests at mean
  **R² = 0.94** — the viscous branch interpolates strain rate correctly.

## Interpretation

This is the inverse of the design trials: instead of choosing a structure to
optimise an objective, we infer hidden material parameters that make the
simulation match data. The same JAX machinery (autodiff + a differentiable
time-integrated constitutive law) does it in seconds. The strong held-out-rate
prediction (R²=0.94) is the key evidence that the *rate-dependent physics* — not
just curve-fitting — was identified: the model was never shown 0.03 s⁻¹ and
predicts it. Amplitude extrapolation to ~1.5× works well; to ~2.2× (λ→8.9) it
degrades exactly where the material's large-stretch upturn outruns the
calibration range — the expected, honest failure mode.

## Diagnostics

Unless noted, numbers reference `metrics.json`.

- intended_effect_confirmed: yes — held-out-rate R²=0.94, amplitude-extrapolation
  R²=0.85 (`metrics.json`), model reproduces rate-dependent hysteresis loops.
- leakage_check: held-out curves were excluded from each study's training set
  (`calibrate.py` train/predict index split); predictions on them use train-only
  parameters. No agent involved.
- overfitting_signal: train vs held-out R²: study A 0.93 → 0.85, study B 0.91 →
  0.94 (held-out ≥ train → no overfit; the model generalises).
- delta_from_prior: first inverse-problem / calibration trial; reuses the
  differentiable-JAX approach of trials 3–4 for parameter ID instead of design.
- unexpected_findings: held-out *rate* R² (0.94) exceeded the training R² (0.91)
  — the middle rate is "easier" (bracketed by the trained rates), a clean
  interpolation.
- next_candidates:
  - Loop-2 agentic version: hand the data + model to the agent and have it write
    the calibration (an inverse-problem pass@k).
  - Add the skeletal-muscle compression data (same Zenodo archive) as a second
    material to show the method transfers to compression.

## Follow-up

- Compare model forms (Ogden N=1 vs N=2; linear Maxwell vs BB) on the same data.
- Quantify parameter uncertainty (the Hessian is available via JAX autodiff).
