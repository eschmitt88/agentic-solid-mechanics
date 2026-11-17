---
kind: experiment
slug: "evolutionary-dehomogenization"
date: "2026-11-16"
status: running
hypothesis: "The evolutionary de-homogenisation method of Xu, Guo, Kawabe & Yaji (Eng. Struct. 2026; arXiv 2510.08830) — homogenisation-based LF designs, phasor de-homogenisation, PCA, NSGA-II selection, VAE latent crossover and image-deformation mutation, scored on HF geometry — reproduces on a new geometry and boundary conditions (a simply supported girder with spring bearings, a distributed deck load and a lined duct), and its reported gains survive controls the paper did not run: a matched budget of extra LF designs, no selection, phase re-realisation, and an independent body-fitted CalculiX evaluator."
result: "PENDING H2a"
related_concepts: ["agentic-design-optimization", "physics-grounded-evaluation", "differentiable-inverse-design"]
related_literature: ["xu2025evolutionary", "woldseth2024dehomtop808"]
tags: ["topology-optimization", "dehomogenization", "phasor", "multifidelity", "mftd", "nsga-ii", "vae", "lattice", "buckling", "calculix", "trial-13"]
---

# evolutionary-dehomogenization

Trial 13. Reproduction of Xu et al. 2026 on a new problem, with pre-registered controls.
Pre-registration: `PREREGISTRATION.md` (frozen 15:20 UTC before any arm ran; Addendum A at
17:00 UTC cut B arms to 30 generations for budget). Running log with every decision and
deviation: `log.md`. Design draft and technical review: `DESIGN.md`, `INTERFACES.md`.
Grades: `results/grades.json`, `results/grade_ccx.json`. Figures: `results/fig_*.png`.

## Setup

- **Problem (new):** 100 × 50 mm girder, plane stress, E = 1, ν = 0.3; spring bearings under 5 mm
  pads (pin left, roller right); uniform load on a 2.5 mm deck; Ø12 mm duct with a 2.5 mm liner at
  quarter span (`hf.GIRDER`). The paper used a double-clamped beam, an L-bracket and a multi-load part.
- **LF:** homogenisation-based TO of a cross-shaped cell (μ1, μ2, θ, ρ) on 40 × 20 elements
  (`homog.py`, `lf.py`); 480 gen-0 designs over V0 × Lmin as in the paper.
- **De-homogenisation:** phasor method ported from deHomTop808 (`dehom.py`, 0.11 s per design).
- **HF (in loop):** 400 × 200 pixel Q4 plane stress, compliance and linear buckling (`hf.py`).
- **EA:** per-channel PCA (506 dims), NSGA-II (100 elites), FC VAE (latent 8) with BLX latent
  crossover, Keys-bicubic deformation mutation (`ea.py`, `run_arm.py`), 110 HF evaluations per generation.
- **Grading oracle (never in loop):** body-fitted CalculiX on the marching-squares contour
  (`verify_ccx.py`, `grade_ccx.py`).

Main deviations from the paper (all in `log.md`): pixel HF instead of body-fitted COMSOL in the loop
(cost); geometric stiffness from stress averaged over r = 1 mm (pixel buckling is otherwise not
mesh-objective: corner singularities); spring bearings (fixed supports give a corner mode whose
BLF → 0 under refinement); swap-invariant 5-channel orientation encoding instead of raw θ; spline
instead of polynomial C^H fit (C^H jumps at w = 1); robust projection for the void indicator;
100 / 30 generations instead of 300.

## Result

See `results/grades.json` and the review page `docs/qa/trial-13-evolutionary-dehomogenization/`.

## Interpretation

PENDING

## Diagnostics

PENDING
