---
kind: experiment
slug: "evolutionary-dehomogenization"
date: "2026-11-16"
status: done
hypothesis: "The evolutionary de-homogenisation method of Xu, Guo, Kawabe & Yaji (Eng. Struct. 2026; arXiv 2510.08830) — homogenisation-based LF designs, phasor de-homogenisation, PCA, NSGA-II selection, VAE latent crossover and image-deformation mutation, scored on HF geometry — reproduces on a new geometry and boundary conditions (a simply supported girder with spring bearings, a distributed deck load and a lined duct), and its reported gains survive controls the paper did not run: a matched budget of extra LF designs, no selection, phase re-realisation, and an independent body-fitted CalculiX evaluator."
result: "Partly confirmed. The EA step replicates (HV +19.5 % stiffness over 100 generations, +14.9 % buckling over 30; paper +15.1 / +19.4 % at 300) and beats a matched budget of extra LF designs on every seed (stiffness +1.6 % for more-LF, buckling +10.1 %), driven by selection (no-selection +0–4 %). Mutation is inert (−1.7 pp, mutants ≤ 2 % of survivors) and the VAE hurts: crossover in PCA space gains 10 pp more because the VAE collapses population diversity. Evolved buckling designs lose to plain SIMP at 3 of 4 volumes in both objectives (confirmed in CalculiX). Gains survive phase re-realisation (86 % / 68 % retained) but the search partly selects lucky realisations, and it kept one pixel-evaluator artefact (CalculiX compliance 2.6× the pixel value); rank agreement with CalculiX holds (ρ 0.99 / 0.90), ratio spread misses (13 % / 26 %)."
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

| Hypothesis | Verdict | Evidence (`results/grades.json`, `results/grade_ccx.json`) |
|---|---|---|
| H1 replication | PASS | S +24.8 / +20.3 / +13.5 %; B +12.7 / +13.7 / +18.4 % |
| H2a EA vs more LF | EA > more LF (S, B) | final HV S 1.48/1.43/1.35 vs 1.20; B 0.670/0.676/0.704 vs 0.654 |
| H2b selection | PASS | NOSEL S +3.0/+0.1/+3.6 %, B +6.0 % |
| H3 mutation | MISS | FULL − NOMUT = −1.7 pp |
| H4 VAE | MISS | FULL − NOVAE = −10.2 pp (NOVAE +28.2/+31.9/+29.2 %) |
| H5 B vs SIMP | MISS | 1 of 4 volumes; compliance 17–42 % worse |
| H6 re-realisation | PASS | S 86 %, B 68 % retained (B: constraint dropped, post-hoc) |
| H7 CalculiX | MISS | Spearman 0.99 / 0.90 pass, dominance agrees; ratio CV 12.7 % / 25.7 % |

## Interpretation

- The evolutionary step works on a new problem, and it is real search, not best-of-N: a matched
  budget of extra LF designs barely moves the stiffness front (+1.6 %). The LF optimiser never sees
  de-homogenisation; the EA scores the de-homogenised result. The gallery shows the same tied-A-frame
  macro-structure before and after evolution, with fewer fragments and dangling bars: the EA mainly
  finds fields that de-homogenise well (the review predicted this).
- Of the paper's three operators, only selection + recombination matter. The VAE is a diversity
  attractor (population spread 31 → 18–25 with it, → 46–57 without), the same failure the hub records
  for learned mutation operators (`gurkan2026mutation`). Mutation survives PCA re-projection at ~62 %
  and almost never survives selection.
- For buckling, lattice infill did not beat a solid SIMP design on this girder; the paper's buckling
  advantage did not transfer (30 generations, stiffness-constrained).
- Two evaluator-side lessons: pixel-grid linear buckling is not mesh-objective (corner singularities)
  without stress regularisation, and an optimiser on a cheap evaluator finds its artefacts (one kept
  design is 2.6× softer in body-fitted CalculiX). The cheap evaluator still ranks correctly.

## Diagnostics

- intended_effect_confirmed: partial — H1/H2 pass, H3/H4/H5/H7 miss (`results/grades.json`)
- leakage_check: evaluator hashes logged before the first arm (`log.md` 15:20 UTC); CalculiX never in the loop (`run_arm.py` never imports `verify_ccx`) — no evaluator drift
- overfitting_signal: winner's curse — evolved fronts lose ~25 % (S) / ~33 % (B BLF) per design under phase re-draw vs ~10 % / ~18 % for gen 0 (`results/rereal/`); one evaluator artefact kept (`results/ccx_grade/buck_FULL_s1_0.35.json`)
- delta_from_prior: vs trial 3 (gradient TO), this is the first gradient-free design loop; SIMP still wins on buckling (`results/simp/simp_ccx.json`)
- unexpected_findings: the VAE lowers performance by collapsing diversity; fixed supports give a buckling mode whose load → 0 under refinement (`log.md` 04:20–04:50)
- next_candidates:
  - Replace the VAE with a diversity-preserving recombination (PCA-space BLX + a novelty archive) and run 300 generations on both objectives to see whether the gap to SIMP closes.
  - Put a body-fitted CalculiX check on the final front inside the loop (or an adversarial archive of re-realisations, per `zhu2026bad`) so selection cannot keep pixel artefacts or lucky phase draws.
  - Add branch closure to the phasor port and measure how much of the EA gain it absorbs (if the EA mainly repairs bad de-homogenisation, closure should shrink it).
