---
kind: experiment
slug: "metamaterial-inverse-design"
date: "2026-11-14"
status: done
hypothesis: "A hand-written differentiable finite-strain solver, validated on the Qiu cosine bistable beam, can drive gradient-based inverse design of a curved-beam metamaterial cell to three prescribed stress–strain curves (stiffening, plateau, snap-through) and to a programmed bistable cell, and the designs survive a body-fitted re-solve in CalculiX within tolerances fixed in advance."
result: "Rung 1 passed on all three targets in CalculiX (curve NRMSE 0.25 % / 1.01 % / 2.00 % against tolerances 5 / 5 / 8 %; solver gap ≈ 0.3 %). Rung 2 (bistable: peak force, second position, return barrier) passed in CalculiX within 0.2 % on each, but only on the sixth optimisation attempt and with the coarsest verifier mesh aborting at a secondary buckling point. Rung 0 missed two of six pre-registered checks against Qiu's rounded closed-form constants (f_bot −10.8 %, d_bot −0.06 h) while matching Qiu's full series to 3 % and CalculiX to 0.12 %. Gradients are exact per state (≤ 1e-6 vs FD) but jump at bifurcations, are branch-dependent for symmetry-breaking parameters, and cannot see a snap moving through an integrated quantity — which is what stalled the bistability optimiser."
related_concepts: ["differentiable-inverse-design", "physics-grounded-evaluation"]
related_literature: []
tags: ["jax", "inverse-design", "metamaterial", "snap-through", "bistability", "implicit-differentiation", "calculix", "bifurcation", "trial-9", "phase-2-track-D"]
---

# metamaterial-inverse-design

Phase 2, track D: gradient-based inverse design of a nonlinear mechanical
metamaterial whose response passes through buckling and snap-through. Review
page: `docs/qa/trial-9-metamaterial/index.html`. Chronological record with
every dead end: `log.md`.

## Hypothesis

A differentiable large-deformation solver that has first been validated on a
beam with a closed-form answer can (1) shape a unit cell by gradient descent so
its uniaxial stress–strain curve matches three prescribed targets of rising
difficulty, and (2) program a bistable cell with a prescribed snap-through
force, second stable position and return barrier — and the resulting designs
hold up when re-meshed and re-solved in an independent solver.

## Pass criteria (fixed before optimisation)

Written at 20:37 UTC before any solver code existed; verbatim in
`PREREGISTRATION.md`. Not changed afterwards.

- **Rung 0** (cosine beam l = 100, t = 1, h = 6 mm): f_top, f_bot within 5 % of
  Qiu eq. (2.47); d_top, d_bot, d_end within 0.05 h of eq. (2.48); JAX curve
  within 2 % NRMSE of CalculiX.
- **Rung 1**, judged on the CalculiX curve only, NRMSE over 50 strains
  normalised by the target range: T1 ≤ 0.05 and monotonic; T2 ≤ 0.05 and plateau
  within 0.06 kPa of 0.40 for 0.12 ≤ ε ≤ 0.33; T3 ≤ 0.08 with a peak followed by
  a valley.
- **Rung 2**: F_pk = 0.040 N/mm, d_2 = 10.0 mm, E_back = 0.045 N·mm/mm, each
  within 5 % in CalculiX, both rest states shown stable.
- Verifier: CalculiX 2.23, NLGEOM, same neo-Hooke constants, whole beam, CPE6
  triangles with element size ≤ t_min/3.

## Setup

- Config: `config.yaml`.
- **Unit cell.** One elastomer beam (span 60 mm) between stiff walls, pushed at
  mid-span through a stiff 4 mm shuttle; cell 70 × 25 mm, plane strain. Stress =
  F/W, strain = d/H. Nine parameters: four centre-line cosine modes, five
  log-thickness modes (0.4–3 mm). The Qiu cosine beam is a member of the family.
- **Design solver** (`fem.py`): JAX float64, Q9 plane-strain elements on a
  body-fitted mapped mesh that is a smooth function of the parameters;
  CalculiX-form compressible neo-Hooke (C10 = 0.5 MPa, D1 = 0.4286 /MPa);
  displacement-controlled continuation with an energy-minimising Newton,
  Cholesky stability check at every step, lowest-mode kick off unstable states,
  saddle-free step when the tangent is indefinite, adaptive step bisection.
  Sensitivities dF_k/dp by implicit differentiation at each converged state.
  CPU only (790 dof; 9–20 s per curve). The GPU was not needed.
- **Optimiser** (`design.py`, `bistable.py`): scipy trust-region least squares
  with the exact Jacobian; four generic starts per target.
- **Verifier** (`ccx.py`, `verify.py`, `verify_bistable.py`): Gmsh body-fitted
  CPE6 mesh of both half-beams, CalculiX NLGEOM static, 100 increments.
- **Reference** (opened this session): J. Qiu, PhD thesis, MIT 2003, ch. 2, eqs
  2.9–2.48, https://pergatory.mit.edu/research/jqiu/thesisscreen.pdf — the
  source of Qiu, Lang & Slocum, JMEMS 13(2) 2004, doi:10.1109/JMEMS.2004.825308
  (`qiu.py`). The scispace copy of the paper itself returned no text.
- No dataset, no DVC stage, no held-out split: the grade is the independent
  solver, not a validation metric.

## Result

All numbers are in `metrics.json`; curves in `results/verify_*.json`.

### Rung 0 — forward solver (`validate_qiu.py`, `results/rung0_qiu_validation.png`)

| check | measured | tolerance | |
|---|---|---|---|
| f_top vs 740 EIh/l³ | −5.0 % | ±5 % | pass |
| f_bot vs 370 EIh/l³ | **−10.8 %** | ±5 % | **miss** |
| d_top vs 0.16 h | +0.042 h | ±0.05 h | pass |
| d_bot vs 1.92 h | **−0.060 h** | ±0.05 h | **miss** |
| d_end vs 1.99 h | −0.033 h | ±0.05 h | pass |
| JAX vs CalculiX, whole curve | 0.12 % | ≤ 2 % | pass |

The two misses stand. Context, not excuse: 740/370 are the Q ≥ 6 asymptote;
Qiu's modal series evaluated at the tested Q = 6 gives 714.5/340.2, and against
that the solver is −1.6 % / −3.0 % (`metrics.json:rung0.rel_err_vs_series`). A
supplementary run at Q = 12, done after seeing the miss, matches the closed form
to −0.7 % / −0.9 % (`results/rung0_qiu_q12_supplementary.json`). CalculiX and
JAX material laws agree to 2e-7 on a one-element test
(`results/material_check.json`).

### Rung 1 — target curves (`results/curves_T*.png`, `results/shapes_T*.png`)

| target | optimiser loss (JAX) | CalculiX NRMSE | tolerance | second criterion | |
|---|---|---|---|---|---|
| T1 stiffening | 0.0002 | **0.0025** | 0.05 | monotonic | PASS |
| T2 plateau | 0.0096 | **0.0101** | 0.05 | plateau dev 0.016 kPa ≤ 0.06 | PASS |
| T3 snap-through | 0.0198 | **0.0200** | 0.08 | peak ε 0.09, valley ε 0.31 | PASS |

Solver gap JAX vs CalculiX 0.25–0.31 % of range; mesh convergence JAX 40×2 vs
80×4 ≤ 0.29 %, CalculiX t/3 vs t/6 ≤ 0.09 % (`results/mesh_convergence.png`).
Only arch-shaped starts reach T2 and T3; sag/flat starts stall at 0.15–0.35
(`results/opt_history.png`).

### Rung 2 — programmed bistability (`results/bistable_curves.png`)

| quantity | target | JAX | CalculiX (t_min/4) | error | |
|---|---|---|---|---|---|
| F_pk (N/mm) | 0.0400 | 0.03999 | 0.03993 | −0.17 % | pass |
| d_2 (mm) | 10.00 | 10.002 | 10.004 | +0.04 % | pass |
| E_back (N·mm/mm) | 0.0450 | 0.04500 | 0.04495 | −0.12 % | pass |

Both rest states stable: tangent positive-definite with dF/dd > 0 in the design
solver; in CalculiX the released shuttle settles at 10.004 mm from 8.30, 10.15
and 10.30 mm and at 0.000 mm from 1.0 mm, with all `*FREQUENCY` eigenvalues
positive. On record against this pass: the CalculiX run at exactly t_min/3
aborted at d = 5.93 mm (near-singular tangent in the design solver at 5.90 mm; t_min/4, t_min/6
and an imperfection-seeded t_min/3 run complete and agree), and one release test
(from d_2 + 0.6 mm) did not converge. It took six optimisation attempts
(`log.md`, `results/opt_bistable_attempt*.json`).

### Gradient check (`gradcheck.py`, `results/gradcheck.png`)

- Implicit gradients match same-branch central FD to ≤ 1e-6 relative for all
  tested parameters at every load step except exactly at the bifurcation.
- Across the mode-3 bifurcation of a symmetric arch (d_c = 1.78 mm) the
  gradient jumps: dF/d(log t0) 0.040 → 0.060, dF/db2 0.0147 → 0.0042.
- For the symmetry-breaking parameter a2 the gradient is 0 before d_c, −0.62 at
  d_c, then decays like (d − d_c)^-1/2; its sign is the branch's, and a central
  FD from re-run paths is exactly 0.
- A fourth boundary found in rung 2 (`snap_diagnostic.py`,
  `results/bistable_stall_diagnostic.png`): an integrated curve quantity
  (E_back) changes when a secondary snap moves or vanishes; the per-state
  gradient predicted the residual going to 0, the actual change had the
  opposite sign (−12.9 % → −19.3 % over the predicted step).

## Interpretation

The forward problem was the hard part, not the optimisation. Every failure in
`log.md` is a continuation or stability failure: CalculiX riding an unstable
branch, my Newton creeping at an indefinite tangent, a line search defeated by
round-off, a force jump sliding across a load-sample point. Once the
solver returned the same curve for the same design, least squares with exact
Jacobians converged in tens of evaluations.

Rung 1 was easier than expected: errors land 4–20× inside tolerance, and the
optimised designs are asymmetric enough that none sits on a bifurcation. The
tolerances were fixed in advance, so they were not tuned to pass, but they were
not a demanding test of this cell family.

Rung 2 is where differentiating through instabilities actually broke. The
optimiser reduced the symmetry-breaking mode (a2 0.3 → 0.15–0.18) and stalled
8–16 % short on a ridge where a secondary snap appears; there the gradient of the area-type objective is wrong in sign for any
finite step. The target was reachable — a start placed one Gauss–Newton step
beyond the ridge converged in 26 evaluations — so the stall was a non-smooth
local optimum, not a physical limit. Getting past it took a diagnosis by hand,
not a better gradient.

The CalculiX agreement (0.1–0.3 %) verifies numerics and geometry hand-off. It
does not verify the shared idealisations: rigid frame, plane strain, no contact,
rate-independent rubber. And the designs are sensitive: with ±0.05 mm / ±3 %
geometric error only 62–88 % of perturbed copies stay inside tolerance
(`results/robustness_T*.json`).

## Diagnostics

Numbers reference `metrics.json` unless noted. There is no validation/test
split in this experiment; "val" below means the independent CalculiX re-solve.

- intended_effect_confirmed: partial — rung 1 and rung 2 pass in CalculiX (`metrics.json:rung1.*.calculix_nrmse`, `metrics.json:rung2_bistable.calculix`); rung 0 misses 2 of 6 pre-registered checks (`metrics.json:rung0.pass_preregistered`).
- leakage_check: targets defined by formula in `PREREGISTRATION.md` before `fem.py` existed, none generated from the parameter space; the one in-space recovery is labelled a sanity check (`sanity.py`, `results/sanity_recovery.json`) — no leakage found.
- overfitting_signal: train=optimiser NRMSE 0.0002 / 0.0096 / 0.0198, val=CalculiX NRMSE 0.0025 / 0.0101 / 0.0200, gap ≤ 0.0023 — only T1 shows the optimiser fitting below its own discretisation error (`results/verify_T1.json:gap_jax40x2_vs_ccx_t3`).
- delta_from_prior: vs jaxfem-topology-optimization (linear compliance, self-checked), this adds finite strain, instabilities and an independent-solver grade; no shared metric to difference (`metrics.json`).
- unexpected_findings: CalculiX's static Newton follows an unstable equilibrium branch on the symmetric beam without warning (`results/rung0_qiu_validation.png`); the curve→shape map is strongly non-unique (in-space recovery to NRMSE 2e-6 with 1.9 mm parameter error, `results/sanity_recovery.json`).
- next_candidates:
  - Replace the fixed-grid area objective in `bistable.py` with an energy-difference objective between tracked equilibria and add pseudo-arclength continuation, then re-run attempts 3–5 to test whether the ridge stall disappears.
  - Re-verify the four final designs in CalculiX with an elastic frame and a rotation-free shuttle to measure how much of the response survives realistic boundary compliance.
  - Pose a harder rung-1 target set (two-plateau curve, prescribed negative-stiffness slope) since the present targets were met 4–20× inside tolerance.

## Follow-up

- Print and test one cell (the shortlist's strongest grade); viscoelastic rubber
  and self-contact are outside both solvers.
- Tolerance-aware design: optimise the expected error under geometric scatter.
- Dynamic-relaxation release test in CalculiX in place of the static release
  that failed from d_2 + 0.6 mm.
