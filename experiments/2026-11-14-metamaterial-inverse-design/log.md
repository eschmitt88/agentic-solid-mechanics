# Log — metamaterial-inverse-design

All times UTC, 2026-11-14.

## 20:31 — orientation
- Read shortlist (track D), trial-3/trial-5 folders and the trial-5 review page.
- Toolchain: JAX 0.10.2, CalculiX 2.23, Gmsh 4.15.2. `micromamba run -n solidmech`
  needs `MAMBA_ROOT_PREFIX=~/projects/.micromamba` (otherwise "prefix does not exist").
- Unit-cell choice: curved elastomer beam between stiff frame walls with a stiff
  mid-span shuttle (negative-stiffness / phase-transforming cell). One family covers
  stiffening (straight/sagging beam), plateau and snap-through (arched beam), and the
  Qiu cosine beam (rung 0) is a member of it, so the validated solver is the design solver.

## 20:34 — reference opened
- scispace copy of the JMEMS paper returned empty text. Opened Qiu's MIT PhD thesis
  (2003) instead: https://pergatory.mit.edu/research/jqiu/thesisscreen.pdf, pdftotext,
  ch. 2 eqs 2.9–2.48. Shape w = h/2 (1 - cos 2πx/l), Q = h/t, F = f l³/(E I h), Δ = d/h.
  Mode-1 cubic F1 (2.38), F3 = 8π⁴ − 6π⁴Δ (2.41, mode 2 constrained, exists for Q > 2.31),
  with higher modes: f_top ≈ 740 EIh/l³, f_bot ≈ 370 EIh/l³, d_top ≈ 0.16h, d_bot ≈ 1.92h,
  d_end ≈ 1.99h, d_mid = 1.33h for Q ≥ 6 (2.47–2.48). My recalled numbers matched the source.

## 20:37 — pre-registration
- Wrote PREREGISTRATION.md (cell, targets, metric, tolerances, verifier) before writing any solver code.

## 20:40–21:05 — rung 0: forward solver built and validated
- `fem.py`: plane-strain Q9 finite-strain solver in JAX float64, CalculiX-form neo-Hooke, body-fitted
  mapped mesh that is a smooth function of the 9 shape parameters, energy-minimising Newton with a
  Cholesky stability check and lowest-mode kick, reaction sensitivities by implicit differentiation.
- `material_check.py`: one CPE8 element under F = diag(1.4,1), diag(0.7,1) and simple shear 0.5:
  CalculiX reactions equal `jax.grad(psi)` to 6e-8 … 2e-7 relative. Same material law confirmed.
- First CalculiX run of the perfect cosine beam (l=100, t=1, h=6): **CalculiX stayed on the unstable
  symmetric branch** — force peak 887 EIh/l³ instead of ~700, then "increment size smaller than
  minimum" at d = 10.04 mm. The JAX solver (energy minimiser + stability check) left that branch at
  the mode-3 bifurcation (d ≈ 1.2 mm). Dead end for the naive verifier deck; fix = seed a 1 %-of-thickness
  mode-3 bump (a_2 = ±0.01 mm). Both signs give the same curve and CalculiX then completes all increments.
- `validate_qiu.py` (results/rung0_qiu_validation.json, .png):
  - JAX 100×2 vs CalculiX CPE6 (lc = t/5), same imperfect geometry: curve NRMSE 0.12 % (tolerance 2 %): PASS.
  - JAX mesh convergence f_top: 25×1 1.1827e-3, 50×2 1.1747e-3, 100×2 1.1719e-3, 100×3 1.1718e-3 N/mm.
  - vs Qiu closed form (2.47–2.48, the Q ≥ 6 asymptote): f_top −5.0 % (inside 5 %), **f_bot −10.8 % (MISS)**,
    d_top +0.042h, **d_bot −0.060h (MISS)**, d_end −0.033h.
  - vs Qiu's own modal series evaluated at the actual Q = 6 (`qiu.py`, eqs 2.23–2.36, 20 modes; gives
    714.5 / 340.2 rather than 740 / 370): f_top −1.6 %, f_bot −3.0 %, d_top −0.0004h, d_bot −0.012h, d_end −0.004h.
  - Reading: the pre-registered check against the rounded constants is missed on f_bot and d_bot because
    740/370 are the large-Q limit; against the same source's equations at Q = 6 the solver is within 3 %.
    The tolerance is not moved; the miss is reported.
- Timing: design mesh 40×2 (810 dof) path of 50 steps = 9 s on 8 CPU cores, Jacobian 0.4 s. CPU only; GPU not needed.
- Quick FD check at a generic arch (h=3, t=1.2): implicit Jacobian vs central FD (h=1e-5) max rel err
  9e-7 (a_1), 2.5e-4 (a_2, the bifurcation-imperfection parameter), 5e-7 (log t0), 3e-7 (b_2).

## 21:07 — rung 1 optimisation launched
- `design.py`: bounded trust-region least squares (scipy trf) on 50 stress residuals with the implicit
  Jacobian; 4 generic starts per target (sag3, flat, arch3, arch6), max 60 evaluations each.

## 21:10 — first failure: T2 and T3 optimisers hung (attempt 1 stopped)
- T2 stalled after eval 11 and T3 after eval 29 (logs kept: results/logs/opt_T{2,3}_attempt1_hung.log); no output for 3 min at
  ~270 % CPU. The trust-region step had pushed the wavy modes to a_3 ≈ 4, a_4 ≈ −4 mm with local thickness 5 mm
  (my bound penalties, weight 3, were far too weak): centre-line radius ~2.5 mm < half thickness, i.e. a folded
  mesh, and the Newton minimiser ground through eigen-solves and 30-halving line searches on garbage.
- Fixes (design-space hygiene, not tolerances): penalty weight 3 → 30; two added validity limits, slope |dy/dx| ≤ 1 and
  curvature × thickness ≤ 0.8 (no folding); Newton: diagonal shift instead of eigen-solve, ≤ 40 iterations, ≤ 14 halvings,
  abort the path at the first non-converged step and hand the optimiser a huge residual so the step is rejected.
- T1 (old code) kept running: it was nowhere near the bounds (sag3 start already at NRMSE 0.0195).
- Relaunched T2, T3 at 21:11.

## 21:12 — gradient check finished (gradcheck.py, results/gradcheck.json, results/gradcheck.png)
- Test design: perfect cosine arch in the design cell (h = 6, t = 1.5 mm); mode-3 bifurcation at d_c = 1.78 mm
  (lowest tangent eigenvalue 1.7e-8 there vs ~1e-5 nearby).
- First version of the check re-ran the whole path for each ±h and gave errors that grew as 1/h (2e-5 → 2e-3):
  the two re-runs sometimes landed on opposite-sign buckled branches whose forces differ slightly. Replaced by
  "same-branch" FD (re-converge from the baseline states) and kept the re-run FD as a second column.
- Same-branch FD vs implicit gradient: ≤ 5e-6 relative for a1, log t0, b2 at every d, including 0.02 mm past d_c.
- The gradient jumps across d_c (kink in F): dF/d(log t0) 0.0398 → 0.0602, dF/db2 0.0147 → 0.0042, dF/da1 0.00449 → 0.00376.
- Symmetry-breaking parameter a2: dF/da2 = 0 before d_c, then −0.62 at d_c, −0.086 at +0.02 mm, −0.022 at +0.3 mm, −0.0007 at +4 mm
  (∝ (d−d_c)^-1/2). Its sign is that of the branch taken; central FD from re-run paths gives exactly 0. With only two
  load points (0.3 mm sub-steps) the re-run did not even follow the sign of a2 for |a2| < ~0.01 mm.
  => at a symmetric design the gradient in symmetry-breaking directions describes one branch, not the function the
  optimiser evaluates. Away from that set (all optimised designs have a2 ≠ 0 and switched = 0) gradients are exact.

## 21:21 — supplementary rung-0 run at Q = 12 (l = 100, t = 0.5, h = 6)
- JAX 100×2 vs Qiu closed form (2.47–2.48): f_top −0.7 %, f_bot −0.9 %, d_top −0.013h, d_bot −0.005h, d_end −0.004h.
  The Q = 6 miss is the closed form's large-Q rounding, not the solver. (Run after the fact; the Q = 6 miss stands.)

## 21:28 — rung 1 optimisations finished (design-solver losses, NRMSE)
- T1: sag3 0.0195, flat 0.0244, arch3 0.0280, arch6 0.0002 (!). T2: sag3 0.187, flat 0.185, arch3 0.150, arch6 0.0096.
  T3: sag3 0.354, flat 0.351, arch3 0.080, arch6 0.020.
- Strongly start-dependent: for T2/T3 only arch starts reach a snap-through-type response; sag/flat starts stall at 0.15–0.35.
- All 12 final designs: no branch switches, no failed steps, bound/validity penalties ≈ 0 (thickness sits on the 3.0 mm
  cap in the T2/T3 designs; T1-sag3 sits on the 0.4 mm floor).
- T1-arch6 is a wavy beam (a2 = −4.7 mm, max slope 0.85); it was found by the old-penalty run but satisfies the new limits.
- Next: CalculiX verification of the best design per target (verify.py).

## 21:30 — rung 2 (programmed bistability) optimisation launched
- bistable.py: residuals = relative errors of [F_pk, d_2, E_back] extracted from the 104-point F(d) curve, chained through the
  implicit Jacobian; 3 cosine-arch starts (h5/t1.6, h6/t1.4, h4.5/t1.2).

## 21:39 — rung 1 verified in CalculiX (verify.py → results/verify_T{1,2,3}.json): all three PASS
| target | optimiser NRMSE (JAX 40×2) | CalculiX NRMSE (CPE6, lc = t_min/3) | tolerance | secondary criterion | verdict |
|---|---|---|---|---|---|
| T1 stiffening | 0.0002 | 0.0025 | 0.05 | monotonic: yes | PASS |
| T2 plateau | 0.0096 | 0.0101 | 0.05 | plateau max dev 0.0155 kPa ≤ 0.06 | PASS |
| T3 snap-through | 0.0198 | 0.0200 | 0.08 | peak (ε 0.09, 0.605 kPa) then valley (ε 0.31, 0.163 kPa) | PASS |
- Solver gap JAX 40×2 vs CalculiX: 0.25 %, 0.31 %, 0.28 % of the target range. For T1 the gap (0.0025) is 15× the
  optimiser loss (0.0002): the optimiser fitted its own discretisation error; the verified number is the honest one.
- Mesh convergence: JAX 40×2 vs 80×4: 0.21 %, 0.29 %, 0.25 %; CalculiX lc t/3 vs t/6: 0.05 %, 0.09 %, 0.08 %.
- CalculiX completed all 100 increments in every run without imperfection seeding (designs are not symmetric-degenerate).
- Laterally free shuttle: curves identical to the laterally fixed ones to all printed digits. Weak test as run: the two
  halves are mirror images, so nothing pushes the shuttle sideways; a sway instability would need a perturbation to show.
- Solver flaw found: the T3 80×2 mesh-convergence run aborted at step 6 ("bad": 45). Cause: near convergence the Armijo
  energy decrease falls below round-off, the line search fails and the step was abandoned at residual ~1e-7. Fix: accept
  the full Newton step when it reduces the residual. T3 verification re-run with the fix.

## 21:40 — rung 2 attempt 1 FAILED: optimiser stalled at all three symmetric starts
- From each perfect cosine arch the first trust-region step lost bistability, and every shorter step made all three
  features worse (e.g. start h5/t1.6: [F_pk, d_2, E_back] = [0.0180, 9.19, 0.0084] → [0.0143, 8.50, 0.0009] at a 5 % step),
  so trf shrank back to the start and stopped on xtol with zero progress (results/opt_bistable_attempt1_symmetric_starts.json).
- Consistent with the gradient-check finding (not separately proven for this run): at a symmetric design the Jacobian columns for the
  symmetry-breaking shape modes belong to one buckled branch and do not predict what a re-run does.
  (Half of the repeated evaluations at the same p also returned "not bistable": the round-off line-search flaw above.)
- Attempt 2 launched 21:41 from starts with broken symmetry (a2 = ±0.3 mm), with the solver fix.

## 21:45 — sway check and labelled sanity check
- sway.py (results/sway_check.json): laterally free shuttle with a constant sideways dead load of 2 % of the target peak force.
  Max sideways drift 0.071 mm (T1), 0.016 mm (T2), 0.033 mm (T3); curves change by < 0.01 % of range. No sway instability.
- sanity.py (LABELLED SANITY CHECK, results/sanity_recovery.json): target generated from an in-space design
  [4.5, 0.6, −0.4, 0, ln 1.3, 0.15, −0.10, 0, 0]. arch6 start: NRMSE 8e-7 at the end (2e-4 at evaluation 9, when this entry was first written), max parameter
  error 1.24 mm; arch3 start: NRMSE 2.4e-6, max parameter error 1.9 mm. The curve→shape map is
  strongly non-unique; the optimiser recovers the response, not the design.

## 21:50 — rung 2 attempt 2 FAILED (solver, not optimiser): stalled at [F_pk, d_2, E_back] = [0.0232, 9.29, 0.0197]
- Start h5/t1.6/a2 = 0.3 improved E_back 0.0095 → 0.0197 in ~16 evaluations, then every further trial step came back
  "not bistable" and trf stopped (results/opt_bistable_attempt2_solver_stall.json).
- Diagnosis: at d = 6.75 mm the path passes a second instability (the buckled shape switches back). There the tangent is
  indefinite and my diagonal-shift Newton crept: 38 shifted iterations, residual stuck at 5e-7, step declared failed, path
  aborted, features undefined. Parameter changes of 1e-7 decided whether this happened: the loss was noise, not physics.
- Fix: saddle-free Newton (eigen-decomposition, |λ|) whenever Cholesky fails, max 60 iterations. The three previously
  inconsistent evaluations now give the same curve. Attempt 3 launched 21:52.

## 21:54 — geometric-tolerance sensitivity (robustness.py, design solver, 16 samples per design)
- Perturbation 1σ: centre-line modes 0.05 mm, thickness scale 3 %, thickness modes 0.02.
- NRMSE median / max / fraction inside tolerance: T1 2.9 % / 10.2 % / 88 %; T2 3.8 % / 13.7 % / 62 %; T3 4.5 % / 15.8 % / 75 %.
  The nominal designs pass with margin; copies made to ordinary printing tolerance often would not.

## 22:06 — rung 2 attempts 3 and 4 STALLED (results/opt_bistable_attempt3_stalled.json, ..._attempt4_hinge_stalled.json)
- Attempt 3 (cosine-like starts with a2 = ±0.3, saddle-free solver), [F_pk, d_2, E_back], target [0.040, 10.0, 0.045]:
  h5/t1.6 → [0.0234, 9.29, 0.0198] (xtol); h5.5/t2.0 → [0.0356, 10.37, 0.0376] (−11 %, +3.7 %, −16 %; stopped moving after
  eval ~28 of 40); third start stopped by me.
- Attempt 4 (my idea: flexure-hinge thickness profile, thin at wall and shuttle): start had the right barrier-to-force ratio
  (E_back/F_pk 2.7 mm vs target 1.1) but too little force; the optimiser stalled at [0.0132, 11.14, 0.0269].
- Common symptom: at the stall point the same design (to 3 decimals) returned E_back anywhere in 0.03748–0.03762. The
  optimiser pulls a2 toward zero (0.3 → 0.18), i.e. toward the symmetric, bifurcating design, where two stable buckled
  states exist at the same displacement; with fixed 0.125 mm load steps Newton hopped onto either one, so the loss was
  discontinuous at the 0.4 % level and the trust region collapsed.
- Fix: adaptive continuation — bisect any load step in which max|Δu| > 4 |Δd| (up to 2⁵), so the connected path is followed
  through the narrow imperfect-bifurcation zone. Five evaluations at the stall point ± 1e-6 now agree to 6 digits.
- Attempt 5 launched 22:08: continue from attempt 3's best, plus three further starts, max 60 evaluations each.

## 22:20 — rung 2 attempt 5 STALLED again; cause found (snap_diagnostic.py, results/bistable_snap_diagnostic.json)
- Attempt 5 results [F_pk, d_2, E_back]: continue-from-attempt-3 [0.0356, 10.37, 0.0376] (no movement);
  h6/t1.9 → [0.0368, 10.50, 0.0392] = (−7.9 %, +5.0 %, −12.9 %), best so far; hinge h5/t1.5 → [0.0172, 10.37, 0.0281];
  hinge h5.5/t1.4 → [0.0262, 11.32, 0.0415] (finished later, also stalled). Adaptive bisection made evaluations repeatable but did not remove the stall.
- At the stalled h6/t1.9 design no bound or validity limit is active (thickness 1.55–2.55 mm, max slope 0.36, penalty 0).
  The feature Jacobian has full rank (singular values 2.55, 1.70, 0.50) and its Gauss–Newton step (|Δp| = 0.10) predicts
  all three residuals → ~0. Actual along that step (fraction 0.1 / 0.3 / 1.0):
  F_pk −7.3 / −6.3 / −3.6 %, d_2 +4.6 / +3.6 / +1.2 % (as predicted), but E_back −14.1 / −15.9 / −19.3 % (predicted: → 0).
  The gradient of E_back has the wrong sign for any finite step in that direction.
- Fine resolution (0.02 mm) of F(d): the stalled design has a steep step in force at d = 7.40 mm (+0.9 mN/mm within 0.02 mm,
  slope 9× the surrounding curve): a secondary snap of the buckled shape on the negative-force branch. After 30 % of the
  step that feature is gone. The optimiser has climbed onto the ridge where this snap appears/disappears; E_back is
  larger with the snap, the pointwise implicit gradient (exact for each state) cannot see a jump moving or vanishing,
  and on the far side the function falls. A non-smooth local optimum of the curve functional, not a bound.
- Correction to the 22:06 entry: the 0.4 % scatter in E_back was this force jump sliding across the d = 7.25 mm sample
  (F there differs by 1.15 mN/mm between two designs 1e-6 apart), not Newton hopping between two long-lived branches.
- Attempt 6 launched 22:30 from the far side of the ridge (stalled design + its Gauss–Newton step).
- verify_bistable.py started on the best design so far (h6/t1.9 from attempt 5) to get the CalculiX numbers regardless.

## 22:40 — rung 2 attempt 6 converged in the design solver
- Start: attempt-5 stalled design plus its own Gauss–Newton step (the far, snap-free side of the ridge).
  trf converged in 26 evaluations to [F_pk, d_2, E_back] = [0.039999, 10.0015, 0.044999] (target [0.040, 10.0, 0.045]).
  So the target was reachable; attempts 3–5 were stuck on the non-smooth ridge, not at a physical limit.
- The verification that had been started on the attempt-5 design was stopped (its design-solver part had confirmed the miss:
  −7.9 %, +5.0 %, −12.9 %); verification restarted on the attempt-6 design.

## 23:03 — rung 2 verification: fine design-solver mesh abandoned
- verify_bistable.py spent 22 min in the 80×4 (2862 dof) design-solver run without finishing (dense eigen-solves at the
  instabilities, 104 load steps with bisection, box load ~25). Stopped; mesh check for rung 2 uses 80×2 instead, plus the
  two CalculiX meshes.

## 23:11–23:16 — rung 2 verified in CalculiX (verify_bistable.py, verify_bistable_extra.py → results/verify_bistable.json)
- Design solver 40×2: [F_pk, d_2, E_back] = [0.039999, 10.0015, 0.044999]; 80×2: [0.039981, 10.0031, 0.045001].
- **CalculiX at the pre-registered coarsest mesh (lc = t_min/3) ABORTED at d = 5.93 mm** ("increment size smaller than
  minimum"), just past the tipping point (d_u = 5.76). Design solver: lowest tangent eigenvalue dips to 2.2e-6 at d = 5.90 mm
  and the continuation bisected to depth 5 at 5.92 — a near-bifurcation on the return branch.
- CalculiX t_min/4, as designed: completes; errors −0.17 %, +0.04 %, −0.12 %. t_min/6: −0.18 %, +0.04 %, −0.15 %.
  t_min/3 with a ±0.01 mm a2 bump: completes for both signs; errors ≤ 0.2 % (F_pk), ≤ 0.05 % (d_2), −0.8 % (E_back).
  All inside the 5 % tolerance. Graded on t_min/4 (the pre-registration says "element size ≤ t_min/3").
- Stability of the two rest states. Design solver: tangent positive-definite and dF/dd > 0 at both (0.0267 and 0.0148 N/mm²).
  CalculiX release tests (shuttle force-free after being held): from 8.30 mm → settles at 10.004 mm; from d_2 + 0.15 and
  + 0.30 mm → 10.004 mm; from 1.0 mm → 0.000 mm; *FREQUENCY about each settled state: all eigenvalues positive.
  **Release from d_2 + 0.6 mm FAILED to converge**: the static Newton solve overshot to 5.92 mm (the tipping point) and stopped.
  A static release is not a physical relaxation; reported as a failed test, not hidden.
- Forward vs backward sweep in the design solver differ by at most 0.24 mN/mm (0.6 % of the peak), near the force corner at d ≈ 4 mm.
- Verdict rung 2: PASS on the numbers, with the aborted t_min/3 run and the one failed release test on record.

## 23:20 — wrap-up
- Figures (plots_*.py), metrics.json and the review page (build_page.py → docs/qa/trial-9-metamaterial/) generated from results/*.json.
- Not done: no dynamic-relaxation check of the release; shuttle rotation never released; no elastic-frame model; rung-1
  targets turned out easy (errors 4–20× inside tolerance), a harder target set was not attempted.
