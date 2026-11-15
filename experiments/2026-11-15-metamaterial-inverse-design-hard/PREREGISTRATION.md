# Pre-registration — trial 12 (metamaterial inverse design, hard track)

Written 2026-11-15 16:37 UTC. At this point `fem.py` (trial 9's solver generalised to more shape modes and
to a series stack) exists and has been checked against trial 9 (same T2 curve to 3e-13 kPa); **no
optimisation of any target below has been run, and the snap remedy has not been coded.** Nothing in this
file is edited after the first optimisation run. Misses are reported as misses.

## Fixed set-up (inherited from trial 9)

Cell 70 × 25 mm (W × H), beam span 60 mm, 4 mm shuttle, plane strain, per mm depth; neo-Hooke
C10 = 0.5 MPa, D1 = 0.428571 /MPa (E = 2.8 MPa, ν = 0.4). Stress σ = F/W (kPa), strain ε = travel / height.
Geometric validity limits (trial 9): 0.4 ≤ t ≤ 3 mm, |y_c| ≤ 9 mm, |slope| ≤ 1, curvature × thickness ≤ 0.8.

Design families:
- **S9** — trial 9's: 4 centre-line cosine modes + log t0 + 4 thickness modes.
- **S13** — one cell, 6 centre-line modes + log t0 + 6 thickness modes.
- **D26** — series stack of two S13 cells (shuttle of cell A carries the frame of cell B, top shuttle pushed
  by the total travel D; strain ε = D / (2H), so ε = 0.5 is D = 25 mm).

## Targets

Metric for curves: NRMSE = rms(σ − σ*) / (max σ* − min σ*) on the 50 strains 0.01 … 0.50.

- **H1 — published foam plateau (S13).** Gibson & Ashby open-cell elastomeric foam: E* = E_s (ρ*/ρ_s)²,
  elastic-collapse plateau σ*_el = 0.05 E_s (ρ*/ρ_s)², densification at ε_D = 1 − 1.4 ρ*/ρ_s. Source opened
  this session: L. J. Gibson, MIT OCW 3.054 "Cellular Solids", Lecture 7 "Foams" notes (C1 ≈ 1, C4 ≈ 0.05,
  ε_D), https://ocw.mit.edu/courses/3-054-cellular-solids-structure-properties-and-applications-spring-2015/a7c6b7bd4d16424aeeff92f1bb277c59_MIT3_054S15_L7_foams_trans.pdf
  (after Gibson & Ashby, Cellular Solids, 2nd ed., CUP 1997). With E_s = 2.8 MPa and ρ*/ρ_s = 0.055:
  E* = 8.47 kPa, σ*_el = 0.4235 kPa, ε_D = 0.923 (outside the range). Target σ*(ε) = min(E* ε, σ*_el):
  linear to ε = 0.05, then flat to ε = 0.50.
  **Pass:** NRMSE ≤ 0.03 **and** |σ − 0.4235| ≤ 0.0212 kPa (5 %) at every sample strain 0.08 ≤ ε ≤ 0.50.
  (Trial 9's plateau was held to 3.9 % over the shorter range 0.12–0.33 and then rose; feasibility unknown.)
- **H2 — two-step staircase (D26).** PCHIP through (0,0) (0.04,0.28) (0.08,0.35) (0.22,0.35) (0.27,0.55)
  (0.31,0.70) (0.45,0.70) (0.50,0.90) in (ε, kPa).
  **Pass:** NRMSE ≤ 0.04 **and** |σ − 0.35| ≤ 0.028 on 0.10 ≤ ε ≤ 0.20 **and** |σ − 0.70| ≤ 0.056 on
  0.33 ≤ ε ≤ 0.43 (8 % bands). Feasibility unknown.
- **H4 — expected infeasible (S13).** PCHIP through (0,0) (0.04,0.20) (0.12,−0.40) (0.40,−0.40) (0.50,0.30).
  Its running integral ∫σ dε is negative from ε ≈ 0.09 on, i.e. the cell would have to hold less elastic energy
  than at rest: impossible for any hyperelastic cell that is stress-free at rest. **Pass rule is the same as for
  any target:** NRMSE ≤ 0.05 in CalculiX. Expected verdict: MISS. Supporting certificate (not the grade): the
  smallest NRMSE of any sampled curve with non-negative running (trapezoid) integral, by quadratic programme.
- **B — programmed bistability (S9), trial 9's target unchanged:** F_pk = 0.040 N/mm, d_2 = 10.0 mm,
  E_back = 0.045 N·mm/mm, each within 5 % in CalculiX. Used for the snap test below; the design verified is the
  final design of the best new arm from the first pre-registered start on which that arm succeeds.
- **R — robust design (S9), trial 9's target T2 unchanged** (PCHIP (0,0) (0.05,0.30) (0.10,0.40) (0.35,0.40)
  (0.42,0.55) (0.50,1.00); pass NRMSE ≤ 0.05 and |σ − 0.40| ≤ 0.06 on 0.12–0.33). Also on H1 if, and only if,
  H1's nominal design passes in CalculiX.

Starts. H1, H4 (S13, cosine arches h/t in mm): sag3 (−3/1.2), flat (−0.3/1.2), arch3 (3/1.2), arch6 (6/1.5),
arch8 (8/1.5), wavy (h 6, a_2 −2, t 2.0). H2 (cell A + cell B): (4/1.2 + 6/2.0), (3/1.0 + 5/1.8),
(6/1.5 + 6/2.2), (−0.3/1.2 + 6/1.5). Optimiser: trial 9's bounded trust-region least squares with the exact
Jacobian, ≤ 80 evaluations per start. The design-solver optimum with the lowest loss is the one verified.

## Verification rule (the rule graded by)

CalculiX 2.23, NLGEOM, same material, both half-beams of every cell meshed (no symmetry), CPE6, 100 increments,
shuttles non-rotating and laterally fixed. **Mesh ladder: element size t_min/3, t_min/4, t_min/6** (t_min =
thinnest point of the design). The graded numbers are those of the **finest mesh that completes all increments**.
If no ladder mesh completes, the design is re-run at t_min/4 with a ±0.01 mm a_2 imperfection (both signs) and
graded on the worse of the two; if those abort too the verdict is **FAIL (unverifiable)**. The spread between
completed ladder meshes is reported; a spread above 1 % of the target range is flagged. The optimiser loss is
reported next to the CalculiX number and never used as the grade. For perturbed copies (robustness) one mesh is
used: t_min/3, falling back to t_min/4 on abort; an abort on both counts as a failed copy.

Verdict vocabulary: PASS (CalculiX inside every tolerance), MISS (verified, outside), FAIL (unverifiable).
For every MISS the report also gives the best loss per start and the projected-gradient norm at the best design
(stationary point of the smooth problem, or stalled).

## Fabrication-error model and robust design

Perturbation δp ~ N(0, diag(sd²)) on the S9 parameters, sd = 0.05 mm on each of the 4 centre-line modes, 0.03 on
log t0 (3 % thickness), 0.02 on each of the 4 thickness modes — trial 9's model. (For S13 the same on the first
four modes of each kind, 0 on modes 5–6.)

- Robust optimisation: sample-average approximation, K = 8 fixed samples (seed 1) plus the nominal; objective
  = mean over the 9 copies of NRMSE² (stacked least squares), started from the nominal optimum.
- Evaluation: 64 fresh samples (seed 2026), identical for nominal and robust design. Pass fraction = share of
  copies meeting the target's full pass rule, (a) in the design solver on all 64, (b) in CalculiX on the first 24.
- Claim "robust design helps" only if the pass fraction rises by ≥ 0.15 in both (a) and (b). The nominal error
  of the robust design is reported next to the nominal optimum's (the price paid).

## Snap test (weakness 3)

Problem: target B. Starts: the 11 distinct starts of trial 9's attempts 1–5 (9 parameters; a_1, a_2, log t0, b_2
shown, others 0 unless listed):
cos_h5_t1.6 (5, 0, ln 1.6), cos_h6_t1.4, cos_h4.5_t1.2 [symmetric, attempt 1]; h5_t1.6_a2 (a_2 = +0.3),
h5.5_t2.0_a2 (+0.3), h6_t1.4_a2 (−0.3) [attempts 2–3]; hinge_h5_t1.1_b-0.8 (a_2 0.2, b_2 −0.8) [attempt 4];
continue_attempt3_best (the full vector in trial 9's opt_bistable_attempt5_A.json), hinge_h5.5_t1.4_b-0.7,
h6_t1.9_a2 (+0.3), hinge_h5_t1.5_b-0.4 [attempt 5]. The attempt-6 start (hand-placed across the ridge) is excluded.

Arms (all automatic, same solver, same bounds, ≤ 60 path evaluations per start, no restarts by hand):
- **B0 baseline** — trial 9's method: features from the 0.125 mm grid (parabola peak, linear zero crossings,
  trapezoid area), per-state implicit Jacobian, trf.
- **A fold-aware** — the three features as state functions: F_pk at the refined limit point, d_2 at the refined
  zero crossing, E_back = Π(d_u) − Π(d_2) − Σ_snaps [Π⁻ − Π⁺] with every snap between d_u and d_2 located as a
  fold point d_s; gradient = envelope terms + (F⁻ − F⁺) · dd_s/dp, with dd_s/dp = −(φᵀ∂_pR)/(φᵀ∂_dR) from the
  augmented limit-point system (φ = null vector of the tangent at the fold). Same trf.
- **B0+O** and **A+O** — the same with an automatic outer loop: (1) if the start's path contains a
  branch switch at a bifurcation, perturb p by 0.05 × a fixed-seed unit normal vector (up to 3 tries) until it
  does not; (2) when trf stops with any feature error > 1 %, take the damped Gauss–Newton step (λ = 1e-2) from
  its end point, evaluate fractions 1, 0.5, 2 of it, and restart trf from the bistable candidate with the lowest
  loss even if that loss is higher than the current one; if none is bistable, restart from a 0.1 × fixed-seed
  normal perturbation; at most 4 restarts, all inside the 60-evaluation budget.

Scoring, identical for every arm: the final design is re-evaluated with trial 9's own feature extractor
(`bistable.indices` / `bistable.features`, grid 0.125 … 13 mm). **Success = all three features within 1 % of
target.** Reported: success count out of 11 per arm, evaluations used, and the split symmetric / non-symmetric
starts. Headline comparison: A+O against B0. The remedy "works" if it succeeds on ≥ 8 of 11 where B0 succeeds
on ≤ 3; anything else is reported as partial or failed.

Gradient check for the new sensitivities (before the snap test is launched): at trial 9's stalled design
(attempt 5, h6_t1.9_a2 end point) and at one design with a clear snap between d_u and d_2: dd_s/dp and dE_back/dp
against central finite differences of independently re-located folds / re-computed energies. Threshold: relative
error ≤ 1e-3 on the three largest components. A failed check is reported and the arm still run.

## Relaxed idealisation (weakness 4)

Not pass/fail by itself; reported as shift of the curve and as "does the design still meet its tolerance".
- **3D, finite depth** (mandatory): final H1, R (nominal and robust) and B designs re-solved in CalculiX as a
  10 mm deep 3D body (free side faces, symmetry at mid-depth, C3D15 wedges, in-plane size t_min/2.5, 4 layers over
  the half depth), force divided by depth. Expectation written now: forces fall by between 0 and 16 %
  (1 − (1 − ν²)), the plane-stress limit, more for thin sections.
- **Elastic frame** (if time allows): wall posts and shuttle meshed in the same elastomer instead of rigid.
Self-contact and rate dependence are not attempted.
