# Log — instability-path-following

All times UTC, 2026-11-14 unless noted. Chronological, including dead ends.

## ~20:30–20:40 — orientation and reference hunt

- Read `docs/2026-11-14-hard-problem-shortlist.md` (track A), the VHB
  experiment folder and the trial-5 review page for house style.
- CalculiX needs `MAMBA_ROOT_PREFIX=~/projects/.micromamba` for
  `micromamba run -n solidmech ccx`; without it micromamba looks in
  `~/.local/share/mamba` and fails ("prefix does not exist").
- Reference hunt (PDFs downloaded to `/tmp/instab_refs`, not kept in the
  repo; URLs in README):
  - Biot: Cao & Hutchinson 2012 gives λ1/λ2 = 0.2956, plane-strain
    λ_W = 0.5437, ε_W = 0.456, eigen-condition r³ − 3r² − r − 1 = 0
    (r = λ2/λ1 = 3.3830). The task brief's 0.456 is confirmed. The
    shortlist's "0.444" is NOT the plane-strain value (0.444 is, from memory,
    the uniaxial-compression variant; not verified, not used).
  - Lee frame: could not find the often-quoted limit load in the arc-length
    paper (arXiv 2005.10192 only has a figure). Found NAFEMS NL7 values via
    the Abaqus Benchmarks Guide: P·L²/EI = 18.552 at u_y/L = 0.407 and
    31.887 at u_y/L = 0.784. Two mirrors failed (vt.edu 403, mit.edu 410),
    uclouvain mirror worked. De Gruyter paper (cls-2022-0212) blocked (405).
  - Deep arch: Horák et al. arXiv 2111.11154 Table 5: converged 8.972922.
  - Koiter: no opened source states the simple formula I remembered
    ((1−ρ)² = (3c/2)ξρ). Hutchinson 2010 writes out Koiter's upper-bound
    eigenvalue equation in full (eq. 13–15) — read from the rendered PDF
    page because pdftotext mangles the equations. I grade against that.
    One exponent in eq. 13 (b1) is hard to read in the scan; resolved by the
    requirement that ξ = 0 must return σ/σ_cl = 1 at γ = 1/2 (it does only
    with b1 = ½(¼+γ²)⁻²[−¼ + cξγ²/(1−σ)] for the cylinder).
  - Budiansky & Hutchinson 1972 fig. 1/eq. 8: for the infinitely long shell
    the exact bifurcation load is slightly BELOW Koiter's bound (long-wave
    modes) — expected sign of any deviation, noted before running.
  - NASA SP-8007 p. 5 read from the rendered page: γ = 1 − 0.901(1 − e^{−φ}),
    φ = (1/16)√(r/t).
- 20:40 wrote the pass criteria into README.md before any solver code ran.

## 20:40–20:48 — rung 1 (path following)

- Wrote `pathfollow.py` (pseudo-arclength predictor/corrector with a bordered
  solve, inertia count of K as stability monitor, bisection on the arc-length
  step to locate limit points and bifurcations, branch switching along the
  null vector) and `rung1_truss.py`. `uv pip install sympy` for the symbolic
  check of the closed form.
- Truss primary path: limit and bifurcation loads match the closed form to
  round-off at the first try (same discrete model, so this only proves the
  solver, not the modelling).
- **Wrong turn 1 — branch switch silently stayed on the primary path.** The
  "switched" branch had sideways displacement 3e-16. Cause: I orthogonalised
  the null vector against the path tangent stored at the located bifurcation
  point, but that tangent comes from solving K t = q with K singular to
  1e-9, so it was pure round-off along the null vector itself; subtracting
  it removed the mode. Caught only because I had written a check of the
  switched branch against its closed form (circle x²+y² = h²−2b², P linear
  in y) — the load–deflection plot alone would have looked plausible.
  Fix: primary tangent computed with the critical mode deflated out of the
  eigen-decomposition; and in normal stepping, fall back to the secant when
  the solved tangent disagrees with it. After the fix the branch follows the
  closed-form circle to 8e-14.
- Beams (`rung1_beams.py`): 2-node Simo–Reissner element, energy only,
  f_int and K by `jax.grad`/`jax.hessian`.
- **Wrong turn 2 — arch runs died near zero load** ("step size underflow").
  My convergence test scaled the residual tolerance with |λ|, and λ is a
  force here; near λ = 0 the tolerance (9e-7) fell below the round-off floor
  of the residual (7e-6 with EA = 1e8). Fixed: tolerance relative to a fixed
  force scale plus a stagnation test on the Newton update.
- Results (finest mesh / Richardson): Lee first limit load 18.5504 / 18.5466
  vs 18.552; deep arch 8.97410 / 8.97284 vs 8.972922; both converge at
  order 2.00.
- **Criterion 1d missed, and the criterion itself was badly posed.** Load at
  u_y/L = 0.784 comes out 39.43, reference 31.887 (+23.6 %, tolerance 2 %).
  The frame is by then a taut chain: an inextensible frame cannot go beyond
  u_y/L = 0.7796 at all (intersection of circles of radius 144 and 96 about
  the supports), so the curve is almost vertical (dP/d(u/L) ≈ 2060) and a
  0.4 % displacement difference is a 24 % load difference. Read the other
  way round — deflection at the reference load 31.887 — my path gives
  u_y/L = 0.7809 (−0.40 % vs 0.784); Abaqus' own B22 result on the same
  page is 0.781 (−0.38 %). I keep 1d as a recorded MISS of the criterion as
  written and report the well-conditioned reading as a post-hoc diagnostic;
  the tolerance is not changed.
- Load-controlled Newton: truss — 29 iterations at the first step past the
  limit load, then lands on the far (inverted) branch, never seeing the
  bifurcation; Lee frame — no convergence at P·L²/EI = 19, 20, 21, then at
  22 it converges (28 iterations) to the taut far branch at u/L = 0.775;
  arch — no convergence at any load above 8.5.
- CalculiX cross-check (`rung1_ccx_lee.py`, B32R, displacement control
  because ccx has no Riks): P_max·L²/EI = 18.533 / 18.519 / 18.518 for
  20 / 40 / 80 elements, 0.17 % below my converged value.

## 20:48–20:56 — rung 2 (Biot surface instability)

- Derived the finite-depth dispersion relation (4×4 determinant) from Cao &
  Hutchinson eq. 2.6 as an independent semi-analytic check; its deep limit
  reproduces r³ − 3r² − r − 1 = 0 (ε = 0.456311).
- **Attempt 1: Taylor–Hood Q2/Q1 mixed elements — FAILED, and in an
  instructive way.** Hessian of the mixed Lagrangian by autodiff, lowest
  constrained eigenvalue by sparse shift-invert near zero, marching down in
  stretch from λ = 0.70.
  - 4×6 mesh: ε_c = 0.4586 (harmonic 1) — looked right.
  - 8×12, 16×24, 32×48: ε_c = 0.3221, 0.3217, 0.3216 in harmonics 7, 14,
    28. A threshold that is "converged under mesh refinement" to three
    digits, close to the published creasing strain 0.35 — a wrong answer
    that would be very easy to believe. The harmonic number scaling with the
    mesh is what gives it away: it is a mesh-scale mode.
  - 64×96: already unstable at the start of the march (strain 0.30).
  - Dense check with the null-space method on the 4×6 mesh: at strain 0.40
    the constrained stiffness already has 3 negative eigenvalues (−1.66,
    −1.32, −1.32). **Shift-invert returns the eigenvalues nearest the shift,
    not the lowest ones**, so it stepped over modes that were already far
    negative; even the 0.4586 on the coarse mesh was luck.
  - Full inertia count (dense eigenvalues of the saddle matrix minus n_p):
    all three meshes lose definiteness between strain 0.02 and 0.03, with
    more negative eigenvalues the finer the mesh. The offending mode lives in
    the top element row: v_y on mid-side nodes only, v_x on corner nodes
    only — an hourglass-like pattern that the continuous Q1 pressure cannot
    see, so it is not incompressible pointwise and feeds on the −p·det(F)
    term (p0 = μ/λ² is large).
  - The base state itself is fine (equilibrium residual 1e-15), so this is a
    discretisation failure, not a formulation bug. I recall published
    warnings that mixed elements show unphysical instabilities under large
    compression when incompressibility is only enforced weakly (Auricchio
    and co-workers) — from memory, not opened, not used for grading.
  - Verdict on criterion 2a for this attempt: MISS (finest-mesh detected
    value 0.3216 or 0.03 depending on the eigen-solver, tolerance ±0.0046).
- **Attempt 2: enforce incompressibility exactly.** Perturbation written
  with a stream function, v = (ψ,₂ , −r ψ,₁), which satisfies the linearised
  constraint r v₁,₁ + v₂,₂ = 0 pointwise; bicubic Hermite (BFS) C¹ rectangles;
  K = Hessian by autodiff of the same energy restricted to that space;
  stability test = banded Cholesky (succeeds iff K is positive definite),
  scan from strain 0.02 in steps of 0.02 then bisection to 1e-11.
  - Stable at every scan point below the threshold on every mesh (no
    spurious modes), first loss in harmonic 1.
  - ε_c = 0.48753, 0.458406, 0.4564522, 0.45632003, 0.456311556,
    0.4563110229 for 2, 4, 8, 16, 32, 64 elements per wavelength;
    reference 0.4563109873. Error falls ×16 per halving (order 4),
    monotone from above as a conforming Ritz method must.
  - Depth study (16 el./wavelength) against the finite-depth determinant:
    H/W = 1, 2 agree to 2e-5; H/W = 0.5, 0.25, 0.125 pick harmonics 2, 2, 3
    and sit 0.2–0.8 % above the analytic value for that harmonic (coarse
    in depth there; consistent sign).
- Criterion 2a/2b: PASS with the second discretisation; the first one is
  kept in the script and the figure as the failure record.

## 20:57–21:12 — rung 3 (cylinder), first half

- 20:57 fixed the geometry in `config.yaml` before running anything:
  R/t = 200, ν = 0.3, L = 4 imperfection wavelengths (L/R = 0.98).
- Method decision. CalculiX has no arc-length procedure and its nonlinear
  static step is plain Newton, so it cannot by itself say where a
  bifurcation is. I therefore wrote my own solver for the graded curve
  (`rung3_cylinder.py`): Donnell shell, Hermite-cubic elements along the
  axis, Fourier harmonics around the circumference; nonlinear axisymmetric
  path by the rung-1 continuation code; bifurcation detected as the first
  loss of positive definiteness (Cholesky) of the per-harmonic tangent
  stiffness K_n, n = 1..40, located by bisection. This is exactly Koiter's
  1963 problem statement, solved without his one-term mode assumption.
  CalculiX S8R is used as the independent check on a finite shell.
- Reference formula sanity: Hutchinson eq. 15 gives 0.9984 at ξ = 1e-6 and
  0.196 at ξ = 1 ("one thickness → a fifth", as Budiansky & Hutchinson say).
  The simple form I remembered, (1−ρ)² = (3c/2)ξρ, is 2–4 points higher
  (0.611 vs 0.576 at ξ = 0.1; 0.236 vs 0.196 at ξ = 1) — it is only the
  leading small-ξ term. Not graded against.
- 3a: linear eigenvalue, cell BCs: 1.00032, 1.00012, 1.00002, 1.00001 ×
  classical for 4, 8, 16, 32 elements per wavelength. Simply supported
  ends: 0.98817.
- 3b: CalculiX `*BUCKLE`, quarter model: 0.9948, 0.9926, 0.9916, 0.9913,
  0.9883 for 4, 6, 8, 12, 16 elements per wavelength (416…6528 S8R). Not
  monotone in the last step — the finest mesh picks up a slightly lower
  mode; it then agrees with my simply-supported value (0.98817) to 1.5e-4.
- 3c (cell BCs, 32 el./wavelength; 8 and 16 agree to 4 digits):
  ξ = 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0 → ρ = 0.6641, 0.5524, 0.4199,
  0.3463, 0.2646, 0.2185, 0.1954; Koiter bound 0.6837, 0.5763, 0.4502,
  0.3737, 0.2850, 0.2279, 0.1963. Always below the bound, by at most 0.030,
  and above the Budiansky–Hutchinson long-wave limit — the ordering theory
  predicts. Critical harmonics n = 9, 9, 8, 8, 7, 6, 8 are lower than the
  n ≈ 13 of Koiter's one-term mode: the true critical mode is a longer-wave
  one. Longer cells (8 and 16 wavelengths) drop the load further (0.535 at
  ξ = 0.1, 0.320 at ξ = 0.3), moving towards the long-wave limit.
- **CalculiX nonlinear, first surprise.** ξ = 0.3, displacement control,
  seed 1e-3 t: the run completed with the load still rising (0.374 at the
  last increment) and I first read that as "Newton stepped over the
  bifurcation". Second run with a 10× larger seed and half the increment:
  same curve. Looking at the tangent stiffness along the path settles it:
  it drops from 0.59 to 0.02 at P/P_cl = 0.3727 and the load then creeps up
  (0.403 at 0.875 of the classical end shortening). So CalculiX did
  bifurcate, at the load my solver predicts for these ends, but under
  prescribed end shortening this shell does not collapse at that point — it
  keeps carrying a slowly rising load in the buckled state. "Collapse load
  = maximum reaction" (my criterion 3d as written) is therefore not defined
  for this case; I report the knee (tangent stiffness below a quarter of the
  initial one) as the CalculiX buckling load and say so.

## 21:12–21:30 — rung 3, CalculiX cross-check: two real problems found

- CalculiX quarter model as planned (seed 1e-3 t, 100 increments), judged by
  "Newton failure → last load; else first local load maximum; else knee":
  - ξ = 0:   0.930 (Newton fails) vs own 0.865 → +7.5 %
  - ξ = 0.1: 0.645 (small load drop, then a stable buckled state) vs own
    0.580 → **+11.2 %, outside the 10 % tolerance**
  - ξ = 0.3: 0.3727 vs own 0.3749 → −0.6 %
  - ξ = 1.0: **no buckling at all** up to 1.25× the classical end
    shortening (load 0.294, own bifurcation load 0.205).
- **Problem A — Newton steps over the bifurcation (ξ = 0.1, also ξ = 0).**
  Evidence: my axisymmetric path, continued past its own bifurcation, lies
  on top of the CalculiX curve (0.4 % apart) all the way to 0.63, where my
  stability test already counts three unstable harmonics (n = 10, 12, 14).
  CalculiX simply stays on the unstable axisymmetric solution because a
  1e-3 t seed is too small to pull Newton off it within 1 % increments.
  With seed 1e-2 t CalculiX fails to converge at 0.575 (−0.8 % vs own); with
  5e-2 t it peaks at 0.569 (−1.9 %). So the CalculiX "collapse load" moves by
  12 % with a modelling choice (seed size) that has no physical meaning at
  this level — the reason I did not use it for the graded curve.
- **Problem B — the quarter model itself is too stiff at large ξ.** The
  axial stiffness of the ξ = 1 shell in the *linear* range was 0.318 in
  CalculiX against 0.302 in my solver. Chased it:
  - own solver with infinite-shell BCs reproduces Koiter's exact
    pre-buckling stiffness 1/(1 + c²ξ²) to 5 digits → my Donnell code is
    right for what it models;
  - CalculiX mesh refinement 4…16 elements per wavelength: 0.321…0.318 →
    not a mesh effect;
  - third opinion, CalculiX axisymmetric *continuum* elements (CAX8, no
    shell theory, `rung3_ccx_axisym_check.py`): 0.3026 relative to the
    perfect shell, own solver 0.3009 (ξ = 0.3: 0.8445 vs 0.8435; ξ = 0.1:
    0.9871 vs 0.9870) → the continuum agrees with my solver, the S8R quarter
    model is the outlier;
  - S8R quarter model with the rotational symmetry constraints removed:
    0.2975; full 360° S8R model: 0.2978. So the culprit is the rotational
    boundary condition on the symmetry edges: CalculiX turns shell nodes
    with rotational constraints into rigid "knots", and on the corrugated
    geometry that stiffens the whole shell by 7 %. It is invisible for the
    perfect shell (1.0028 either way) and small at ξ ≤ 0.3, which is why
    the first comparisons looked fine.
- Decision: keep the planned quarter-model numbers as the recorded outcome
  of criterion 3d (2 of 4 amplitudes outside tolerance), and rerun on a
  full 360° model (6592 S8R, no symmetry planes, no rotational constraints)
  with seed 1e-2 t as the corrected cross-check. Launched 21:30 through the
  coordinator (`job-asm-instab-ccx-full-11142130`), three amplitudes in
  parallel, 3 threads each.

## 21:33–21:37 — rung 3: a hole in my own graded calculation (3c)

- Wanted one more check of the own solver: a cell only one ripple
  wavelength long admits exactly the axial periodicity of Koiter's trial
  mode, so it should land just under his bound. It did not — it gave
  0.785, 0.733, 0.952 … for ξ = 0.05, 0.1, 0.2, far ABOVE an upper bound.
- Cause: symmetry conditions at the cell ends admit only modes that are even
  about the ends. With ξ > 0 the ends sit on outward crests; Koiter's mode
  has its largest waves on the inward troughs (where the hoop stress is
  compressive) and is odd about the crests, so my cell excluded it. My
  reasoning while designing the solver, that the sign of ξ cannot matter (not written down at the time),
  was wrong: shifting the cell by half a ripple changes which mode family
  is admitted.
- With ξ < 0 the one-wavelength cell gives 0.5714 (n = 13), 0.3669 (n = 12),
  0.1950 (n = 9) for ξ = 0.1, 0.3, 1.0, against Koiter's 0.5763 (n = 13.0),
  0.3737 (n = 12.0), 0.1963 (n = 8.7): just below the bound, same wave
  numbers. That is the check I was after. (Budiansky & Hutchinson say the
  exact result for this mode class is within 0.0042 of Koiter's; I get
  0.005–0.007 at ξ = 0.1–0.3 — close, not identical; not resolved.)
- Consequence for the graded 3c numbers: the infinitely long shell admits
  both families, so the right quantity is the minimum over both signs. For
  the four-wavelength cell the longer-wave mode (n = 9, 8) is critical for
  both signs at ξ = 0.1 and 0.3 (0.5524, 0.3463, unchanged), and at ξ = 1
  the other sign is slightly lower (0.1950 vs 0.1954). So the graded values
  barely move — by luck, not by design. Recomputing all seven amplitudes
  for ξ < 0 on all three meshes (`rung3_phase.py`) and grading the minimum.

## 21:37–21:57 — rung 3 wrap-up

- `rung3_phase.py` finished. Other imperfection phase, four-wavelength
  cell: identical to four digits for ξ ≤ 0.75 (a longer-wave mode that both
  phases admit is critical), slightly lower at ξ = 1 (0.19497 vs 0.19536).
  Graded 3c values are now the minimum over both phases; largest gap to
  Koiter's bound 0.030 (ξ = 0.2), tolerance 0.05.
- One-wavelength cell, correct phase: 0.6806, 0.5714, 0.4452, 0.3669,
  0.2787, 0.2254, 0.1950 against Koiter 0.6837, 0.5763, 0.4502, 0.3737,
  0.2850, 0.2279, 0.1963, critical n = 13, 13, 13, 12, 11, 10, 9 against
  Koiter's 13.2, 13.0, 12.6, 12.0, 10.9, 9.6, 8.7. Always just below the
  bound.
- Caveat written into README: 3c is graded on the cell length fixed in
  config (4 wavelengths). Cells of 8 and 16 wavelengths give 0.535 at
  ξ = 0.1 and 0.337 / 0.320 at ξ = 0.3 — the gap to Koiter at ξ = 0.3 would
  be 0.054 with the 16-wavelength cell.
- Full 360° CalculiX model (6592 S8R, seed 1e-2 t), ~21–26 min per run:
  - ξ = 0.1: Newton fails at 0.5882 after the load has started to fall
    (own 0.5798, +1.4 %)
  - ξ = 0.3: knee at 0.3725 (own 0.3747, −0.6 %), load then rises slowly
  - ξ = 1.0: knee at 0.2046 (own 0.2053, −0.3 %), load then rises slowly
  - ξ = 0 launched 21:52 (`job-asm-instab-ccx-full0-11142152`).
  Two things changed relative to the planned model (full circumference and
  a larger seed), so the improvement cannot be attributed to one of them
  alone; the linear-stiffness tests above show the symmetry constraints
  were a defect regardless.
- Wrote `collect_metrics.py` → `metrics.json` (10 of 12 pass; 1d and 3d
  miss) and `make_review_page.py` → `docs/qa/trial-7-instability/`.

## 21:57–22:12 — last run and close-out

- Full-model CalculiX run for the perfect shell (seed 1e-2 t): load turns
  over at 0.8576, then Newton fails (increments down to 2e-6 of the step);
  own solver 0.8652 (−0.9 %). All four corrected CalculiX points are within
  1.5 % of the own solver.
- Final regrade: 10 of 12 pre-registered criteria pass; misses 1d and 3d
  stand. `metrics.json`, README and the review page regenerated from the
  same result files.
- Housekeeping: `results/ccx_work/` (CalculiX decks and output, ~140 MB) is
  excluded by `results/.gitignore`; removed the stale
  `results/r2_mode_16x24.npy` left by the first version of the rung-2
  script. `/tmp/instab_refs` (downloaded PDFs) is outside the repo and not
  kept.
- Not done: no post-bifurcation analysis in the own shell solver; no
  non-axisymmetric or measured imperfection; only one imperfection phase
  for the simply supported shell; the 0.007 vs 0.0042 discrepancy with
  Budiansky & Hutchinson's statement about Koiter's bound is unresolved; no
  browser was available to render the review page, so its layout was
  checked only by parsing the HTML and viewing the figures.
