---
kind: experiment
slug: "instability-path-following"
date: "2026-11-14"
status: done
hypothesis: "An LLM agent can build and operate, without supervision, the path-following machinery that structural instability needs (arc-length continuation, bifurcation detection from the tangent stiffness, branch switching) and reach physically correct answers on a three-rung ladder: (1) snap-through/snap-back of trusses and geometrically exact beams, (2) the Biot surface instability of a compressed neo-Hookean half-space, (3) the imperfection sensitivity of an axially compressed thin cylindrical shell — each graded against a closed form, a published reference value or an independent solver, with tolerances fixed before the run."
result: "Partly confirmed: 10 of 12 pre-registered checks pass. Rung 1: own arc-length solver reproduces truss closed forms to round-off, the Lee-frame limit load (18.550 vs 18.552) and the deep-arch maximum (8.9741 vs 8.9729), with CalculiX agreeing to 0.15 %; one badly posed check (load at a near-vertical part of the Lee curve) missed. Rung 2: Biot threshold 0.4563110 vs 0.4563110 with exactly incompressible elements, after a first attempt with Taylor-Hood elements produced a mesh-converged wrong value (0.322). Rung 3: classical cylinder load to 6e-6, knockdown curve within 0.030 of Koiter's upper bound at 7 amplitudes; the planned CalculiX cross-check missed at 2 of 4 amplitudes (Newton stepping over the bifurcation; rotational symmetry constraints stiffening the quarter model) and agreed to 1.5 % once rerun on a full model."
related_concepts: ["physics-grounded-evaluation"]
related_literature: []
tags: ["jax", "arc-length", "continuation", "bifurcation", "buckling", "snap-through", "biot", "shell-buckling", "imperfection-sensitivity", "calculix", "trial-7"]
---

# instability-path-following

Trial 7 / phase 2, track A of `docs/2026-11-14-hard-problem-shortlist.md`:
structural instability and equilibrium-path following, the tier FEM-Bench
names as its weakest.

## Hypothesis

See frontmatter. Concretely: one generic arc-length continuation solver
(JAX, float64, tangent stiffness by automatic differentiation) passes limit
points, detects bifurcations from the inertia of the tangent stiffness and
switches branches; the same tangent-based detection finds a continuum
bifurcation threshold and a shell knockdown curve to within the tolerances
below.

## Pre-registered pass criteria

Written 2026-11-14 ≈20:40 UTC, **before any solver for the respective rung was
run**. Tolerances are not moved afterwards; a miss is reported as a miss.
Reference values come only from sources opened in this session (list below).

### Rung 1 — path following

| id | quantity | reference | tolerance |
|----|----------|-----------|-----------|
| 1a | von Mises truss, shallow (1 DOF path): limit load | closed form of the same discrete model (Green–Lagrange bar), derived in-session and checked symbolically | 0.1 % |
| 1b | von Mises truss, steep (2 DOF): bifurcation load on the primary path, located from the tangent stiffness; branch switch must land on the asymmetric branch | closed form, same derivation | 0.1 % |
| 1c | Lee frame: first limit load P·L²/EI | 18.552 at u_y/L = 0.407 (NAFEMS NL7 as quoted in the Abaqus Benchmarks Guide) | 1 % on load |
| 1d | Lee frame: load at u_y/L = 0.784 on the re-stiffened branch, P·L²/EI | 31.887 (same source; my reading of the table is that this is a path point after the snap-back — stated before running) | 2 % |
| 1e | 215° clamped–hinged deep circular arch: maximum load P·R²/EI | 8.972922 (Horák et al. 2022, converged inextensible, shear-rigid value; DaDeppo & Schmidt 8.97) | 1 % |
| 1f | independent solver: CalculiX (displacement control, NLGEOM beams/trusses) limit load of the Lee frame vs my solver | — | 2 % between the two solvers |

Also required (qualitative, reported not graded): load-controlled Newton on
the same problems, showing what it does at the limit point.

### Rung 2 — Biot surface instability

| id | quantity | reference | tolerance |
|----|----------|-----------|-----------|
| 2a | critical compressive nominal strain of a plane-strain incompressible neo-Hookean half-space, detected as the zero crossing of the lowest eigenvalue of the (constrained) tangent stiffness, on the finest mesh | ε_W = 1 − 0.5437 = 0.4563 (λ1/λ2 = 0.2956, root of r³ − 3r² − r − 1 = 0, r = 3.3830; Cao & Hutchinson 2012 eq. 1.1, 2.8) | 1 % of ε_W (±0.0046 strain) |
| 2b | mesh convergence: error decreases monotonically under uniform refinement over ≥ 3 levels | — | monotone |

### Rung 3 — cylinder imperfection sensitivity

Chosen geometry is fixed in `config.yaml` before running. Imperfection for the
graded comparison is Koiter's: axisymmetric, in the shape of the classical
axisymmetric buckling mode, amplitude ξ·t.

| id | quantity | reference | tolerance |
|----|----------|-----------|-----------|
| 3a | classical buckling load by linear eigenvalue analysis (own solver, converged mesh) | σ_cl = E t / (R √(3(1−ν²))) (Budiansky & Hutchinson 1972 eq. 1; SP-8007 eq. 4 with γ = 1) | 3 % |
| 3b | same by CalculiX `*BUCKLE` (S8R), converged mesh | same | 5 % |
| 3c | knockdown factor ρ(ξ) = bifurcation/collapse load ÷ classical load, at ξ ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}, computed on the nonlinear axisymmetric pre-buckling path with boundary conditions that reproduce Koiter's infinitely long shell | Koiter's (1963) upper-bound formula as written out by Hutchinson 2010 eq. 13–15 (P2 = 0 minimised over the circumferential wavenumber γ) | |ρ − ρ_Koiter| ≤ 0.05 at every ξ |
| 3d | independent solver: CalculiX geometrically nonlinear collapse load vs my solver for the same finite shell and imperfection, at ≥ 3 amplitudes | — | 10 % relative |

Not graded, reported: comparison with the NASA SP-8007 lower-bound knockdown
γ = 1 − 0.901(1 − e^{−φ}), φ = (1/16)√(R/t) (the imperfection amplitude at
which the computed curve meets it), the Budiansky–Hutchinson long-wave limit
curve (their eq. 8), realistic simply supported ends, and the simple
asymptotic form (1 − ρ)² ≈ (3c/2)·ξ·ρ that I remembered — the latter is from
memory and is only shown as the small-ξ limit of the opened-source formula.

## Sources opened this session

- Cao Y., Hutchinson J.W. (2012) "From wrinkles to creases in elastomers",
  Proc. R. Soc. A 468:94–115, doi:10.1098/rspa.2011.0384 —
  https://groups.seas.harvard.edu/hutchinson/papers/Wrinkleinstability2011F.pdf
  (Biot threshold, eq. 1.1, 2.8; creasing strain 0.35).
- Abaqus Benchmarks Guide, "NL7: Lee's frame buckling problem" (quotes NAFEMS
  NNB rev. 1, 1989) — https://abaqus.uclouvain.be/English/SIMACAEBMKRefMap/simabmk-c-nl7.htm
- Horák M., La Malfa Ribolla E., Jirásek M. (2022) "Efficient formulation of a
  two-noded geometrically exact curved beam element", arXiv:2111.11154v3,
  §4.4 and Table 5 (deep arch 8.972922; DaDeppo & Schmidt 8.97; Simo &
  Vu-Quoc 9.0528; Ibrahimbegović 8.973).
- Kadapa C. (2020) "A simple extrapolated predictor for overcoming the
  starting and tracking issues in the arc-length method", arXiv:2005.10192,
  §3.3 (Lee frame data A = 6, I = 2, E = 720, geometry 120/24/96) and §3.4.
- NASA SP-8007 "Buckling of thin-walled circular cylinders" (1965, rev. Aug
  1968), NTRS 19690013955, p. 5 eq. 4–6 —
  https://ntrs.nasa.gov/api/citations/19690013955/downloads/19690013955.pdf
- Hutchinson J.W. (2010) "Knockdown factors for buckling of cylindrical and
  spherical shells subject to reduced biaxial membrane stress", IJSS
  47:1443–1448, doi:10.1016/j.ijsolstr.2010.02.009 —
  https://groups.seas.harvard.edu/hutchinson/papers/IJSS-Buckling-2010.pdf
  (eq. 3–5, 13–15: Koiter upper bound).
- Budiansky B., Hutchinson J.W. (1972) "Buckling of circular cylindrical
  shells under axial compression", in Contributions to the Theory of Aircraft
  Structures, pp. 239–260 —
  https://groups.seas.harvard.edu/hutchinson/papers/324.pdf (eq. 1–8, fig. 1).

Opened but not used for grading: Hutchinson 1965 (NASA CR, NTRS 19650004259;
scan too garbled to read the equations), Gerasimidis, Virot, Hutchinson &
Rubinstein 2018 (J. Appl. Mech., "On establishing buckling knockdowns…"),
Hutchinson, Tennyson & Muggeridge 1971 (AIAA J. 9:48), Destrade et al. on
Biot-type instabilities.

## Setup

- Config: `config.yaml`. No randomness; no data set; no train/validation
  split — every number is graded against physics.
- Code (all in this folder; run with `JAX_PLATFORMS=cpu uv run python <file>`):
  - `pathfollow.py` — arc-length continuation: bordered pseudo-arclength
    corrector, inertia count of the tangent stiffness as stability monitor,
    bisection on the step to locate limit points and bifurcations, branch
    switching along the null vector, plus a plain load-controlled Newton for
    comparison.
  - `rung1_truss.py`, `rung1_beams.py`, `rung1_ccx_lee.py` — von Mises truss
    (closed form checked with sympy), Simo–Reissner beam elements for the Lee
    frame and the 215° arch, CalculiX B32R cross-check.
  - `rung2_biot.py` — compressed neo-Hookean block; attempt 1 (Taylor–Hood
    Q2/Q1) and attempt 2 (stream function, C¹ Hermite); finite-depth
    dispersion relation as semi-analytic check.
  - `rung3_cylinder.py`, `rung3_phase.py` — own Donnell-shell solver
    (Hermite elements along the axis, Fourier harmonics around), linear
    eigenvalue analysis and bifurcation from the nonlinear axisymmetric path;
    Koiter / Budiansky–Hutchinson / SP-8007 formulas.
  - `rung3_ccx.py`, `rung3_ccx_axisym_check.py` — CalculiX S8R shell models
    (quarter and full), `*BUCKLE` and NLGEOM runs; axisymmetric continuum
    (CAX8) third opinion.
  - `rung3_figures.py`, `collect_metrics.py`, `make_review_page.py` —
    grading, `metrics.json`, review page.
- CalculiX runs: `rung3_ccx.py buckle 4 6 8 12 16`;
  `rung3_ccx.py collapse <xi> <elements per wavelength> <range factor> <seed/t> <tag> <increment>`
  (planned quarter-model runs: seed 1e-3, increment 0.01); the corrected
  runs add `CYL_FULL=1` (full circumference) with seed 1e-2;
  `CYL_NO_ROT_BC=1` is the diagnostic switch that drops the rotational
  symmetry constraints. Decks and raw output go to `results/ccx_work/`
  (git-ignored, ~140 MB).
- In every solver only the strain energy is hand-written; internal forces
  and tangent stiffness are `jax.grad` / `jax.hessian` of it (float64).
- Extra package: `sympy` (installed with `uv pip install`, not added to
  `pyproject.toml`).

## Result

All numbers below are in `metrics.json`; per-rung detail in
`results/r1_*.json`, `results/r2_biot.json`, `results/r3_*.json`; review page
`docs/qa/trial-7-instability/index.html`.

| id | computed | reference | error | tolerance | verdict |
|----|----------|-----------|-------|-----------|---------|
| 1a | truss limit load 0.009132131 | 0.009132131 (closed form) | < 1e-15 | 0.1 % | pass |
| 1b | truss bifurcation load 0.211211446; switched branch on the closed-form circle to 8e-14 | 0.211211446 | 4e-14 | 0.1 % | pass |
| 1c | Lee first limit load 18.5504 (160 el.; Richardson 18.5466, order 2.00) | 18.552 | 0.009 % | 1 % | pass |
| 1d | Lee load at u/L = 0.784: 39.43 | 31.887 | 23.6 % | 2 % | **miss** |
| 1e | arch maximum 8.97410 (320 el.; Richardson 8.97284) | 8.972922 | 0.013 % | 1 % | pass |
| 1f | CalculiX Lee limit load 18.518 | own 18.547 | 0.15 % | 2 % | pass |
| 2a | Biot strain 0.4563110229 (64 el./wavelength) | 0.4563109873 | 7.8e-8 | 1 % | pass (second discretisation) |
| 2b | errors 6.8e-2, 4.6e-3, 3.1e-4, 2.0e-5, 1.2e-6, 7.8e-8 | — | order 4.0 | monotone | pass |
| 3a | own linear eigenvalue 1.0000056 × classical | 1 | 5.6e-6 | 3 % | pass |
| 3b | CalculiX `*BUCKLE` 0.9883 × classical (6528 S8R) | 1 | 1.2 % | 5 % | pass |
| 3c | ρ = 0.664, 0.552, 0.420, 0.346, 0.265, 0.218, 0.195 at ξ = 0.05…1.0 | Koiter bound 0.684, 0.576, 0.450, 0.374, 0.285, 0.228, 0.196 | max 0.030 | 0.05 | pass |
| 3d | CalculiX as planned: 0.930, 0.645, 0.373, none at ξ = 0, 0.1, 0.3, 1.0 | own 0.865, 0.580, 0.375, 0.205 | +7.5 %, +11.2 %, −0.6 %, no buckling found | 10 % | **miss** |

Plots: `results/r1_truss.png`, `r1_lee_frame.png`, `r1_arch.png`,
`r1_beam_convergence.png`, `r2_biot.png`, `r2_biot_mode.png`,
`r3_knockdown.png`, `r3_ccx_paths.png`.

Supplementary, not pre-registered:

- 1d read the well-conditioned way: deflection at the reference load 31.887
  is 0.7809 L vs 0.784 L (−0.40 %); Abaqus' own value on the reference page
  is 0.781.
- Load-controlled Newton on the rung-1 problems: fails to converge just
  above each limit load, or converges to a remote branch (truss: 29
  iterations, lands on the inverted configuration; Lee frame: no convergence
  at P·L²/EI = 19–21, then the taut far branch at 22; arch: nothing above
  8.5).
- Rung 2, attempt 1 (Taylor–Hood): detected threshold 0.4586 on 4×6, then
  0.3221, 0.3217, 0.3216 on 8×12…32×48 with shift-invert near zero; a full
  inertia count puts the first loss of stability at strain 0.02–0.03 on
  every mesh (spurious element-scale surface mode).
- Rung 3, one-wavelength cell (admits only the axial period of Koiter's
  trial mode): 0.6806, 0.5714, 0.4452, 0.3669, 0.2787, 0.2254, 0.1950 —
  0.001–0.007 below Koiter's bound at all seven amplitudes, with his wave
  numbers (n = 13…9).
- Rung 3, simply supported ends (own solver): 0.865 (perfect shell,
  nonlinear pre-buckling), 0.694, 0.580, 0.449, 0.375, 0.288, 0.233, 0.205.
- Rung 3, CalculiX after correcting the model (full 360°, 6592 S8R, seed
  0.01 t; `metrics.json:3d.corrected_full_model`): 0.858, 0.588, 0.372,
  0.205 at ξ = 0, 0.1, 0.3, 1.0 against 0.865, 0.580, 0.375, 0.205 from the
  own solver (−0.9 %, +1.4 %, −0.6 %, −0.3 %). Two things changed relative
  to the planned model (full circumference, larger seed); at ξ = 0 and 0.1
  the run ends in Newton failure after the load turns over, at ξ = 0.3 and
  1.0 the shell buckles and keeps carrying load.
- Axisymmetric pre-buckling stiffness, third opinion with continuum CAX8
  elements: 0.9871 / 0.8445 / 0.3026 of the perfect shell at ξ = 0.1 / 0.3 /
  1.0 against 0.9870 / 0.8435 / 0.3009 from the own solver
  (`results/r3_axisym_stiffness_check.json`).
- NASA SP-8007 factor for R/t = 200 is 0.471; the computed curves reach it at
  ξ ≈ 0.16 (infinite shell) and 0.18 (simply supported). This is orientation,
  not validation: the NASA factor is a lower bound to tests on shells with
  irregular imperfections.

## Interpretation

- The path-following machinery itself was the easy part. A generic
  continuation solver with autodiff tangents took one sitting and hit every
  well-posed reference to 4–5 digits.
- Every serious error in this trial was of one kind: **a number that looked
  right and was not**, caught only because an independent check existed.
  - a branch "switch" that silently stayed on the primary path (caught by a
    closed form for the secondary branch);
  - a Biot threshold "converged" to 0.322 on three meshes (caught by the
    analytic value; the tell-tale was the mode number scaling with the mesh);
  - an eigen-solver that returns eigenvalues nearest a shift, hiding modes
    that were already strongly negative (caught by a dense inertia count);
  - CalculiX staying on an unstable solution 11 % beyond the bifurcation
    (caught by overlaying my axisymmetric path);
  - a CalculiX symmetry model 7 % too stiff (caught by a continuum model);
  - my own infinite-shell cell excluding Koiter's mode for one imperfection
    phase (caught by a one-wavelength cell exceeding an upper bound).
- For shell imperfection sensitivity, plain Newton in CalculiX is not a
  measuring instrument: its "collapse load" moved by 12 % with the size of a
  non-physical seed. A stability test on the tangent stiffness along the
  path is.
- Bifurcation is not collapse: at ξ = 0.3 and 1.0 the shell buckles at the
  predicted load and then carries a slowly rising load under prescribed
  shortening (as Budiansky & Hutchinson 1972 conclude for larger
  imperfections); at ξ = 0.1 it snaps.

## Diagnostics

Unless noted, numbers reference `metrics.json`. There is no validation/test
split in this experiment (HCE rule not applicable); `metrics.json` holds
physics-graded numbers only.

- intended_effect_confirmed: partial — 10 of 12 pre-registered criteria pass
  (`metrics.json:summary`); the two misses are 1d (ill-posed criterion) and
  3d (second-solver model defects), both kept as misses.
- leakage_check: reference values were taken only from sources opened this
  session and written into this README before the corresponding solver ran
  (`log.md` 20:40, `config.yaml:rung3` 20:57) — no reference value was used
  to tune a solver; the one post-hoc change of method (rung 2, second
  discretisation) is declared and the first attempt's result is kept.
- overfitting_signal: n/a (no fitted parameters). Closest analogue: 3c
  passes with the cell length fixed in `config.yaml` (4 wavelengths, max gap
  0.030); a 16-wavelength cell lowers ρ at ξ = 0.3 to 0.320, a gap of 0.054
  (`results/r3_cylinder.json:knockdown_long_cell`) — the deviation is the
  long-wave mode family Budiansky & Hutchinson describe, but the pass is
  sensitive to that choice.
- delta_from_prior: first instability trial; no prior experiment to compare.
- unexpected_findings: Taylor–Hood elements lose stability at 3 % strain on
  a homogeneously compressed neo-Hookean block and, combined with a
  shift-invert eigen-solve, yield a mesh-independent false threshold of 0.322
  (`results/r2_biot.json:taylor_hood`). CalculiX rotational SPCs on shell
  symmetry edges stiffen a corrugated shell by 7 %
  (`log.md` 21:12–21:30).
- next_candidates:
  - Add a coupled-harmonic (0, n, 2n) post-bifurcation model to the own shell
    solver so collapse versus stable buckling is decided without CalculiX.
  - Replace the ring ripple by a measured or random imperfection field and
    compare the resulting knockdown with a published test series.
  - Reuse `pathfollow.py` for track D (inverse design through a snap-through)
    and check gradient quality near the limit point.
  - Repeat rung 2 with Q2/P1-discontinuous and F-bar elements to map which
    mixed elements show the spurious surface mode.

## Follow-up

- 1d should be re-posed as "deflection at the reference load" in any re-run.
- The discrepancy between my one-wavelength-cell result and Koiter's bound
  (up to 0.007) is slightly larger than the 0.0042 Budiansky & Hutchinson
  quote for that mode class; not resolved.
- Only one imperfection phase (ends on crests) was run for the simply
  supported shell and CalculiX.
- `docs/2026-11-14-hard-problem-shortlist.md` quotes the Biot strain as
  0.444; the plane-strain value is 0.4563 (Cao & Hutchinson 2012). The
  shortlist was not edited (outside this track's write scope).
