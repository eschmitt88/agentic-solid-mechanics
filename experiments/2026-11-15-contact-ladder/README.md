---
kind: experiment
slug: "contact-ladder"
date: "2026-11-15"
status: running     # running | done | abandoned  (rungs 1-2 are done; the rung-3 blind prediction awaits grading)
hypothesis: "An LLM agent can solve contact and friction problems to physically correct answers without supervision, both by operating CalculiX and with a contact solver it writes itself, on a three-rung ladder: (1) frictionless contact with exact answers (patch test, Hertz), (2) friction (Cattaneo-Mindlin partial slip with a closed form; large-sliding 'ironing' with published curves), (3) a blind prediction of a real experiment (Sandia Mechanics Challenge 2023, bolted joint under shock) - graded against closed forms, published values, the other solver and, for rung 3, an outcome the agent has not seen, with tolerances fixed before each rung."
result: "Partly confirmed: 22 of 31 pre-registered checks pass; the blind prediction is frozen and ungraded. Rung 1 (15/16): both solvers reproduce Hertz to about 1 % and each other to 1.2 %; the sphere force is 2.6 % above Hertz in both, which is the finite sphere and not solver error; CalculiX's contact-pressure output is wrong next to the symmetry axis (the one miss). Rung 2 (4/11): the own solver reproduces the Cattaneo-Mindlin stick zone to 0.01 a, the centre traction to 0.3 % and the traction curve to 5.3 %, and its ironing forces are converged, but its sliding tangential force (1.31e8) is 7 % below the lowest published curve (miss); CalculiX fails the ironing problem in the pre-registered settings (stops 4 % into the slide) and did not finish the stick-slip problem on the pre-registered mesh, which accounts for 6 of the 7 misses; in settings that do run, its ironing force depends on the increment size (1.78e8 to 1.36e8) and extrapolates to the own solver's value (1.312e8 vs 1.313e8). Rung 3: the registered reduced-order model failed its own static check against a CalculiX 3D model (44 % too stiff, rigid flange) and was rebuilt with an elastic-plastic flange hinge before freezing; prediction: first fastener fracture at a velocity change of 23.7 m/s (5-95 %: 18.3-30.1), all six from 31.7 m/s (median)."
related_concepts: ["physics-grounded-evaluation"]
related_literature: []
tags: ["contact", "friction", "hertz", "cattaneo-mindlin", "ironing", "augmented-lagrangian", "mortar", "jax", "calculix", "blind-prediction", "sandia-mechanics-challenge", "bolted-joint", "trial-10"]
---

# contact-ladder

Trial 10 / phase 2, track C of `docs/2026-11-14-hard-problem-shortlist.md`:
contact and friction. Review page: `docs/qa/trial-10-contact/index.html`.

## Hypothesis

See frontmatter. Each problem of rungs 1–2 is solved twice — CalculiX 2.23
("ccx") and a JAX contact solver written for this trial ("own") — and each
is graded against the reference and against the other.

## Pre-registered pass criteria

In `PREREGISTRATION.md` (rungs 1–2 written 16:36 UTC before any solver
existed; rung 3 appended 17:27 UTC before any rung-3 code). Not moved.
A solver that cannot complete a problem in its registered configuration
counts as a miss.

## Sources opened this session

- Wikipedia, "Contact mechanics" (Hertz sphere: a³ = 3FR/4E*, d = a²/R,
  F = (4/3)E*√R d^1.5, p0 = 3F/2πa²) — https://en.wikipedia.org/wiki/Contact_mechanics
- Barber J.R., "Static friction" notes/manuscript (2D Hertz eq. 4.2;
  Cattaneo–Mindlin eq. 2.1–2.2, 4.5–4.8; Ciavarella/Jäger exactness in 2D) —
  https://public.websites.umich.edu/~jbarber/Static-friction.pdf
- Code_Aster validation document V6.03.153, "SSNP153 – Contact frottant
  déformable-déformable 2D en grandes déformations (shallow ironing)",
  geometry, loading, tabulated forces and the four-team comparison figure
  (originally from Yastrebov's thesis) — https://ericca.uqtr.ca/fr13.6/man_v/v6/v6.03.153.pdf
- Kramer S.L.B., Ivanoff T.A., Corona E. (2023) "Sandia Mechanics Challenge
  2023 Information Packet: Structure with a Threaded Fastener Joint",
  SAND2023-06848, OSTI 2430623 — https://www.osti.gov/servlets/purl/2430623
  (copy in `raw/smc2023/`; data DOI 10.18126/gmi9-62tm, not obtainable).
- Looked at, not graded against: arXiv 2302.13158 (deep ironing fig. 12–13),
  Wikipedia "Frictional contact mechanics".
- From memory only, never graded against: the sliding-contact asymmetry
  exponent for dissimilar materials (used only as a plausibility argument
  in the log), Johnson's *Contact Mechanics* equation numbers, T-stub
  formulas for bolted flanges (used to anticipate flange yielding, then
  replaced by a CalculiX computation).

## Setup

- Config: `config.yaml`. No data set, no training; Monte Carlo seed 20261115.
- Code (run with `uv run python <file>` in this folder):
  - `contact2d.py` — the own solver: Q4 plane-strain/axisymmetric solids
    (linear or neo-Hookean, energy only, forces and tangent by
    `jax.grad`/`jax.hessian`), contact with nodal multipliers on the slave
    surface and tractions integrated at fixed slave Gauss points,
    augmented Lagrangian (Uzawa) for the gap and for Coulomb stick/slip,
    contact tangent by `jax.jacfwd`, sparse solves with pypardiso.
  - `meshes.py`, `ccx.py` — gmsh CLI all-quad meshes, deck writing,
    CalculiX runs and `.frd`/`.dat` parsing.
  - `rung1_own.py`, `rung1_ccx.py` — patch test, 2D Hertz, sphere.
  - `rung2_mindlin_own.py`, `rung2_mindlin_ccx.py` — Cattaneo–Mindlin.
  - `rung2_ironing_own.py`, `rung2_ironing_ccx.py` — shallow ironing.
  - `smc_rom.py`, `smc_predict.py` — rung-3 reduced-order drop model,
    self-checks, Monte Carlo, prediction file; `smc_ccx_check.py`,
    `smc_ccx_pushover.py` — CalculiX 3D models used to check and correct it.
  - `collect_metrics.py`, `fig_rung1.py`, `fig_rung2.py`, `fig_rung3.py`,
    `make_review_page.py`.
- Decks, meshes and raw solver output: `results/ccx_work/`, `results/work/`
  (git-ignored, ≈1.1 GB).
- Extra packages: none.

## Result

Numbers in `metrics.json` (`checks`, `ironing_supplementary`,
`calculix_characterisation`); frozen prediction in `results/prediction.json`.

### Rung 1 — frictionless contact (15 of 16)

| id | computed | reference | error | tol. | verdict |
|----|----------|-----------|-------|------|---------|
| 1a-own | patch test, max stress deviation 0.53 % (6 integration points per edge) | 0 | 0.53 % | 1 % | pass |
| 1a-ccx | 0.02 % (surface-to-surface) | 0 | 0.02 % | 1 % | pass |
| 1b-a | half-width own 0.04946, ccx 0.05024 | 0.05 | −1.1 %, +0.5 % | 5 % | pass, pass |
| 1b-p0 | peak pressure own 27.441, ccx 27.778 | 27.473 | −0.12 %, +1.1 % | 3 % | pass, pass |
| 1b-dist | pressure curve L2 own 1.4 %, ccx 1.2 % | Hertz | — | 5 % | pass, pass |
| 1c-F | sphere force own 0.18794, ccx 0.18789 | 0.18315 | +2.62 %, +2.59 % | 3 % | pass, pass (finite-size effect, see below) |
| 1c-a | contact radius 0.05024 (both) | 0.05 | +0.5 % | 5 % | pass, pass |
| 1c-p0 | peak pressure own 35.12; ccx 54.6 (mean within 0.1 a), 171 max | 34.98 | +0.4 %; +56 % | 3 % | pass; **miss** |
| 1d | own errors coarsest → finest: p0 0.26 → 0.12 %, sphere force 2.68 → 2.62 % | — | — | monotone, in tol. | pass |
| 1e | own vs ccx: 2D peak pressure 1.2 %, sphere force 0.03 % | — | — | 2 % | pass |

- The sphere force excess is physical: 0.57, 1.20, 2.64, 5.69 % at
  a/R = 0.0125, 0.025, 0.05, 0.1 (`results/r1_sphere_scaling_own.json`),
  i.e. proportional to a/R. I prescribe the approach of the equatorial
  plane of a finite sphere; Hertz's approach is that of points at infinity.
  The 3 % tolerance was set assuming an O((a/R)²) effect.
- 1c-p0-ccx: CalculiX's nodal contact pressure in axisymmetric models is
  0, 171, 7.6, 42.6 on the first four nodes from the axis (Hertz 35.0) and
  within 1.5 % further out; the total force is right.
- Own patch test is quadrature-limited: 7.5, 3.2, 1.5, 0.53, 0.26, 0.03 %
  for 2, 3, 4, 6, 10, 20 points per slave edge; 1e-14 on matching meshes.

### Rung 2 — friction (4 of 11)

| id | computed | reference | error | tol. | verdict |
|----|----------|-----------|-------|------|---------|
| 2a-own | stick half-width c/a = 0.857, 0.701, 0.499 at Q/fP = 0.25, 0.5, 0.75 (h = a/64) | 0.866, 0.707, 0.500 | −0.009, −0.006, −0.001 a | 0.05 a | pass |
| 2b-own | centre traction | closed form | +0.26, +0.20, +0.18 % | 5 % | pass |
| 2c-own | traction curve L2 at Q/fP = 0.5: 5.3 % (13.4 %, 7.9 % on the two coarser meshes) | closed form | — | 8 % | pass |
| 2a/2b/2c-ccx | no result on the registered mesh (61 min, 8 increments, 62 divergence messages, stopped) | — | — | — | **miss ×3** |
| 2d | own vs ccx centre traction: no ccx result | — | — | 5 % | **miss** |
| 2e-own | ironing, h = 0.05: vertical 3.31e8 (t = 1), 4.21e8 (mid-slide); tangential 0.711e8, **1.307e8** | bands 2.85–3.35, 3.55–4.23, 0.55–0.74, 1.40–2.00 e8 | 3 of 4 in band | in band | **miss** |
| 2e-ccx | registered configuration (h = 0.05, K = 50E/h, increments 0.005) fails at t = 1.04 | — | — | in band | **miss** |
| 2f | own vs ccx ironing: no ccx result in the registered configuration | — | — | 5 % | **miss** |
| 2g | own, ironing: step halving 1.3 %, mesh 0.1 → 0.05 0.6 % (mesh 0.2 → 0.1: 4.4 %) | — | — | 3 % | pass |

Supplementary, not pre-registered:

- CalculiX Cattaneo–Mindlin one mesh level coarser (h = a/32), registered
  contact settings: stick zone within 0.02 a, centre traction +2.0 %,
  traction curve L2 19 % (fails the 8 % mark); friction traction after
  normal loading alone up to 0.53 f·p0 where theory has none (own: 0.03).
  Own vs ccx centre traction on that mesh: +1.7 %. Settings on that mesh
  (`metrics.json:calculix_characterisation.mindlin`): stick slope ×10 —
  not finished after 45 min (403 divergence messages, stopped); stick
  slope ÷10 — finishes, stick zone 0.05 a too wide, L2 11 %; penalty ÷10 —
  finishes fastest, centre +2.8 %, L2 10.5 %. No setting tried gets the
  traction curve inside 8 %.
- CalculiX ironing on the registered mesh (h = 0.05) with 0.001 increments
  got past the point where the 0.005 run fails and was stopped by me at
  t = 1.16 (25 min, 161 of 1000 sliding increments); up to there its
  tangential force is within 1–4 % of the own solver on the same mesh
  (`results/r2_ironing_ccx_iron_SURFACETOSURFACE_h0.05_dm0.001_partial.json`).
- CalculiX ironing by setting (`results/r2_ccx_settings.png`): fails at
  t ≈ 1.04–1.05 with K = 50E/h and increments 0.005 on h = 0.1 and 0.05,
  and with K = 500E/h on h = 0.2; node-to-surface fails at t = 0.70;
  completes with K = 5E/h (tangential 1.62e8) and with K = 50E/h and
  increments 0.001 (1.398e8) or 0.0005 (1.355e8). The first coarse run
  (h = 0.2, K = 50E/h) completed with 90 cutbacks and 1.78e8.
- Linear extrapolation of the two fine-increment CalculiX runs to zero
  increment: 1.312e8; own solver on the same mesh: 1.313e8. With increments
  0.0005 CalculiX differs from own by +0.2 %, +0.1 % (vertical), −4.2 %,
  +3.2 % (tangential) on the four graded numbers — inside the 5 % of 2f,
  but in a configuration chosen after the registered one failed.
- Material law: with the textbook coupled neo-Hookean instead of the
  CalculiX form the own solver gives a sliding tangential force of 1.387e8
  at h = 0.1 (+5.6 %; +17 % on the coarse mesh) and tangential/vertical
  0.339 (Code_Aster: 1.45e8, 0.347).

### Rung 3 — blind prediction, Sandia Mechanics Challenge 2023

Obtained: the information packet PDF only. Not obtained: calibration data
files, measured pulses, part dimensions (Materials Data Facility, Globus
login). Not opened: the post-blind report OSTI 3013870.

Self-checks: 3a pass (1144.5 vs 1195.5 Hz, −4.3 %); **3b miss as
registered** (rigid flange: tip deflection −44 %, top-fastener force −22.5 %
against CalculiX 3D), model rebuilt with a flange hinge fitted to that run
(+0.2 %, +1.3 %) and an elastic-plastic hinge law from a CalculiX 3D
pushover; 3c pass (energy balance 0.03 %); 3d pass (0.06 %).

Frozen 18:31:32 UTC (`results/prediction.json`, sha256 66a14b1b…8f3097):

| quantity | nominal | median | 5–95 % |
|----------|---------|--------|--------|
| Q1, lowest ΔV that fractures a fastener | 25.2 m/s (test 9, 82.6 ft/s) | 23.7 m/s (77.9 ft/s) | 18.3–30.1 m/s |
| Q2, lowest ΔV that fractures all six | 30.1 m/s (test 13) | 31.7 m/s | 23.7 m/s – not reached (fastener 4 survives every test in 40 % of samples) |
| Q3, fractured per test (nominal) | tests 1–8: 0; 9: 3; 10–12: 5; 13–18: 6; 19: 5 | | |
| order | 1, then 2 and 6, then 3 and 5, then 4; tension-dominated; within the same pulse | | |
| Q4 block/Keensert fracture | no (judgment) | | |
| Q5 cantilever fracture | no (judgment); flange bent permanently in every test | | |

Q6/Q7 (relative displacements of points B and C at T1–T10 for seven tests)
are in the prediction file.

Plots: `results/r1_hertz_pressure.png`, `r1_convergence.png`, `r1_patch.png`,
`r2_mindlin.png`, `r2_ironing_mesh.png`, `r2_ironing_forces.png`,
`r2_ccx_settings.png`, `r3_joint.png`, `r3_prediction.png`, `r3_histories.png`.

## Interpretation

- Frictionless contact is in hand for both solvers. The cross-check was
  worth having mainly for what it showed about my tolerance (the sphere)
  and CalculiX's output (the axis).
- Friction is where the two solvers separate, and the separation is in
  robustness, not in the converged answer. My solver needed six repairs
  before it ran at all (two of them wrong derivatives), then gave
  step- and mesh-converged answers without tuning. CalculiX ran the first
  time on coarse meshes, returned a plausible number inside the published
  band, and that number was 30 % off its own small-increment limit.
- "Inside the published band" was a poor check: the band mixes material
  laws and is wider than the effect in question. The check I should have
  registered is the one I ended up doing after the fact: both solvers,
  same law, increment and mesh refinement on both.
- Rung 3: the pre-registered self-check did its job. It exposed that the
  flange cannot be rigid, and the corrected model predicts fracture about
  25 % later than the first one would have. Whether the corrected model
  is right is for the grading to say.

## Diagnostics

- intended_effect_confirmed: partial — 22 of 31 checks (`metrics.json:summary`); 6 of the 9 misses are CalculiX configurations that did not produce a result (`metrics.json:checks` ids 2a-ccx, 2b-ccx, 2c-ccx, 2d, 2e-ccx, 2f).
- leakage_check: outcome hygiene for rung 3 — the post-blind report was never opened (`log.md` 16:29 and 18:31 entries); prediction hash recorded before grading (`results/prediction.json:frozen_utc`).
- overfitting_signal: n/a (no training). Analogue: the rung-3 model was changed after a self-check; all three stages are recorded (`log.md` 17:27–18:32: rigid flange, nominal first fracture at test 6 = 20.2 m/s in runs that still had the pulse-exponent bug; elastic hinge, median 19.4 m/s; plastic hinge = frozen, median 23.7 m/s). No outcome was seen at any stage.
- delta_from_prior: vs instability-path-following (trial 7), same pattern — a converged-looking second-solver answer was wrong (there CalculiX stepping over a bifurcation, here CalculiX's increment-size dependence in sliding friction, `metrics.json:ironing_supplementary`).
- unexpected_findings: CalculiX's large-sliding friction force is first-order in the increment size and extrapolates to the own solver's value (`metrics.json:ironing_supplementary.ccx_Ft_mid_linear_extrapolation_to_zero_increment`); the 0.25 in flange yields at a quarter of the mildest test's load (`results/r3_flange_hinge.json`).
- next_candidates:
  - Re-run the ironing comparison as a registered check: both solvers with the same law, three increment sizes and two meshes each, graded on the extrapolated values.
  - Replace the fixed-quadrature contact integration of `contact2d.py` by true segment-to-segment (mortar) integration so that the patch test is exact, and re-grade 1a and 2c.
  - After the orchestrator grades rung 3: rerun the drop model with the measured pulses and data files (needs Globus) to separate read-off error from model error.

## Follow-up

- The h = 0.05 CalculiX ironing run with 0.001 increments was stopped at
  t = 1.16 of 2; at about 2.5 h it is affordable and would give the
  registered-mesh comparison that 2f was meant to be.
- A 3D explicit model with element erosion would be the proper tool for
  rung 3; none is available on this box.
