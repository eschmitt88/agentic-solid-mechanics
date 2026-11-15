# Pre-registration — contact ladder (trial 10)

Rungs 1 and 2 written 2026-11-15 16:36 UTC (file time; the "≈17:10" I first
typed here was a guess without looking at the clock and was corrected at
17:09, nothing else in these two sections was touched), **before any contact
solver code existed or any deck was run**. The rung-3 section is appended later,
before rung 3 is run, with its own timestamp. Tolerances are not moved
afterwards; a miss is reported as a miss.

"own" = the JAX contact solver written for this trial (`contact2d.py`).
"ccx" = CalculiX 2.23. Every check that names both solvers is graded for
each solver separately (suffix `-own`, `-ccx`). For CalculiX the graded
configuration is fixed here: **surface-to-surface penalty contact
(`*CONTACT PAIR, TYPE=SURFACE TO SURFACE`, `PRESSURE-OVERCLOSURE=LINEAR`),
linear 4-node elements, on the same mesh the own solver uses at its finest
graded level**. The penalty stiffness is set by one rule fixed in advance:
K = 50·E/h_c (E = softer body's modulus, h_c = contact element size), and
stick slope = K/10 where friction is used. Every other CalculiX setting
tried (node-to-surface, mortar, quadratic elements, other stiffnesses) is
reported as characterisation, not graded.

## Reference formulas and where they were opened

- Hertz, sphere on flat: a³ = 3FR/(4E*), d = a²/R, F = (4/3)E*√R·d^1.5,
  p0 = 3F/(2πa²), p(r) = p0·√(1 − r²/a²), 1/E* = (1−ν1²)/E1 + (1−ν2²)/E2.
  Source: Wikipedia "Contact mechanics", fetched this session (text
  extraction lost radicals; formulas reconstructed and cross-checked for
  dimensional and mutual consistency: F and d formulas imply a² = Rd).
- Hertz, cylinder on half-space (2D): p(x) = E*·√(a² − x²)/(2R),
  P = πE*a²/(4R) per unit length (so p0 = E*a/(2R) = 2P/(πa)).
  Source: Barber, "Static friction" notes (Proc. R. Soc. A manuscript),
  eq. 4.2, https://public.websites.umich.edu/~jbarber/Static-friction.pdf
- Cattaneo–Mindlin, 2D, similar materials, P held then Q raised
  monotonically: q(x) = f·(E*/2R)·[√(a² − x²) − √(c² − x²)] (roots zero
  where the argument is negative), Q = f·πE*(a² − c²)/(4R), hence
  c/a = √(1 − Q/(fP)); the superposition is exact in 2D (Ciavarella,
  Jäger). Source: same Barber notes, eq. 2.1–2.2 and 4.5–4.8.
- Shallow ironing: Code_Aster validation document V6.03.153 (SSNP153),
  https://ericca.uqtr.ca/fr13.6/man_v/v6/v6.03.153.pdf — geometry
  (d1 = 0.2, d2 = 1.2, d3 = 10.6, r = 0.75, h1 = 0.95, h2 = 4, a1 = 0.3,
  a2 = 0.2), E = 68.96e8 / 6.896e8, ν = 0.32, μ = 0.3, indenter top moved
  down 1 over t∈[0,1] then sideways 10 over t∈[1,2], slab base clamped;
  its tabulated Code_Aster forces (non-regression values) and its figure 2
  comparing four teams (Code_Aster; Hartmann & Oliver; Fischer &
  Wriggers; Yastrebov). My read-offs from that figure (uncertainty
  ±0.05e8 on each):

  | quantity | Code_Aster | Hartmann–Oliver | Fischer–Wriggers | Yastrebov | band |
  |---|---|---|---|---|---|
  | vertical force, t = 1.0 | 3.30e8 (table: 3.299e8) | 3.2e8 | 2.95e8 | 2.9e8 | 2.9–3.3e8 |
  | vertical force, t = 1.5 | 4.18e8 | 4.08e8 | 3.65e8 | 3.60e8 | 3.60–4.18e8 |
  | tangential force, t = 1.0 (before sliding) | 0.69e8 (table: 0.690e8) | ≈0.63e8 | ≈0.60e8 | ≈0.60e8 | 0.60–0.69e8 |
  | tangential force, t = 1.5 (mean of the oscillation) | 1.45e8 | 1.55e8 | 1.95e8 | 1.72e8 | 1.45–1.95e8 |

  There is no exact answer for this problem: the document itself says the
  published results scatter. The graded statement is therefore "inside
  the published band", which is a weak test (the band is 14–30 % wide).
  The strong test on this problem is the cross-solver check 2f.

Not opened, not graded against: Johnson's *Contact Mechanics* (formula
numbers recalled from memory only), Yang–Laursen deep ironing (only seen
second-hand in arXiv 2302.13158 fig. 12–13, geometry ambiguous).

## Rung 1 — frictionless contact with exact answers

Fixed set-up: linear elasticity, small strain, ν = 0.3, E = 1000, R = 1.
1b: half-disc (plane strain) on a rigid flat, top face tied, load chosen so
that the Hertz half-width is a = 0.05 R. 1c: hemisphere (axisymmetric) on a
rigid flat, flat top face moved down by d = 0.0025 R (Hertz a = 0.05 R);
by symmetry this is a sphere squeezed between two rigid plates.

| id | quantity | reference | tolerance |
|----|----------|-----------|-----------|
| 1a | contact patch test: two stacked blocks with non-matching meshes, uniform pressure on top, frictionless interface. Max deviation of contact pressure and of σ_yy at all integration points from the applied pressure | exact uniform stress | 1 % (each of own, ccx) |
| 1b-a | 2D Hertz contact half-width, taken as the midpoint between the last contact point carrying pressure and the first one that does not (no curve fitting), finest mesh | a = √(4PR/(πE*)) | 5 %, and the contact element size must be ≤ a/15 so the resolution is inside the tolerance |
| 1b-p0 | 2D Hertz peak pressure | p0 = 2P/(πa) | 3 % |
| 1b-dist | 2D pressure distribution: relative L2 error over the contact, Σ(p_FE − p_H)²/Σp_H² at the contact points | Hertz ellipse with the theoretical a, p0 | √ of it ≤ 5 % |
| 1c-F | sphere: total force at the prescribed approach d | F = (4/3)E*√R·d^1.5 | 3 % |
| 1c-a | sphere: contact radius, same edge definition as 1b-a | a = √(Rd) | 5 % |
| 1c-p0 | sphere: peak pressure | p0 = 3F_H/(2πa²) | 3 % |
| 1d | mesh convergence of 1b-p0 and 1c-F over ≥ 3 levels (own solver): error at the finest level smaller than at the coarsest and inside tolerance | — | as stated |
| 1e | own vs ccx on identical mesh: peak pressure (2D) and force (sphere) | — | 2 % between solvers |

The pressure at the very edge of contact is not graded (slope is infinite
there); p0 and force are chosen because they are flat functions of the
discretisation.

## Rung 2 — friction

2a set-up: half-disc on an elastic block of the same material (Dundurs
β = 0, as the closed form requires), E* = E/(2(1−ν²)), f = 0.3, normal load
giving a = 0.05 R, then Q/(fP) raised 0 → 0.25 → 0.5 → 0.75 with P held.
Block 3R wide × 1.5R deep, base clamped; top face of the half-disc tied
(translates, cannot rotate).

| id | quantity | reference | tolerance |
|----|----------|-----------|-----------|
| 2a | stick-zone half-width c at Q/(fP) = 0.25, 0.5, 0.75; edge = midpoint between the last contact point with \|q\| < 0.99·f·p and the first with \|q\| ≥ 0.99·f·p, averaged over the two sides | c/a = √(1 − Q/(fP)) = 0.866, 0.707, 0.5 | \|c_FE − c\| ≤ 0.05·a at each of the three loads |
| 2b | tangential traction at the centre, q(0), at the three loads | q(0) = f·p0·(1 − c/a) | 5 % |
| 2c | tangential traction distribution, relative L2 error over the contact at Q/(fP) = 0.5 | closed form with theoretical a, c | ≤ 8 % |
| 2d | own vs ccx: q(0) at Q/(fP) = 0.5 | — | 5 % between solvers |
| 2e | shallow ironing, each solver: vertical and tangential force at t = 1.0 and at t = 1.5 (mean over t∈[1.4,1.6] to average the mesh-passing oscillation) inside the published band widened by the read-off uncertainty: vertical 2.85–3.35e8 and 3.55–4.23e8; tangential 0.55–0.74e8 and 1.40–2.00e8 | SSNP153 fig. 2 + table | in band (4 numbers per solver) |
| 2f | shallow ironing, own vs ccx with the same neo-Hookean energy (CalculiX form: C10(Ī1−3) + (J−1)²/D1) and comparable meshes: vertical and tangential force at t = 1.0 and mean over t∈[1.4,1.6] | — | 5 % between solvers on each of the 4 numbers |
| 2g | own solver, ironing: mesh refinement (≥ 2 levels) and load-step halving change the four 2e numbers by | — | ≤ 3 % |

Reported, not graded: CalculiX parameter sensitivity (contact type,
penalty stiffness, stick slope, element order), where it fails to
converge and what the failure looks like; the ratio tangential/vertical
during sliding; force oscillation amplitude; sensitivity of the ironing
forces to the neo-Hookean volumetric form.

If a solver cannot complete a problem, the checks that need it are
recorded as **miss (no result)**, not dropped.

## Rung 3 — blind prediction, Sandia Mechanics Challenge 2023

Appended 2026-11-15 17:27 UTC (from `date`), **before any rung-3 model code was written
or run**. At this point rungs 1–2 are partly run (see `log.md`); nothing
above this line was changed except the corrected timestamp.

**What I have.** Only the public information packet (SAND2023-06848, OSTI
2430623, in `raw/smc2023/`): scenario, drawings, the 19-test table (ΔV,
peak g, impact duration, weights), calibration results *as figures*. The
calibration data files, the measured acceleration histories and the
part-dimension tables are on Materials Data Facility behind Globus and
were not obtainable. So every calibration number is a read-off from a
figure (uncertainty stated in `config.yaml`), and the pulses are
reconstructed from the three tabulated numbers per test.

**What I have not seen.** The outcome. The post-blind report (OSTI
3013870) exists; I fetched its bytes once by accident and never opened
them (see `log.md` 16:29). I recall nothing about the results.

**Model class, fixed now.** A reduced-order dynamic model, not a 3D
explicit simulation (no explicit-dynamics code with element erosion is
available on this box and the 3D data are not either): Timoshenko beam for
the solid rod, rigid flange, six fasteners as coupled tension/shear
elastic-plastic springs with fracture (calibrated to the packet's fastener
tension and double-shear curves, intermediate rate as baseline), hole
clearance, preload from the stated torque, unilateral contact of the
flange face on a rigid block with Coulomb friction, base acceleration
pulse per test. Uncertainty by Monte Carlo over the read-off and modelling
parameters; reported value = median, band = 5th–95th percentile.

**Self-checks that must pass before the prediction is frozen** (these I
can grade myself; they test the model, not the outcome):

| id | check | tolerance |
|----|-------|-----------|
| 3a | first bending frequency of the rod clamped at the flange, reduced model vs CalculiX 3D solid `*FREQUENCY` | 5 % |
| 3b | static joint response in the elastic range (bolts linear, contact active, one load level below first yield): top-fastener force increase and tip deflection, reduced model vs CalculiX 3D solid model with contact and bolt springs | 15 % each. If it misses, the flange/bolt-seat stiffness of the reduced model may be recalibrated once against this CalculiX run, and that is reported |
| 3c | energy balance of the time integration at the highest-severity test: \|external work − (kinetic + elastic + plastic + frictional)\| / external work | ≤ 2 % |
| 3d | halving the time step changes the peak plastic elongation of fastener 1 in a test near first fracture by | ≤ 2 % |

**Frozen prediction** (`results/prediction.json`, written before any
outcome is opened; graded by the orchestrator, not by me):

- Q1: minimum ΔV (m/s) that fractures any fastener, with 5–95 % band.
- Q2: minimum ΔV (m/s) that fractures all six, with band.
- Q3: number and identity of fractured fasteners for each of the 19 tests,
  and the order in which they go.
- Q4/Q5: fracture in block/Keenserts and in the cantilevered part (yes/no).
  The reduced model cannot predict these; they will be judgment calls and
  labelled as such.
- Q6/Q7: relative displacement of points B and C with respect to A at
  T1–T10 for tests 2, 4, 7, 9, 11, 13, 16, in the packet's coordinate
  system (X right, Y down, Z into the block).

Suggested scoring, fixed now so it cannot be tuned later: Q1 and Q2 count
as hits if the experimental value lies inside my 5–95 % band, and the
point error is reported in m/s and in test steps (tests are ≈1.5 m/s
apart); Q3 as the number of the 19 tests whose fractured-fastener count I
get exactly, and within ±1; Q6/Q7 as relative error of the peak
|displacement| of point C per test. A band wider than 15 m/s on Q1 would
make a "hit" meaningless and will be flagged as such.

Known weaknesses stated in advance: figure read-offs instead of data;
reconstructed pulses; rigid block and rigid flange (the flange is 0.25 in
thick aluminium and may yield in bending); fastener fracture under
combined tension and shear is an assumed interaction, the packet has only
pure tension and double shear; threads/Keensert pull-out not modelled.
