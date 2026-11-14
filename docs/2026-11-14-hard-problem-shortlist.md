---
title: Hard-problem shortlist for phase 2
date: 2026-11-14
status: proposed — awaiting pick
---

# Hard-problem shortlist for phase 2

Phase 1 (trials 1–6) covered linear elasticity, sizing loops, SIMP topology
optimization, viscohyperelastic calibration and a convection solver. This
note collects candidates for genuinely hard problems and proposes which to
deep dive on.

**Evidence level.** Built from three web sweeps by research subagents on
2026-11-14. Most citations are search-snippet or recall level, not read in
full. Anything marked (unverified) must be checked before an experiment
depends on it. Dataset availability in particular needs a download test.

## Where the agent frontier is

- No benchmark found tests an agent on contact, plasticity, fracture or
  buckling with a physical-correctness grade. Existing suites are linear
  elasticity, beams/trusses, CFD, or "does the code run".
- "Runs" and "right" are far apart. FEABench (arXiv 2504.06260): best
  agents solve 5 of 15 when re-run in arXiv 2609.03718. CFDLLMBench
  (2509.20374): 14–34 %. Execution vs field scoring differ by about 20
  points on FoamBench (2609.03718).
- Generic coding agents now match specialised multi-agent systems on the
  easy suites (FoamBench 96.4 %); repair budget and supplied solver
  tutorials are what matter (2609.03718).
- FEM-Bench (2512.20732) names buckling and geometric nonlinearity as its
  weakest tier.
- LLM-driven topology optimization preprints (TO-Master 2607.01812,
  TopOptAgents 2605.23273) stay at compliance and aggregated stress. No
  agent found driving nonlinear, contact or material-identification
  inverse loops.

So the open ground is: hard nonlinear solid mechanics, graded on physics.

## Candidate tracks

Each track is a ladder: a rung with analytical truth first, then a rung
where modelling judgment decides the outcome.

### A. Instability and path following (recommended)
1. Snap-through: von Mises truss, shallow arch, Lee frame. Needs
   arc-length continuation and branch switching; naive Newton jumps
   branches. Truth: analytical and Crisfield benchmark curves.
2. Elastomer bifurcations: Biot surface instability (critical strain
   0.444), creasing (about 0.35), cavitation. Truth: analytical thresholds.
3. Cylindrical shell buckling with imperfection sensitivity: reproduce
   the gap between the classical load and test loads (NASA SP-8007
   knockdown factors). Test-data availability (unverified).

Why: named as the weakest area in FEM-Bench, cheap to compute, and the
continuation solver it produces is reused by track D.

### B. Fracture, ending in a blind challenge (recommended)
1. Phase-field fracture benchmarks: single-edge-notched tension and shear
   (Miehe 2010), Kalthoff–Winkler crack angle. Traps: mesh size versus
   length scale, staggered-solve stagnation, energy split, irreversibility.
2. Third Sandia Fracture Challenge: additively manufactured 316L with
   internal cavities; 21 teams scattered widely. Data reported at
   acdc.alcf.anl.gov/mdf/detail/pub_29_kramer_third_v1.2/ (download
   untested). Alternative: Damage Mechanics Challenge (Morris et al. 2024),
   calibration data on Purdue PURR 3905 (unverified).

Why: rung 2 is the closest thing to a real exam — calibrate from the
supplied test data, predict, then compare with the experiment and with
21 expert teams.

### C. Contact
1. Hertz, contact patch test, ironing benchmark. CalculiX contact is
   reported fragile; active-set changes break Newton.
2. Sandia Mechanics Challenge 2023: threaded fastener joint under shock.
   Packet reported on MDF Open as smc2023_v1.1; blind and post-blind
   report OSTI 3013870 (both unverified).

### D. Nonlinear inverse design (recommended)
Design a metamaterial unit cell whose stress–strain curve matches a
target that includes buckling or contact stiffening; stretch goal is a
bistable cell with a prescribed energy barrier. Differentiating through
bifurcations is ill-conditioned. Grade: RMS curve error on an
independent fine-mesh verification solve in a different solver. Baselines:
Bastek & Kochmann 2023 (Nature Machine Intelligence), Bordiga et al. 2024.
A printed and tested cell would be the strongest grade.

### E. Material identification from real full-field data
Elastoplastic identification on the Sandia 304L dataset: DIC plus IR
thermography for seven geometries (Scientific Data,
doi 10.1038/s41597-024-03949-y; download untested). Calibrate on some
geometries, predict held-out ones. Natural sequel to the VHB 4910 trial;
easier warm-up is EUCLID-style constitutive discovery (arXiv 2510.24747).

### F. Constrained topology optimization
Buckling-constrained (topBuck250, arXiv 2101.02973) or stress-constrained
L-bracket, passed only if an independent CalculiX run on a refined mesh
confirms the constraint. Closest to existing agent work, so least novel.

### G. Additive-manufacturing residual stress
NIST AM-Bench 2022 AMB2022-01, IN718 bridge: predict residual strain and
deflection after cutting. Data open at data.nist.gov/od/id/mds2-2711.
Heaviest compute and modelling load (layer activation, inherent strain
versus thermomechanical).

## Recommendation

Deep dive on A, B and D, in that order. A builds the continuation
machinery and has clean analytical truth; B supplies the blind-challenge
exam; D is the inverse-design counterpart and reuses A's solver. E is the
swap-in if a real-data track is preferred over D. C and G are the next
tier; F is lowest priority.

Also dropped: dynamic crack branching and composite failure exercises
(contested ground truth), Taylor bar and fatigue (less distinctive).

## Before starting any track
- Download-test each dataset and check its licence.
- Read the primary papers for the reference values in full and ingest
  them into `literature/`.
- Fix the grading protocol first (read `evaluation.md`): independent
  verifier, mesh convergence, held-out data for the blind rungs.
