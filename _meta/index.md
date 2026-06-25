---
name: index
description: Entry-point index for this project's knowledge graph.
---

# Index

Orientation for the project knowledge graph. Updated by `/wrap`, `/ingest`,
and `/new-experiment`.

## Maps of Content

- [[gpu-differentiable-physics-simulation]] — the GPU-accelerated & differentiable
  physics-sim landscape (2026-06-25 ultracode sweep): DSLs/compilers (Warp, Taichi),
  the JAX physics stack, GPU lattice-Boltzmann, GPU CFD, diff-physics engines —
  mapped to our two loops + the window-shade natural-convection recommendation.

## Concepts (seeded 2026-06-16; +6 on 2026-06-25)

Three agentic design axes:
- [[agent-as-solver-operator]] — axis 1 / loop 1: drive a solver (write/run/debug CalculiX decks)
- [[differentiable-inverse-design]] — axis 2 / loop 2: differentiate a solver (gradient design via autodiff FEM)
- [[code-from-scratch-numerical-solver-agent]] — axis 3: agent writes the solver itself (AutoNumerics)

Cross-cutting:
- [[agentic-design-optimization]] — shared target of both loops' advanced trials
- [[multi-agent-self-correction]] — role-split agents correcting each other
- [[physics-grounded-evaluation]] — grade against physics/ground truth, not self-report

GPU / differentiable physics-sim (2026-06-25 sweep):
- [[gpu-differentiable-physics-simulation]] · [[differentiable-lattice-boltzmann]] ·
  [[boussinesq-natural-convection]] · [[implicit-differentiation-through-solve]] ·
  [[weak-form-fem-dsl]] · [[multi-backend-differentiable-physics-framework]]

## Literature (15 ingested 2026-06-16)

Agentic-operator & benchmarks: [[ni2023mechagents]] (5) · [[deotale2026allfem]]
(5) · [[mohammadzadeh2025fembench]] (5) · [[yue2025foamagent]] (4) ·
[[guo2025engdesign]] (4) · [[park2026self]] (TopOptAgents, 4) ·
[[du2026autonumerics]] (3).
Differentiable/inverse: [[xue2022jaxfem]] (5) · [[xue2025implicit]] (3) ·
[[bouziani2024differentiable]] (3).
Solver repos: [[calculix]] (5, committed) · [[deepmodeling-jax-fem]] (4) ·
[[fenics-dolfinx]] (3) · [[nvidia-warp]] (3) · [[sfepy]] (3).
Triage curated → `raw/_candidates/_done/2026-06-16-llm-agents-solid-mechanics-solvers.md`

### GPU / differentiable physics-sim sweep (2026-06-25, ultracode workflow — 93 found, 18 verified)
GPU/differentiable solvers: [[xlb]] (4, JAX LBM) · [[phiflow]] (4, multi-backend NS+heat) ·
[[optimistix]] (4) · [[lineax]] (3.5) · [[pyfr]] (2.5, GPU high-order CFD) ·
[[mujoco-mjx]] (2) · [[taichi]] (3, stale/maintenance-mode).
Surveys/papers: [[newbury2024diffsim]] (3) · [[sapienza2024diffprog]] (4) ·
[[difvm2026]] (3) · [[zheng2022venetianblinds]] (4, natural-convection validation ref).
Synthesis + window-shade recommendation → [[gpu-differentiable-physics-simulation]].
Triage archived → `raw/_candidates/_done/2026-06-25-gpu-differentiable-physics-sim.md`

## Active experiments

- `experiments/2026-06-16-calculix-cantilever-baseline/` — **done** (trial 1):
  agent-as-operator on CalculiX. Agent authored its own deck unaided →
  deflection 0.275% err (PASS). Deterministic reference + `claude -p` harness
  (subscription, no API key), pass@k.
- `experiments/2026-06-21-calculix-cantilever-design-loop/` — **done** (trial 2):
  operator design loop (minimise mass s.t. deflection+stress). Agent found the
  min-width corner + shear correction → feasible, +0.07% above FE-true optimum,
  3 FE evals.
- `experiments/2026-06-22-calculix-tapered-design-loop/` — **done** (trial 2b):
  TAPERED design loop (no closed-form deflection → FE essential). Agent ran a
  29-design FE search → matched the FE-true optimum (b=10/h_root=148/h_tip=37 mm,
  7.26 kg, 14.8% lighter than prismatic); noted "beam theory unsafe".
- `experiments/2026-06-22-jaxfem-topology-optimization/` — **done** (trial 3,
  loop 2): differentiable inverse design. Agent wrote its own OC optimizer using
  JAX autodiff sensitivities → min-compliance topology opt, +0.31% vs reference,
  PASS. All 3 agentic axes now demonstrated (drive / design / differentiate a
  solver). GPU now working (post-reboot); added external-gold verification
  (forward model → Timoshenko <1%; reproduces the canonical MBB beam) + GPU grid
  scaling to 120×40.
- `experiments/2026-06-22-jaxfem3d-cantilever-topology/` — **done** (trial 4):
  3D differentiable FEM (8-node hex, SIMP) in JAX on the GPU. Element exact
  (6 rigid-body modes); forward model converges to Euler–Bernoulli (−0.61%);
  3D topology opt at compliance 49.98. Matrix-free (sparse) solver is
  bit-identical to dense and solves 121,875 unknowns in 0.43 s (dense ≈119 GB).
  **Loop-2 agentic pass@10**: handed only the differentiable solver, the agent
  writes its own optimiser — 10/10 feasible, ~1.2% stiffer than our reference,
  leakage clean.
- `experiments/2026-06-23-vhb-viscohyperelastic-calibration/` — **done** (trial 5):
  inverse problem / material-model calibration. Ogden + Bergström–Boyce
  viscohyperelastic model (differentiable JAX material point) calibrated to real
  VHB 4910 cyclic data (Hossain 2012, CC-BY-4.0) by gradient descent: fit R²=0.86,
  predicts a held-out strain rate at R²=0.94 and held-out larger amplitudes at
  R²=0.85. Shows JAX autodiff is as strong for calibration/inverse problems as for
  design. Data in `raw/data/vhb4910-hossain2012/`.

## QA review surface

`docs/qa/` (GitHub Pages, linked from the landing page → *QA review*): per-trial
interactive 3D scenes + plots + an agentic-legitimacy panel + external-gold
verification. Generated by `_meta/qa/build_qa.py`. See `_meta/qa/README.md`.

## Open questions

- Does deck-vs-API (CalculiX text deck vs. FEniCS Python) change agent success rate?
- Multi-agent decomposition vs. single-agent + retry — how much does structure buy?
- Is a local fine-tune (cf. ALL-FEM) worth it on this box, or does prompt-only suffice?
- Axis ablation: does an agent do better generating a CalculiX deck (drive),
  or hand-rolling a small Python FEM (code-from-scratch)?
- Install caveat to verify before loop-2 trials: JAX-FEM needs a recent jaxlib
  (CUDA 12.6+) emitting sm_120 PTX for the RTX 5080 (Blackwell) — confirm
  `jax.devices()` sees the GPU before trusting timings.
