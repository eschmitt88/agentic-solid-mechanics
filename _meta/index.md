---
name: index
description: Entry-point index for this project's knowledge graph.
---

# Index

Orientation for the project knowledge graph. Updated by `/wrap`, `/ingest`,
and `/new-experiment`.

## Maps of Content

(promote a cluster of ≥5 related concepts into `mocs/<theme>.md`)

## Concepts (seeded 2026-06-16)

Three agentic design axes:
- [[agent-as-solver-operator]] — axis 1 / loop 1: drive a solver (write/run/debug CalculiX decks)
- [[differentiable-inverse-design]] — axis 2 / loop 2: differentiate a solver (gradient design via autodiff FEM)
- [[code-from-scratch-numerical-solver-agent]] — axis 3: agent writes the solver itself (AutoNumerics)

Cross-cutting:
- [[agentic-design-optimization]] — shared target of both loops' advanced trials
- [[multi-agent-self-correction]] — role-split agents correcting each other
- [[physics-grounded-evaluation]] — grade against physics/ground truth, not self-report

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

## Active experiments

(none yet — trial roadmap in README; trial 1 = CalculiX cantilever operator baseline)

## Open questions

- Does deck-vs-API (CalculiX text deck vs. FEniCS Python) change agent success rate?
- Multi-agent decomposition vs. single-agent + retry — how much does structure buy?
- Is a local fine-tune (cf. ALL-FEM) worth it on this box, or does prompt-only suffice?
- Axis ablation: does an agent do better generating a CalculiX deck (drive),
  or hand-rolling a small Python FEM (code-from-scratch)?
- Install caveat to verify before loop-2 trials: JAX-FEM needs a recent jaxlib
  (CUDA 12.6+) emitting sm_120 PTX for the RTX 5080 (Blackwell) — confirm
  `jax.devices()` sees the GPU before trusting timings.
