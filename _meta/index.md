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

- [[agent-as-solver-operator]] — loop 1: write/run/debug solver decks toward a goal
- [[differentiable-inverse-design]] — loop 2: gradient-based design via autodiff FEM
- [[agentic-design-optimization]] — shared target of both loops' advanced trials
- [[multi-agent-self-correction]] — role-split agents correcting each other
- [[physics-grounded-evaluation]] — grade against physics/ground truth, not self-report

## Literature (seeded 2026-06-16)

- [[ni2023mechagents]] (rel 5) · [[deotale2026allfem]] (rel 5) ·
  [[mohammadzadeh2025fembench]] (rel 5) · [[xue2022jaxfem]] (rel 5) ·
  [[yue2025foamagent]] (rel 4) · [[guo2025engdesign]] (rel 4)
- Triage backlog: `raw/_candidates/2026-06-16-llm-agents-solid-mechanics-solvers.md`

## Active experiments

(none yet — trial roadmap in README; trial 1 = CalculiX cantilever operator baseline)

## Open questions

- Does deck-vs-API (CalculiX text deck vs. FEniCS Python) change agent success rate?
- Multi-agent decomposition vs. single-agent + retry — how much does structure buy?
- Is a local fine-tune (cf. ALL-FEM) worth it on this box, or does prompt-only suffice?
