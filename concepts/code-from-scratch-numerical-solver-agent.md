---
kind: concept
name: "Code-from-scratch numerical solver agent"
status: seedling
added: "2026-06-16"
sources: ["du2026autonumerics", "ni2023mechagents"]
related_concepts: ["agent-as-solver-operator", "multi-agent-self-correction", "physics-grounded-evaluation"]
related_experiments: []
tags: ["agentic-simulation", "code-generation", "design-axis"]
---

# Code-from-scratch numerical solver agent

## Definition

An agent that authors the numerical solver *itself* — discretization, assembly,
linear solve, time-stepping — as classical numerical-analysis code generated
from a natural-language problem statement, rather than driving an existing
solver library or training a neural surrogate.

## Why it matters here

It is the third design axis for agentic simulation, distinct from the two we've
committed to:

1. **Drive a library** → [[agent-as-solver-operator]] (our loop 1, CalculiX).
2. **Differentiate a library** → [[differentiable-inverse-design]] (our loop 2, JAX-FEM).
3. **Write the solver from scratch** → *this concept* (AutoNumerics; MechAgents'
   FEM code generation sits partway here).

Tracking it lets us frame our library-driving choice as a deliberate point on a
spectrum, and gives a natural ablation: does an agent do better generating a
CalculiX deck, or hand-rolling a small FEM in Python? Transparency and
verifiability (residual checks) are the claimed upside; correctness scales worse
with problem complexity than a battle-tested library is the likely downside.

## Connections

- Canonical instance: [[du2026autonumerics]] (PDE-agnostic, residual-based
  self-verification on 24 problems).
- Partial instance: [[ni2023mechagents]] (agents write FEM code, though leaning
  on FEM libraries).
- Correctness graded by [[physics-grounded-evaluation]]; robustness leans on
  [[multi-agent-self-correction]].
