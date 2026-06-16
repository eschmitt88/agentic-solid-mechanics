---
kind: concept
name: "Physics-grounded evaluation"
status: seedling
added: "2026-06-16"
sources: ["mohammadzadeh2025fembench", "guo2025engdesign", "ni2023mechagents"]
related_concepts: ["agent-as-solver-operator", "agentic-design-optimization", "multi-agent-self-correction"]
related_experiments: []
tags: ["evaluation", "hce", "core-loop"]
---

# Physics-grounded evaluation

## Definition

Grading an agent's simulation output against objective physical ground truth —
analytical solutions, mesh-convergence behavior, conservation laws, or
simulation-in-the-loop functional verification — rather than against the agent's
self-reported confidence or mere execution-without-error.

## Why it matters here

This is the reason solid mechanics is a good agentic testbed at all:
correctness is checkable. It is also how this project avoids the classic
autonomous-loop failure of overfitting to a self-generated signal (see the
project's HCE rule — held-out `test/` scoring, validation vs. final metrics
separation). "Execution success" (Foam-Agent, ALL-FEM) is necessary but not
sufficient; we grade against physics.

## Connections

- Benchmark realizations: [[mohammadzadeh2025fembench]] (objective verification,
  pass@5) and [[guo2025engdesign]] (simulation-driven functional verification).
- The critic in [[multi-agent-self-correction]] is stronger when its signal is
  physics-grounded rather than self-consistency-based.
- Grades both [[agent-as-solver-operator]] and [[agentic-design-optimization]].
