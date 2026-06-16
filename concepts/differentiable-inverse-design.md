---
kind: concept
name: "Differentiable inverse design"
status: seedling
added: "2026-06-16"
sources: ["xue2022jaxfem"]
related_concepts: ["agentic-design-optimization", "agent-as-solver-operator"]
related_experiments: []
tags: ["differentiable-fem", "inverse-design", "core-loop", "gpu"]
---

# Differentiable inverse design

## Definition

Solving design or parameter-identification problems by differentiating through
a finite-element solve — using automatic differentiation to get exact
sensitivities of an objective w.r.t. design variables (shape, topology,
material), then driving a gradient-based optimizer. No hand-derived adjoints.

## Why it matters here

This is loop (2) of the project. The agent's role shifts from writing solver
input to **orchestrating** the optimization: defining the objective and
constraints, choosing/driving the optimizer, and interpreting results — a
genuinely different agentic skill from [[agent-as-solver-operator]]. Planned
substrate: **JAX-FEM** (GPU-native autodiff), with NVIDIA Warp as the
custom-kernel fallback.

## Connections

- Canonical solver/method: [[xue2022jaxfem]] (JAX-FEM; topology optimization,
  inverse problems on GPU).
- When an LLM agent drives this loop toward a design goal it becomes
  [[agentic-design-optimization]] (cf. TopOptAgents).
- Results graded by [[physics-grounded-evaluation]].
