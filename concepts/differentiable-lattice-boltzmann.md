---
kind: concept
name: "differentiable-lattice-boltzmann"
status: seedling
added: "2026-06-25"
sources: ["(workflow research 2026-06-25) — GPU/differentiable physics-sim landscape sweep"]
related_concepts:
  - gpu-differentiable-physics-simulation
  - boussinesq-natural-convection
related_experiments: []
tags: [gpu, differentiable-simulation]
---

# differentiable-lattice-boltzmann

## Definition

A lattice-Boltzmann CFD method implemented so that automatic differentiation flows through the collision/streaming step and boundary-condition kernels, yielding gradients for sensitivity analysis and inverse problems.

## Why it matters here

Captures XLB's core technique and distinguishes LBM-based differentiable CFD from FEM/FVM approaches in the landscape.

## Connections

- [[gpu-differentiable-physics-simulation]]
- [[boussinesq-natural-convection]]
