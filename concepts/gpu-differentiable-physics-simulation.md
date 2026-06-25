---
kind: concept
name: "gpu-differentiable-physics-simulation"
status: seedling
added: "2026-06-25"
sources: ["(workflow research 2026-06-25) — GPU/differentiable physics-sim landscape sweep"]
related_concepts:
  - differentiable-inverse-design
  - differentiable-lattice-boltzmann
  - weak-form-fem-dsl
  - implicit-differentiation-through-solve
related_experiments: []
tags: [gpu, differentiable-simulation]
---

# gpu-differentiable-physics-simulation

## Definition

The class of physics solvers that run on GPU and expose end-to-end gradients of outputs w.r.t. inputs/parameters via automatic differentiation or adjoints, enabling gradient-based inverse design and parameter identification.

## Why it matters here

The umbrella theme of the entire sweep and the proposed MoC; ties Warp, XLB, JAX-FEM, PhiFlow, Taichi together.

## Connections

- [[differentiable-inverse-design]]
- [[differentiable-lattice-boltzmann]]
- [[weak-form-fem-dsl]]
- [[implicit-differentiation-through-solve]]
