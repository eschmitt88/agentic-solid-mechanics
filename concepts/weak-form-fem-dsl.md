---
kind: concept
name: "weak-form-fem-dsl"
status: seedling
added: "2026-06-25"
sources: ["(workflow research 2026-06-25) — GPU/differentiable physics-sim landscape sweep"]
related_concepts:
  - gpu-differentiable-physics-simulation
  - code-from-scratch-numerical-solver-agent
related_experiments: []
tags: [gpu, differentiable-simulation]
---

# weak-form-fem-dsl

## Definition

A domain-specific language in which a user declares finite-element function spaces and Galerkin integrands (constant/linear/bilinear forms) and the framework assembles and solves the sparse system on GPU, with the forms differentiable by autodiff.

## Why it matters here

Names the distinctive capability of warp.fem and JAX-FEM's custom-weak-form path; central to how an agent would author a Boussinesq solver.

## Connections

- [[gpu-differentiable-physics-simulation]]
- [[code-from-scratch-numerical-solver-agent]]
