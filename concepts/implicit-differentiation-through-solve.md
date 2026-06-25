---
kind: concept
name: "implicit-differentiation-through-solve"
status: seedling
added: "2026-06-25"
sources: ["(workflow research 2026-06-25) — GPU/differentiable physics-sim landscape sweep"]
related_concepts:
  - differentiable-inverse-design
  - gpu-differentiable-physics-simulation
related_experiments: []
tags: [gpu, differentiable-simulation]
---

# implicit-differentiation-through-solve

## Definition

Computing gradients through a converged linear or nonlinear solve using the implicit function theorem (differentiating the optimality/residual condition) rather than unrolling solver iterations, giving stable, memory-cheap adjoints.

## Why it matters here

The correct primitive for Loop-2 inverse design (d(h)/d(angle) through a converged steady flow); the shared differentiator of Optimistix and Lineax.

## Connections

- [[differentiable-inverse-design]]
- [[gpu-differentiable-physics-simulation]]
