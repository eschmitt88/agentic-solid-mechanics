---
kind: concept
name: "multi-backend-differentiable-physics-framework"
status: seedling
added: "2026-06-25"
sources: ["(workflow research 2026-06-25) — GPU/differentiable physics-sim landscape sweep"]
related_concepts:
  - gpu-differentiable-physics-simulation
related_experiments: []
tags: [gpu, differentiable-simulation]
---

# multi-backend-differentiable-physics-framework

## Definition

A simulation framework that runs identical model code across multiple array/autodiff backends (e.g. JAX, PyTorch, Warp, NumPy), inheriting each backend's GPU execution and differentiation rather than shipping its own kernels.

## Why it matters here

Names the architectural pattern shared by PhiFlow and XLB and explains why JAX-backed paths sidestep sm_120 build risk.

## Connections

- [[gpu-differentiable-physics-simulation]]
