---
kind: repo
name: "NVIDIA Warp"
url: https://github.com/NVIDIA/warp
commit:
source: "raw/repos/nvidia-warp.md"
added: "2026-06-16"
relevance: 3
status: skimmed
related_experiments: []
related_concepts:
  - differentiable-inverse-design
  - agent-as-solver-operator
  - agentic-design-optimization
  - physics-grounded-evaluation
tags: [gpu, differentiable-simulation, fem, autodiff, python, cuda]
---

# NVIDIA Warp

## Purpose

Warp is a Python framework that JIT-compiles ordinary Python functions (decorated `@wp.kernel`)
into efficient CUDA (or CPU) kernel code. Kernels are differentiable via built-in reverse-mode
autodiff, and tensors interop zero-copy with PyTorch, JAX, and Paddle. For this project it is the
**differentiable-GPU fallback** for loop 2: when a problem needs a custom hand-written physics
formulation that beats a turnkey structural solver, Warp lets an agent author and differentiate
that formulation on the GPU through a clean Python API rather than dropping to C++/CUDA.

This positions it primarily against [[differentiable-inverse-design]] — Warp's autodiff makes
gradients of a simulation w.r.t. design parameters available for end-to-end optimization, which is
the core enabler for gradient-based inverse design.

## Shape

- **Core kernel model** — `@wp.kernel` Python functions JIT-compiled to CUDA/CPU; launched with
  `wp.launch(...)`. Differentiable (reverse-mode), so kernels slot into ML pipelines.
- **`warp.fem`** — FEM building blocks. Bundled examples cover diffusion, mixed elasticity, APIC
  fluid, magnetostatics, distortion energy, nonconforming contact, and shape optimization
  (Darcy level-set, elastic shape optimization).
- **`warp.examples.optim`** — differentiable optimization examples (diffray, fluid checkpoint,
  particle repulsion, Navier-Stokes perturbation).
- **Interop** — PyTorch / JAX / Paddle, zero-copy arrays.
- **Install** — `pip install warp-lang` (PyPI wheels bundle the CUDA 12 runtime; driver >= 525 for
  CUDA 12 builds). Python >= 3.10, x86-64/ARMv8 Linux+Windows, Apple Silicon macOS. CPU fallback
  works without a GPU; GPU needs a CUDA-capable NVIDIA card (>= GTX 9xx).
- **License** — Apache-2.0. (Build-from-source pulls NVIDIA libmathdx under the NVIDIA SLA; PyPI
  wheels statically link it.)

## Useful bits

- **Easiest GPU install in the differentiable-sim space.** A single `pip install warp-lang` with the
  CUDA runtime bundled — no separate CUDA toolkit, no conda gymnastics — which makes it cheap for an
  agent to spin up in a fresh environment. This is its strongest practical advantage as a fallback
  for [[agent-as-solver-operator]].
- **Clean differentiable Python API.** Kernels are plain typed Python; reverse-mode autodiff is
  built in. An agent can write a custom constitutive/structural formulation and get gradients
  without leaving Python — relevant to [[agentic-design-optimization]] where the agent itself
  authors the solver.
- **Solid mechanics is example-level, not turnkey.** `warp.fem` ships elasticity/diffusion/fluid
  *examples* (mixed elasticity, elastic shape optimization), but there is no packaged structural
  solver: you build your own weak form, elements, and assembly. Compared to a turnkey FEM tool this
  is DIY — appropriate only as loop-2 fallback when a custom formulation is the point, not for
  off-the-shelf [[physics-grounded-evaluation]] of standard structural problems.

## Follow-up

- Skim `warp/examples/fem/example_mixed_elasticity.py` and `example_elastic_shape_optimization.py`
  to gauge how much scaffolding a custom structural formulation actually requires.
- Confirm gradient quality / checkpointing behavior for multi-step structural sims (see
  `example_fluid_checkpoint.py` pattern) before relying on it for inverse design.
- Compare against a turnkey structural solver note to sharpen the loop-2 fallback decision boundary.
