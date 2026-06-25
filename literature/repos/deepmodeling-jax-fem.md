---
kind: repo
name: "JAX-FEM (deepmodeling/jax-fem)"
url: https://github.com/deepmodeling/jax-fem
commit:
source: "raw/repos/deepmodeling-jax-fem.md"
added: "2026-06-16"
relevance: 4
credibility: 5
status: skimmed
related_experiments: []
related_concepts:
  - differentiable-inverse-design
  - agent-as-solver-operator
  - agentic-design-optimization
  - multi-agent-self-correction
  - physics-grounded-evaluation
tags: [jax, fem, differentiable-solver, inverse-design, solid-mechanics, loop-2]
---

# JAX-FEM (deepmodeling/jax-fem)

The library/API companion to the JAX-FEM paper ([[xue2022jaxfem]]). The paper
note covers the method and claims; this note covers the actual codebase an
agent must drive: what it provides, how to install it, and how to use its
autodiff to do inverse design. This is the **loop-2 differentiable solver**.

## Purpose

A differentiable finite element package built on JAX. It is Automatic
Differentiation + FEM: forward FE assembly and solves are written so that JAX
can take gradients of any scalar objective with respect to design inputs
(material parameters, density fields, loads, BCs) **without hand-derived
sensitivities (no manual adjoint)**. This is exactly what
[[differentiable-inverse-design]] needs — the gradient of a physics-grounded
objective flows back to the design variables automatically.

## Coverage (solid mechanics)

- Elements: 2D quad/tri, 3D hex/tet; first and second order.
- BCs: Dirichlet / Neumann / Robin.
- Physics: heat equation, **linear elasticity, hyperelasticity, plasticity
  (both macro and crystal plasticity)**, plus multi-physics (monolithic
  multi-variable) and Stokes flow.
- Demonstrated inverse/design applications: topology optimization, optimal
  thermal control.
- PETSc integration for solver options.

## Shape (what an agent drives)

The agent-facing pattern (per the paper/docs) is: subclass a `Problem` (e.g.
`LinearElasticity`, `HyperElasticity`, `Plasticity`) and define the weak form
via volume/surface integral kernels; build a mesh (or import one, e.g. via
meshio/Gmsh) and set boundary locations + value functions; call the solver to
get the forward solution; then wrap a scalar objective and use JAX
(`jax.grad` / VJP via the package's differentiable solve) to get design
gradients for an optimizer. For an agent doing inverse design this means:
**the agent writes the Problem definition + objective, and the gradient is
free** — no adjoint code to generate or debug. That makes it a clean
[[agent-as-solver-operator]] target for [[agentic-design-optimization]], and
the gradient/residual signals are natural footholds for
[[physics-grounded-evaluation]] and [[multi-agent-self-correction]].

## JAX-FEM Express

The repo advertises **JAX-FEM Express**, an LLM-agent front-end for JAX-FEM
(hosted at bohrium.com/apps/jax-fem-express, with a demo video). Directly
relevant prior art for this project's thesis — an LLM driving a differentiable
solver — and worth examining as a baseline for the [[agent-as-solver-operator]]
interface. Note: it is a hosted app, not (visibly) part of this open repo.

## Install / runtime

- Python package; depends on **JAX**. Standard route is `pip install jax-fem`
  (or install from source); follow the docs at
  https://deepmodeling.github.io/jax-fem/ for the authoritative steps and the
  correct JAX/jaxlib + CUDA pairing.
- **GPU/CUDA**: JAX-FEM is "GPU-accelerated"; real performance needs a
  CUDA-enabled jaxlib. The hard part is always the JAX↔CUDA match, not jax-fem
  itself.
- **License: GPL-3.0** — copyleft. Fine for internal research; commercial use
  requires contacting the author. Anything we build *on top of* and distribute
  inherits GPL obligations — keep that in mind before vendoring code.

## Install caveat for our RTX 5080 (Blackwell, sm_120)

Blackwell `sm_120` is newer than the CUDA target baked into older jaxlib
wheels, so an off-the-shelf `pip install "jax[cuda12]"` may not emit `sm_120`
PTX and can fall back to CPU or fail at first kernel launch. Install a **recent
JAX/jaxlib build** (CUDA 12.6+/cuDNN matching, current jaxlib) so the GPU is
actually used; verify with `jax.devices()` showing the 5080 before trusting any
timing. Pin the working JAX version once found.

## Useful bits

- Differentiable solve = autodiff design gradients with no manual adjoint —
  the core enabler for loop-2.
- Built-in `Plasticity` (macro + crystal) and `HyperElasticity` problems cover
  the nonlinear solid-mechanics cases we care about.
- PETSc backend gives robust linear/nonlinear solver options at scale.

## Follow-up

- Read the docs' install page and pin a 5080-working JAX/jaxlib combo; record
  it in an experiment config.
- Inspect a `LinearElasticity` / `HyperElasticity` example end-to-end to nail
  the exact Problem-subclass + objective + `jax.grad` API surface an agent must
  emit.
- Probe **JAX-FEM Express**: is the prompt→Problem interface documented? Use it
  as a baseline for [[agent-as-solver-operator]].
- Confirm the licensing posture (GPL-3) is acceptable for how we intend to use
  generated/derived code.

## 2026-06-25 re-verification — CFD / loop-2 angle (landscape sweep)

Re-verified active (commits Jun 2026), peer-reviewed (CPC 2023). For the natural-convection work:

- AD-driven Newton: the user implements get_tensor_map/get_mass_map/get_surface_maps (or get_universal_kernel for arbitrary multiphysics) on a Problem subclass; JAX autodiff produces the tangent stiffness, so no manual sensitivity derivation for either the nonlinear solve or the inverse/design gradient. This is the exact pattern to study before extending our own differentiable FEM.
- Solver stack is assembled-matrix, not matrix-free: backends are JAX built-in sparse (GPU), UMFPACK (CPU direct), and PETSc KSP (CPU iterative + preconditioning), plus AMGX env support; nonlinear solvers include Newton-Raphson, arc-length, and dynamic relaxation. Contrast with our own matrix-free CG-on-GPU approach.
- Install is conda env (Python 3.13, NumPy 2.4, petsc4py 3.25, meshio, gmsh, fenics-basix) plus `pip install jax-fem` (or `pip install -e .`), with JAX installed separately per the official JAX hardware instructions -- so the CUDA/Blackwell wheel choice is decoupled from the package. License is GPL-3.0 (commercial use requires contacting the author).
- Ships Poisson/heat, elasticity, hyperelasticity, plasticity, and **Stokes flow** — but **no Navier–Stokes/Boussinesq**; the `get_universal_kernel` custom-weak-form path is the credible route to a coded Boussinesq convection solve with autodiff h(angle) sensitivities. **License: GPL-3.0** (matters if our code becomes a derivative work).
- The closest published analog to our own hand-rolled JAX FEM. See [[gpu-differentiable-physics-simulation]].
