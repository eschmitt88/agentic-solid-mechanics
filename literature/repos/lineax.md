---
kind: repo
name: "Lineax"
url: https://github.com/patrick-kidger/lineax
commit:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 4
status: skimmed
related_experiments: []
related_concepts:
  - implicit-differentiation-through-solve
  - gpu-differentiable-physics-simulation
tags: [jax, linear-solver, differentiable, matrix-free, python]
---

# Lineax

## Purpose

Lineax is a JAX/Equinox library that unifies linear solves and linear least-squares (square, rectangular, and ill-posed Ax=b) behind a single user-extensible linear-operator abstraction, with autodifferentiation built in and no requirement to hand-write derivative rules. It supports both dense/structured operators and fully matrix-free operators (e.g. JacobianLinearOperator) that only ever apply matrix-vector products, and offers stable implicit-differentiation gradients through the solve rather than unrolling iterations. It ships direct solvers (LU, QR, SVD, Cholesky) and iterative ones (CG, NormalCG, GMRES, BiCGStab, LSMR), inheriting JAX's jit/vmap/GPU/TPU support. It is the linear-algebr…

## Shape

- **GPU:** Yes, by inheritance. Pure JAX with no custom kernels — runs on whatever backend JAX targets (CUDA/GPU/TPU). It carries no CUDA code of its own and no sm_120-specific concerns; GPU viability is entirely a function of the underlying jaxlib/CUDA build, which the project already runs…
- **Differentiable:** Yes. Core design claim, verified in paper + docs: solvers and operators are autodifferentiable with no user-supplied custom derivative rules. Provides numerically stable gradients through linear least-squares (implicit-diff through the solve rather than unrolling), which is the k…
- **Install:** pip (PyPI: `pip install lineax`); requires Python 3.10+, JAX 0.4.38+, Equinox 0.11.10+. No conda-forge package; pip into the existing solidmech/JAX conda env is the path. Pure-Python, no compiled CUDA of its own.
- **License:** Apache-2.0 (code). Companion arXiv paper is CC-BY 4.0.
- **Maintained:** Active. v0.1.1 released 2026-05-01; latest commit 2026-06-04 ("add state argument to invert"), with a steady stream of commits through May-June 2026. Part of Patrick Kidger's well-maintained JAX scientific stack (Equinox…

Capabilities:
- Unified API for square linear solves and rectangular/ill-posed linear least-squares (Ax=b)
- User-extensible AbstractLinearOperator hierarchy: dense, diagonal, tridiagonal, triangular, symmetric, plus matrix-free FunctionLinearOperator and JacobianLinearOperator
- Solvers: LU, QR (now via JAX ormqr), SVD, Cholesky, CG, NormalCG, GMRES, BiCGStab, LSMR, Diagonal/Triangular/Tridiagonal
- Matrix-free solves using only matrix-vector products (Jacobian-vector products via autodiff, never materializing A)
- Autodifferentiable end-to-end with numerically stable implicit-diff gradients through the solve
- PyTree-valued operators and structure tags (symmetric, positive-definite, etc.) that select efficient solver paths
- Inherits JAX jit/vmap/GPU/TPU/autoparallelism

## Useful bits

- Matrix-free + autodiff: FunctionLinearOperator/JacobianLinearOperator let you solve Ax=b using only mat-vec products while still getting stable gradients through the solve — directly applicable to adjoint-based inverse design and to slotting into the project's existing matrix-free CG FEM.
- Implicit differentiation of the linear solve gives numerically stable least-squares gradients without unrolling, which is the correct primitive for differentiable-inverse-design loops (e.g. solving for slat angle to hit a target effective h).
- Trivial to adopt: pure-JAX, `pip install lineax`, Apache-2.0, actively maintained (v0.1.1 May 2026, commits through June 2026); zero Blackwell/sm_120 risk beyond the already-working jaxlib build.

## Reality check

Claims hold up; no vaporware and no overstatement. Maturity is real: backed by a peer-reviewed-workshop paper (NeurIPS 2023 AI4Science), Apache-2.0, active commits into June 2026, and a maintainer with a strong track record across the JAX ecosystem. The only honest caveat is scope inflation risk in our context: it is a linear-solve/operator layer, not a physics simulator, and its CG already overlaps the project's existing matrix-free CG. The differentiable least-squares + clean operator abstraction are the genuine differentiators.

## Relevance here

- **Window-shade (natural convection):** Indirect but real. It will not model the 2D venetian-blind Boussinesq natural convection itself. Its value appears once you have a discretized convection/heat-transfer formulation: pressure-Poisson or implicit-diffusion solves, and especially the inverse map of solving for slat angle that targets a desired effective h — the stable autodiff through linear solves is exactly what makes h(angle) sensi…
- **Project:** High as infrastructure for loop 2 (differentiable/inverse design). It gives a principled, autodiff-stable, matrix-free linear-operator API that can replace ad-hoc CG plumbing in the hand-rolled JAX FEM and underpin SIMP topology-opt adjoint solves and any implicit-solver-in-the-loop optimization. Also a clean example for the "code-from-scratch numerical solver agent" theme of how to expose linear …

## Follow-up

- Trial the most relevant capability against our stack before relying on it.
