---
kind: repo
name: "Optimistix"
url: https://github.com/patrick-kidger/optimistix
commit:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 4
status: skimmed
related_experiments: []
related_concepts:
  - implicit-differentiation-through-solve
  - differentiable-inverse-design
tags: [jax, optimization, root-finding, differentiable, python]
---

# Optimistix

## Purpose

Optimistix is a modular JAX+Equinox library of general-purpose nonlinear solvers: root-finding, unconstrained minimisation, nonlinear least-squares, and fixed-point iteration. Built by Jason Rader / Patrick Kidger's group, it composes with Optax and uses lineax for the inner linear solves, and crucially is fully differentiable — it backpropagates through converged solutions via the implicit function theorem rather than naively unrolling iterations. It is the solver layer, not a physics engine: you supply a residual or objective and it converges and differentiates it, inheriting JAX's GPU/TPU execution and jit/vmap for free.

## Shape

- **GPU:** yes (indirect) — pure JAX library, inherits JAX's GPU/TPU backend automatically; no CUDA code of its own, GPU support is whatever the installed jaxlib provides.
- **Differentiable:** yes — built on JAX+Equinox; solvers are differentiable, with implicit-function-theorem adjoints for root-finds/fixed-points so you can backprop through converged solutions (not just unrolled iterations).
- **Install:** conda-forge (noarch): `conda install conda-forge::optimistix` v0.1.0; also pip `pip install optimistix`. Pure-Python noarch package, no compiled extensions.
- **License:** Apache-2.0
- **Maintained:** active — v0.1.0 released 2026-02-16 (conda-forge upload 2026-02-21); latest main commit 2026-05-13 ("fix reporting nonconvergence as divergence"), steady commits through March-May 2026.

Capabilities:
- Nonlinear root-finding (Newton, Chord, Bisection)
- Unconstrained minimisation (BFGS, gradient descent, Nonlinear CG, Levenberg-Marquardt for NLLS)
- Nonlinear least-squares (Gauss-Newton, Levenberg-Marquardt, Dogleg)
- Fixed-point iteration
- Differentiable through converged solutions via implicit function theorem
- PyTree-valued state, jit/vmap compatible, interoperates with Optax and uses lineax for linear solves
- Auto-conversion between problem types (e.g. root-find -> least-squares)

## Useful bits

- Implicit-function-theorem adjoints mean you get correct, memory-cheap gradients through a converged nonlinear solve — exactly what inverse/differentiable design of a steady-state physics solve needs (no unrolled-iteration backprop blowup).
- Pure-Python noarch package: installs cleanly via conda-forge with no compilation, so it adds zero sm_120/Blackwell build risk on the no-sudo stack — GPU support is whatever jaxlib already provides.
- Fixed-point and Newton/Levenberg-Marquardt solvers map directly onto steady-state Boussinesq natural-convection solves and onto fitting/identifying parameters (e.g. an h(angle) surrogate) from simulation residuals.

## Reality check

Not vaporware and not overstated. The README's "GPU/TPU + autodiff" claim is honest because those properties are inherited from JAX, not bespoke — the flip side is that Optimistix itself contributes no GPU code, so any Blackwell sm_120 issue would be a jaxlib problem, not an Optimistix one. From the same well-regarded author as Equinox/Diffrax/Lineax, with real recent commits (latest 2026-05-13) and a clean v0.1.0 on both PyPI and conda-forge. Main honesty caveat: it is a nonlinear-solve toolkit, NOT a CFD/PDE solver — anyone expecting it to "do natural convection" will be disappointed; it does the algebra around a residual you must build yourself. Pre-1.0 versioning means minor API churn is…

## Relevance here

- **Window-shade (natural convection):** Indirect but genuinely useful. A steady-state 2D Boussinesq natural-convection problem is a coupled nonlinear residual (momentum + energy + incompressibility); Optimistix's Newton / fixed-point / Levenberg-Marquardt solvers are a natural choice to drive that residual to convergence on GPU, and its differentiability lets you take d(h)/d(slat angle) through the converged flow for gradient-based desi…
- **Project:** High for the project's loop-2 (differentiable/inverse design) thesis. Optimistix is essentially the canonical JAX answer to "I have a differentiable residual/objective and need a robust, composable, differentiable nonlinear solver," which is the recurring need under SIMP topology optimization, inverse design, and parameter identification. It pairs naturally with the existing hand-rolled matrix-fre…

## Follow-up

- Trial the most relevant capability against our stack before relying on it.
