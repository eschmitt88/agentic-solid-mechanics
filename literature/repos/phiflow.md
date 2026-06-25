---
kind: repo
name: "PhiFlow (ΦFlow)"
url: https://github.com/tum-pbs/PhiFlow
commit:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 4
status: skimmed
related_experiments: []
related_concepts:
  - multi-backend-differentiable-physics-framework
  - gpu-differentiable-physics-simulation
  - boussinesq-natural-convection
  - differentiable-inverse-design
tags: [gpu, cfd, differentiable, jax, pytorch, python]
---

# PhiFlow (ΦFlow)

## Purpose

PhiFlow (ΦFlow) from TUM Physics-based Simulation is a backend-agnostic, fully differentiable PDE/physics framework whose primary strength is grid-based incompressible Navier-Stokes for ML and gradient-based optimization. Written in Python, it runs the same simulation code on NumPy, PyTorch, JAX, or TensorFlow, inheriting each backend's autodiff and GPU execution (via phi.jax.flow → JAX/XLA → CUDA). It was published at ICML 2024 ("ΦFlow: Differentiable Simulations for PyTorch, TensorFlow and Jax"). It targets end-to-end differentiable pipelines that mix learned models with physics solvers — flow control, inverse problems, PIV, and gradient-descent-on-simulation demos.

## Shape

- **GPU:** Yes — GPU runs through the chosen backend. With phi.jax.flow, tensors execute under JAX/XLA and use CUDA automatically; PhiFlow ships no JAX CUDA kernels of its own (legacy custom CUDA kernels were TensorFlow-only, optional). README states "fully differentiable simulations that c…
- **Differentiable:** Yes — genuinely differentiable. Backend-agnostic autodiff (jax.grad / torch.autograd); repo includes Gradient Descent, Optimize Throw, Learning to Throw, PIV, and Differentiable Pressure demos. ICML 2024 paper is explicitly about differentiable simulation. Not an overstated claim…
- **Install:** pip (PyPI: `pip install phiflow`, Python 3.10-3.12); no conda-forge package. Pin a develop-branch commit since the 3.4.0 tag (2025-08) lags active development.
- **License:** MIT (verified via GitHub API)
- **Maintained:** active — latest tagged release 3.4.0 on 2025-08-02, but commits are recent (last commit 2026-05-27, last push 2026-06-22 per GitHub API); ~1894 stars. The release tag is stale relative to a live develop branch.

Capabilities:
- Grid-based incompressible Navier-Stokes (Chorin pressure projection, semi-Lagrangian advection)
- Heat flow / advection-diffusion on grids and meshes
- Particle methods (SPH, FLIP)
- Unstructured mesh simulations (backward-facing step, wake flow)
- Backend-agnostic execution on NumPy/PyTorch/JAX/TensorFlow with shared code
- End-to-end differentiable sim+NN pipelines; gradient descent / optimization over simulations
- GPU execution via the selected backend's CUDA path
- Dimension-independent (1D/2D/3D) named-dimension tensor system

## Useful bits

- GPU and differentiability are inherited, not hand-rolled: selecting the JAX backend (phi.jax.flow) routes everything through JAX/XLA, so jax.grad/jax.jit work and CUDA is used automatically — meaning it sidesteps the Blackwell sm_120 kernel-compilation problem entirely (PhiFlow ships no JAX CUDA kernels of its own; the legacy custom CUDA kernels were TensorF…
- Ships ready-made grid demos relevant to a 2D enclosure CFD problem: Lid-Driven Cavity, Smoke Plume, Wake Flow, Taylor-Green, Moving/Variable Obstacles, plus standalone Heat Flow (advection-diffusion) on both grids and meshes — but there is NO named Boussinesq/natural-convection example; buoyancy coupling of temperature into the momentum source must be assemb…
- Install is pip-only (pip install phiflow, Python 3.10-3.12, MIT license); no conda-forge package exists, so it drops into a micromamba env as a pip dependency. Latest tagged release is 3.4.0 (2025-08-02) but the develop/master branch is actively committed (last commit 2026-05-27, last push 2026-06-22, ~1894 stars), so pin a commit rather than relying on the …

## Reality check

Hype is largely real, with two honest caveats. The differentiable + multi-backend + GPU claims all check out (ICML 2024, real autodiff demos, MIT, ~1894 stars, live commits through June 2026) — not vaporware. Caveat 1: the headline "incompressible Navier-Stokes" is a relatively standard semi-Lagrangian + pressure-projection grid solver, fine for research/ML but not a high-fidelity production CFD code; do not expect FLUENT-grade convection accuracy. Caveat 2: the natural-convection use case is aspirational here — there is NO Boussinesq example, so "supports heat flow" means advection-diffusion, not coupled buoyant convection out of the box. The stale 3.4.0 tag vs active develop branch is a mi…

## Relevance here

- **Window-shade (natural convection):** Moderately high but not turnkey. It directly provides the needed ingredients for the 2D venetian-blind problem — grid incompressible NS, scalar heat advection-diffusion, and an existing Lid-Driven Cavity enclosure demo — and is differentiable, so h(slat-angle) could become a gradient-friendly objective for inverse design. But there is no built-in Boussinesq buoyancy term and the slats need immerse…
- **Project:** High for the overall project thesis. It is one of the canonical differentiable, multi-backend physics frameworks (alongside Taichi/DiffTaichi, JAX-FEM, Warp), so it belongs in the GPU/differentiable-simulation landscape map. For loop-2 (differentiable/inverse design) it offers a Python-native, JAX-compatible NS solver that an LLM agent could plausibly drive and differentiate end-to-end, and for lo…

## Follow-up

- Trial the most relevant capability against our stack before relying on it.
