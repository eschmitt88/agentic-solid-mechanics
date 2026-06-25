---
kind: repo
name: "XLB (Accelerated Lattice Boltzmann)"
url: https://github.com/Autodesk/XLB
commit:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 4
status: skimmed
related_experiments: []
related_concepts:
  - differentiable-lattice-boltzmann
  - gpu-differentiable-physics-simulation
  - boussinesq-natural-convection
  - differentiable-inverse-design
tags: [gpu, lbm, cfd, jax, warp, differentiable, python]
---

# XLB (Accelerated Lattice Boltzmann)

## Purpose

XLB (Autodesk) is a fully differentiable, massively parallel Lattice-Boltzmann CFD library in Python, built on JAX with an additional NVIDIA Warp backend (single-GPU performance) and a Neon backend (multi-resolution grid refinement). Its JAX backend exposes LBM kernels and boundary conditions to JAX autodiff, making the whole solver end-to-end differentiable and composable with the JAX ML stack (Flax/Optax) for optimization and inverse problems; it scales to billions of cells across multi-GPU/TPU. It is an actively maintained, peer-reviewed project (Computer Physics Communications, 2024; arXiv 2311.16080) under Apache-2.0. The decisive caveat for us: it is presently isothermal — fluid-therma…

## Shape

- **GPU:** Yes — `xlb[cuda]` installs jax[cuda13]>=0.8.2; Warp (warp-lang>=1.10) and Neon GPU backends. Paper demonstrates multi-GPU scaling to billions of cells.
- **Differentiable:** Yes — design built on JAX autodiff; differentiable LBM kernels and boundary conditions; dedicated differentiable inverse-problem example added 2026-05-29 (commit #156).
- **Install:** pip (PyPI: `xlb`, with extras `xlb[cuda]` -> jax[cuda13]>=0.8.2, `xlb[warp]`, `xlb[tpu]`, `xlb[neon]`); also source via git. No conda-forge package.
- **License:** Apache-2.0
- **Maintained:** active — latest release v0.3.1 (2026-01-12); recent commits 2026-05-29 (differentiable inverse-problem example) and 2026-05-15 (Neon multi-res backend)

Capabilities:
- 2D/3D incompressible CFD via Lattice-Boltzmann Method
- End-to-end differentiable solver on JAX backend (autodiff through kernels + BCs)
- Distributed multi-GPU/TPU scaling (billions of cells, giga-scale LUPS)
- NVIDIA Warp backend for fast single-GPU
- Neon backend for grid refinement / multi-resolution
- Custom collision models, boundary conditions, LES (Smagorinsky)
- STL/mesh import (trimesh, numpy-stl) for flow over complex geometry
- 2D/3D LBM: D2Q9, D3Q19, D3Q27 lattices

## Useful bits

- Differentiability is real and exercised: a May 29 2026 commit (#156) adds a JAX differentiable-LBM inverse-problem example, and the paper frames the design around JAX autodiff through LBM kernels + boundary conditions. Differentiability lives on the JAX backend specifically; the Warp/Neon backends are for raw performance, not JAX-AD.
- Multi-backend architecture worth borrowing as a pattern: same model code runs on JAX (CUDA13/TPU, distributed multi-GPU, differentiable) or Warp (fastest single-GPU) or Neon (grid refinement) — directly relevant to our project's interest in the GPU/differentiable physics landscape (Warp, JAX-FEM, Taichi/DiffTaichi).
- Install is clean for our no-sudo conda+JAX stack: pip with extras — `pip install xlb[cuda]` pulls jax[cuda13]>=0.8.2; core deps numpy>=2.1, pydantic>=2.9, trimesh, pyvista; Python>=3.11. No conda-forge package. jax[cuda13] is forward-compatible with Blackwell sm_120, so it should run on the RTX 5080.
- No thermal/Boussinesq support today: natural convection is absent and 'Fluid-Thermal Simulation Capabilities' is explicitly a roadmap WIP — so XLB cannot directly model the venetian-blind buoyancy problem out of the box.
- Closest existing analogue to our hand-rolled differentiable JAX FEM, but for fluids: a production differentiable JAX solver where jax.grad flows through the whole time-stepping loop — a reference design for the loop-2 differentiable/inverse-design goal in CFD.

## Reality check

Claims hold up. Not vaporware and not stale: peer-reviewed in CPC (2024), Apache-2.0, v0.3.1 released Jan 2026, commits through late May 2026, and the differentiability claim is concretely backed by a shipped inverse-problem example rather than just marketing. The honest caveats are scope, not credibility: (1) it's a single-phase isothermal LBM — no thermal/Boussinesq today, so the headline 'multi-physics' is potential (extensible framework) more than delivered; (2) differentiability is real only on the JAX backend, not the faster Warp/Neon backends, so there's a perf-vs-differentiability tradeoff; (3) Blackwell sm_120 support rests on jax[cuda13] forward-compat (likely fine) but the bundled…

## Relevance here

- **Window-shade (natural convection):** Limited for the immediate problem. The 2D venetian-blind window-shade task is fundamentally a Boussinesq natural-convection (buoyancy + thermal) problem, and XLB is currently isothermal with thermal coupling only on its roadmap. One could in principle implement a thermal/double-distribution LBM with a Boussinesq force on top of XLB's extensible collision/BC framework, but that is non-trivial resea…
- **Project:** High as a landscape/architecture reference and as a differentiable-CFD backbone. It is a peer-reviewed, actively maintained, Apache-2.0 example of exactly the thing the project studies: a GPU, differentiable, JAX-native physics solver usable for inverse design — directly comparable to JAX-FEM and complementary to Warp/Taichi. Strong candidate to have an LLM agent operate (loop 1) and to drive diff…

## Follow-up

- Trial the most relevant capability against our stack before relying on it.
