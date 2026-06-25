---
kind: repo
name: "Taichi (taichi-lang) / DiffTaichi"
url: https://github.com/taichi-dev/taichi
commit:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 3
status: skimmed
related_experiments: []
related_concepts:
  - gpu-differentiable-physics-simulation
tags: [gpu, dsl, differentiable, python, mpm, stale]
---

# Taichi (taichi-lang) / DiffTaichi

## Purpose

Taichi is a Python-embedded imperative DSL that JIT-compiles parallel kernels to CPU, CUDA, Vulkan, OpenGL and Metal, with built-in spatially-sparse data structures (SNodes) and compile-time source-code-transformation autodiff; it is the engine beneath DiffTaichi. All three headline claims verify: multi-backend GPU execution, reverse-mode differentiability, and broad portability are genuinely implemented and well-exercised in differentiable graphics/physics. The important caveat is trajectory, not capability — the maintainer confirmed in mid-2024 that major development effectively stopped around 2023 and the project is now in bugfix/compatibility maintenance, with the last release (v1.7.4) s…

## Shape

- **GPU:** Yes. Multi-backend: CUDA, Vulkan, OpenGL 4.3+, Apple Metal, CPU (x64/ARM). CUDA backend lowers kernels to PTX via LLVM/NVPTX and JIT-compiles through the CUDA driver.
- **Differentiable:** Yes, but constrained. Reverse-mode (and forward-mode) autodiff via compile-time source-code transformation + lightweight `ti.ad.Tape`; foundation under DiffTaichi (ICLR 2020). Hard limitation: does not checkpoint intermediate global-field values, enforcing a "no mutation after re…
- **Install:** pip (manylinux glibc 2.27+ x86-64 wheels, Python 3.9-3.13); `pip install taichi`. No first-party conda-forge package; no source build needed for our stack. No sudo required.
- **License:** Apache-2.0
- **Maintained:** stale-but-alive (maintenance mode). Latest release v1.7.4 on 2025-07-31; prior 1.7.3 on 2024-12-23. Maintainer Yuanming Hu stated (2024-06-14) that major feature development effectively halted ~Aug 2023; team is in bugfi…

Capabilities:
- Python-embedded imperative DSL that JIT-compiles parallel kernels to CPU/CUDA/Vulkan/Metal/OpenGL
- Built-in spatially-sparse data structures (SNodes) for sparse grids/voxels
- Reverse- and forward-mode autodiff via source-code transformation + ti.ad.Tape (DiffTaichi foundation)
- Megakernel fusion and struct-for parallelism over fields
- NumPy/PyTorch/Paddle external-array bridges
- Source-code-transformation reverse-mode autodiff over imperative GPU kernels (megakernel + two-scale local/global gradient scheme)
- 10 example differentiable simulators: 2D/3D elasticity (MPM/ChainQueen), mass-spring, rigid body, billiards, 3D Euler smoke fluid, height-field shallow water, differentiable water renderer with CNN
- Multi-backend execution via Taichi: CPU (x64/ARM), CUDA, Vulkan, Metal

## Useful bits

- Differentiable physics in Taichi means writing the solver as @ti.kernel routines and capturing them under `ti.ad.Tape`; gradients come from compile-time source transformation, NOT from a tensor library — there is no automatic checkpointing of global fields, so any field you differentiate through must obey 'no mutation after read' (DiffTaichi sidesteps this w…
- It is pip-installable into a no-sudo conda env (manylinux wheels, Py 3.9-3.13, Apache-2.0), but its CUDA backend reaches new GPUs via PTX JIT and is not validated on Blackwell sm_120 — verify a trivial `ti.init(arch=ti.cuda)` kernel runs on the RTX 5080 before investing, and keep the Vulkan backend as a fallback.
- Interop is one-way copies via from_numpy/to_numpy and from_torch/to_torch only; there is no native JAX or DLPack path, so Taichi cannot share an autodiff graph with our JAX FEM — treat it as a self-contained alternative engine, not a JAX accelerator.
- Source-transform autodiff design: Taichi differentiates imperative kernels with a 'two-scale' (local + global) automatic differentiation scheme, a useful template for our hand-rolled JAX FEM where we already do matrix-free CG — it shows how to get reverse-mode gradients through explicit time-stepping loops without taping every array op.
- The fluid examples are limited: a 3D Eulerian smoke/advection sim and a 2D height-field 'shallow water' surface sim. Neither solves Boussinesq buoyancy-driven natural convection, has thermal/energy coupling, or extracts a heat-transfer coefficient, so they are not a starting point for the venetian-blind problem.

## Reality check

Claims are accurate, not vaporware: GPU multi-backend and reverse-mode autodiff are genuinely implemented and battle-tested (DiffTaichi). The honest caveat is maturity-as-decline rather than overhype — the project is feature-frozen in maintenance mode, the autodiff has real structural limits, and Blackwell support is untested. It is a solid, real tool whose trajectory is flat, not rising.

## Relevance here

- **Window-shade (natural convection):** Moderate-low. Taichi/DiffTaichi has been used for differentiable fluids (MPM, Eulerian smoke, lattice-Boltzmann), so a 2D Boussinesq natural-convection solver is expressible. But it is a from-scratch-kernel toolkit, not a CFD package: no built-in incompressible Navier-Stokes/buoyancy solver, no boundary-condition machinery, and the autodiff 'no-mutation-after-read' rule makes differentiable steady…
- **Project:** Useful as landscape/context. It is the canonical reference point for 'GPU + differentiable physics DSL' and the literal foundation of DiffTaichi, so it anchors the differentiable-simulation map even though it is unlikely to be a primary solver on a JAX-centric, Blackwell stack. Best treated as a comparison baseline and a source of differentiable-sim patterns (megakernels, sparse SNodes, tape-based…

## Follow-up

- Trial the most relevant capability against our stack before relying on it.
