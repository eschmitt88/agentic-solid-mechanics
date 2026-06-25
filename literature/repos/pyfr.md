---
kind: repo
name: "PyFR"
url: https://github.com/PyFR/PyFR
commit:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 2
status: skimmed
related_experiments: []
related_concepts:
  - gpu-differentiable-physics-simulation
tags: [gpu, cfd, high-order, flux-reconstruction, multi-backend, python]
---

# PyFR

## Purpose

PyFR is an open-source, Python-based high-order CFD framework that solves advection-diffusion (compressible and artificial-compressibility incompressible Navier-Stokes) problems using the Flux Reconstruction approach, a unifying high-order DG-family scheme on mixed unstructured curved meshes. Its defining feature is runtime JIT generation of compute kernels from a Mako-based domain-specific language, giving a single codebase that targets NVIDIA (CUDA), AMD (HIP), multi-vendor (OpenCL), Apple (Metal), and CPU (OpenMP) backends, with demonstrated MPI scaling to thousands of GPUs. It is a forward, scale-resolving (DNS/LES) turbulence solver from the Vincent/Witherden group — actively maintained…

## Shape

- **GPU:** Yes — genuinely multi-backend, not vaporware. Backends: CUDA (NVIDIA, compute capability >=3.5, CUDA >=11.4), HIP/ROCm (AMD, ROCm >=6.4.1), OpenCL (multi-vendor), Metal (Apple silicon), OpenMP (CPU). Kernels are JIT-generated from a Mako-based DSL at runtime, which is the documen…
- **Differentiable:** No. The claim "differentiable=no" is correct. No AD, adjoint, or sensitivity/gradient capability found in repo, docs, or literature. PyFR is a forward scale-resolving (DNS/LES/ILES) solver only; differentiable/inverse design is out of scope.
- **Install:** pip (PyFR 3.1 installable via pip + virtualenv) or source (setup.py / PYTHONPATH). No official conda-forge package found. Hard dependency on Python 3.11+.
- **License:** New BSD (3-clause); docs under CC-BY-4.0. Verified on the GitHub repo.
- **Maintained:** Active. v3.1 released 2026-03-13; latest develop-branch commit 2026-06-25 ("Allow the explicit PI controller to infer dt-max automatically," by FreddieWitherden). ~2,331 commits. Imperial College / Stanford (Vincent, Wit…

Capabilities:
- High-order Flux Reconstruction (FR), a unifying DG-family scheme, on mixed unstructured curved meshes (hex/tet/prism/pyramid, 2D and 3D)
- Compressible Euler / Navier-Stokes solver for scale-resolving DNS/LES/ILES of turbulent flows
- Incompressible Navier-Stokes via artificial compressibility with dual time stepping and P-multigrid acceleration
- JIT kernel generation from a Mako DSL targeting CUDA / HIP / OpenCL / Metal / OpenMP backends
- Massively parallel: MPI-distributed, demonstrated on thousands of GPUs (GH200, MI250X)

## Useful bits

- The Mako-DSL-to-JIT-kernel pattern is the architectural lesson: one numerical kernel description, JIT-specialized at runtime per backend (CUDA/HIP/OpenCL/Metal/OpenMP) — a clean alternative to our hand-rolled-per-backend or JAX-only approaches, and a concrete reference for the 'code-from-scratch numerical solver' design axis.
- PyFR ships an incompressible Navier-Stokes solver via artificial compressibility + dual time stepping + P-multigrid (Loppi et al., CPC 2018), GPU-friendly because the work casts to matmul-bound operations — but it is isothermal, with no bundled energy equation or Boussinesq buoyancy, so natural convection requires custom coupling.
- Credible, peer-reviewed, HPC-grade reference solver (CPC 2014; v2.0.3 CPC 2025 scaling to 2048 GH200 / 1024 MI250X) — suitable as a high-fidelity forward oracle for an agent-operates-a-real-simulator task, but offers nothing for gradient-based inverse design (no AD/adjoint).

## Reality check

The headline claims are real and well-substantiated, not hype. It genuinely JIT-generates GPU kernels for NVIDIA/AMD/Intel/Apple from a Mako DSL, is high-order FR/DG, and is demonstrably maintained and HPC-scaled by a credible academic group (peer-reviewed CPC papers, 2014 through 2025). The one caveat against the project description is scope creep around our use case: it is a turbulence-focused forward solver, so its relevance to a differentiable-inverse-design project is limited to being a high-fidelity reference/oracle, not an optimizable component. For the specific window-shade natural-convection problem, the gap is that PyFR's incompressible solver is isothermal and does not bundle Bous…

## Relevance here

- **Window-shade (natural convection):** Low-to-moderate. PyFR has an incompressible artificial-compressibility solver and high-order accuracy that could in principle do a 2D venetian-blind cavity, but it lacks a built-in energy equation / Boussinesq buoyancy coupling, so the core natural-convection physics (h vs slat angle) is not available out of the box and would need a custom coupled scalar-transport + buoyancy source implementation.…
- **Project:** Moderate as a landscape datapoint and a candidate forward-simulation oracle for loop 1 (agent-operates-a-simulator); not relevant to loop 2 (no autodiff, no JAX interop). Strong exemplar of GPU-portable JIT-DSL high-order CFD design.

## Follow-up

- Trial the most relevant capability against our stack before relying on it.
