---
kind: candidates
topic: "GPU-accelerated & differentiable physics-simulation landscape (Taichi/Warp, JAX-CFD, GPU lattice-Boltzmann, OpenFOAM-GPU, differentiable physics engines, physics-ML) — and the most practical GPU/differentiable route to a 2D natural-convection h(slat-angle) study"
discovered: 2026-06-25
source: ultracode-workflow
n_requested: many
n_returned: 93
disposition: ingested
---

Ultracode workflow (31 agents): 12 parallel discovery sweeps -> 93 unique items
-> top 18 adversarially deep-dived (GPU? differentiable? maintained? installable on
our no-sudo conda + JAX/CUDA Blackwell stack? license?) -> synthesis. Keepers ingested
as literature notes; see `mocs/gpu-differentiable-physics-simulation.md` for the full
landscape + window-shade recommendation. This file archives the ranked sweep.

## Ingested (literature notes written)

XLB · PhiFlow · Lineax · Optimistix · PyFR · MuJoCo MJX · Taichi/DiffTaichi ·
NVIDIA Warp (updated) · JAX-FEM (updated) · Newbury 2024 diff-sim review ·
Sapienza 2024 diff-programming review · DiFVM 2026 · Zheng 2022 venetian blinds.

## Full ranked sweep (top 35)

1. **XLB** (rel 5, library, gpu=jax, diff=yes) — Differentiable, massively parallel lattice-Boltzmann library in Python built on JAX (with a NVIDIA Warp backend), distributed multi-GPU, designed for ML integration.
   https://github.com/Autodesk/XLB
2. **NVIDIA Warp** (rel 5, framework, gpu=cuda, diff=yes) — Python framework that JIT-compiles kernels to GPU and auto-generates forward+backward (reverse-mode AD) passes for differentiable simulation, with a built-in warp.fem weak-form FEM module.
   https://github.com/NVIDIA/warp
3. **warp.fem (module within NVIDIA Warp)** (rel 5, library, gpu=cuda, diff=yes) — Weak-form finite-element layer inside Warp: define function spaces and integrands in Python on 2D/3D grids/meshes, assemble on GPU, and differentiate end-to-end.
   https://nvidia.github.io/warp/
4. **XLB (Accelerated Lattice Boltzmann)** (rel 5, library, gpu=multi-backend, diff=yes) — JAX/Warp/Neon-backed differentiable LBM library built explicitly for physics-based ML and inverse design — the single best fit for this project's stack.
   https://github.com/Autodesk/XLB
5. **PhiFlow (ΦFlow)** (rel 5, framework, gpu=multi-backend, diff=yes) — Differentiable PDE/physics framework focused on grid-based incompressible Navier-Stokes (plus heat flow, SPH/FLIP, mesh), backend-agnostic (JAX/PyTorch/TF/NumPy), GPU, MIT.
   https://github.com/tum-pbs/PhiFlow
6. **JAX-FEM** (rel 5, library, gpu=jax, diff=yes) — Differentiable, GPU-accelerated 3D finite element solver in pure Python/JAX for solid mechanics, with automatic-differentiation-driven inverse/design problems.
   https://github.com/deepmodeling/jax-fem
7. **Lineax** (rel 5, library, gpu=jax, diff=yes) — JAX library unifying linear solves and linear least-squares behind an autodifferentiable, user-extensible linear-operator API (matrix-free supported).
   https://github.com/patrick-kidger/lineax
8. **Optimistix** (rel 5, library, gpu=jax, diff=yes) — Modular JAX+Equinox nonlinear optimisation: minimisation, nonlinear least-squares, root-finding, and fixed-point iteration.
   https://github.com/patrick-kidger/optimistix
9. **A Review of Differentiable Simulators (Newbury et al.)** (rel 5, survey, gpu=n/a, diff=yes) — Comprehensive 2024 survey of differentiable physics simulators: foundations, design trade-offs, a catalog of open-source simulators, and applications across computational physics, robotics, and ML.
   https://arxiv.org/abs/2407.05560
10. **PyFR** (rel 5, framework, gpu=multi-backend, diff=no) — Python-based high-order Flux Reconstruction (FR/DG-family) scale-resolving CFD framework that JIT-generates kernels for NVIDIA/AMD/Intel/Apple GPUs from a Mako DSL.
   https://github.com/PyFR/PyFR
11. **DiFVM** (rel 5, paper, gpu=jax, diff=yes) — First GPU-accelerated, end-to-end differentiable finite-volume CFD solver operating natively on unstructured polyhedral meshes, built in JAX.
   https://arxiv.org/abs/2603.15920
12. **MuJoCo MJX (MuJoCo XLA)** (rel 5, library, gpu=jax, diff=yes) — JAX-native, fully differentiable reimplementation of MuJoCo expressing rigid-body dynamics, collision, constraints, and integration as XLA primitives, runnable on NVIDIA/AMD GPUs, Apple Silicon, and T
   https://mujoco.readthedocs.io/en/stable/mjx.html
13. **XLB (differentiable Lattice-Boltzmann)** (rel 5, library, gpu=multi-backend, diff=yes) — Differentiable, GPU-accelerated Lattice-Boltzmann CFD library with both JAX and NVIDIA Warp backends, aimed at scalable flow simulation and ML integration.
   https://github.com/Autodesk/XLB
14. **Differentiable Programming for Differential Equations: A Review (Sapienza, Rackauckas et al.)** (rel 5, survey, gpu=n/a, diff=yes) — 75-page review of how to differentiate ODE/PDE solvers: forward vs reverse AD, discrete vs continuous adjoints, and the trade-offs, across Julia/Python/C++ ecosystems.
   https://arxiv.org/abs/2406.09699
15. **Zheng et al. (2022) — interior venetian blinds at a full-scale window glazing** (rel 4.5, paper, gpu=none, diff=no) — 2D CFD + experimental study of how interior venetian-blind slat angle and blind-window spacing govern natural-convective heat exchange at a window.
   https://www.sciencedirect.com/science/article/abs/pii/S1359431122013928
16. **Taichi (taichi-lang)** (rel 4, dsl, gpu=multi-backend, diff=yes) — Python-embedded imperative DSL that JIT-compiles parallel kernels to CPU/CUDA/Vulkan/Metal/OpenGL, with built-in sparse data structures (SNodes) and reverse-mode autodiff; the foundation under DiffTai
   https://github.com/taichi-dev/taichi
17. **DiffTaichi** (rel 4, repo, gpu=multi-backend, diff=yes) — Collection of 10 differentiable physical simulators (2D/3D elasticity, MPM, mass-spring, rigid body, 3D fluid, height-field water, billiards) built on Taichi's source-transform autodiff, accompanying 
   https://github.com/taichi-dev/difftaichi
18. **Taichi differentiable programming (mainline autodiff)** (rel 4, library, gpu=multi-backend, diff=yes) — Taichi's source-transformation reverse-mode AD via ti.ad.Tape() (single 0D scalar output) and kernel.grad() (manual seeding for multiple outputs), plus forward mode (ti.ad.FwdMode) for inverse/few-out
   https://docs.taichi-lang.org/docs/differentiable_programming
19. **JAX-MPM** (rel 4, paper, gpu=jax, diff=yes) — Differentiable, GPU-accelerated material-point-method framework in JAX for large-deformation Lagrangian simulation and geophysical inverse modeling, supporting joint physics+DL training.
   https://arxiv.org/abs/2507.04192
20. **NVIDIA PhysicsNeMo (formerly Modulus)** (rel 4, framework, gpu=cuda, diff=yes) — NVIDIA's flagship open-source PyTorch Physics-ML framework spanning PINNs, neural operators (FNO/DeepONet/DoMINO), and GNN mesh surrogates (MeshGraphNet), with a symbolic PDE module (physicsnemo.sym) 
   https://github.com/NVIDIA/physicsnemo
21. **JAX-CFD** (rel 4, library, gpu=jax, diff=yes) — Google's experimental JAX library for incompressible Navier-Stokes (finite-volume on staggered grids + pseudospectral), 2D, GPU/TPU and fully differentiable, but now unmaintained.
   https://github.com/google/jax-cfd
22. **JAX-Fluids (JAXFLUIDS) 2.0** (rel 4, framework, gpu=jax, diff=yes) — Fully-differentiable high-order JAX CFD solver for compressible single- and two-phase flows, CPU/GPU/TPU, with HPC (multi-device) support.
   https://github.com/tumaer/JAXFLUIDS
23. **Diffrax** (rel 4, library, gpu=jax, diff=yes) — Foundational JAX numerical differential-equation library (explicit/implicit/stiff/stochastic), fully autodiff and GPU/TPU; the time-integration backbone for method-of-lines PDE/CFD pipelines.
   https://github.com/patrick-kidger/diffrax
24. **Equinox** (rel 4, library, gpu=jax, diff=yes) — Filtered-transformation JAX library that represents parameterised functions (and neural nets) as PyTrees, the substrate the rest of the Kidger stack builds on.
   https://github.com/patrick-kidger/equinox
25. **jax-am** (rel 4, framework, gpu=jax, diff=yes) — GPU-accelerated differentiable additive-manufacturing multiphysics toolbox bundling DEM, Lattice-Boltzmann, CFD (Navier-Stokes with temperature), and phase-field methods in JAX.
   https://github.com/tianjuxue/jax-am
26. **FluidX3D** (rel 4, framework, gpu=multi-backend, diff=no) — OpenCL LBM with first-class thermal/Boussinesq natural-convection support — directly relevant physics, but C++/OpenCL only and not differentiable, so a reference/validation tool rather than an in-loop
   https://github.com/ProjectPhysX/FluidX3D
27. **NekRS** (rel 4, framework, gpu=multi-backend, diff=no) — GPU-oriented spectral-element Navier-Stokes / thermal-fluids solver (successor to Nek5000, forked from libParanumal) for incompressible and low-Mach flow plus scalar transport.
   https://github.com/Nek5000/nekRS
28. **Diff-FlowFSI** (rel 4, paper, gpu=jax, diff=yes) — GPU-optimized, fully differentiable JAX CFD/FSI platform: FVM on staggered grids, Chorin projection, immersed boundary for rigid/flexible structures.
   https://arxiv.org/abs/2505.23940
29. **Genesis (genesis-world)** (rel 4, framework, gpu=multi-backend, diff=partial) — Unified Pythonic multi-physics engine (rigid, FEM, MPM, SPH, PBD, IPC, Stable-Fluids) with a coupler, photoreal renderer, and a multi-backend compiler (CUDA/ROCm/Metal/Vulkan).
   https://github.com/Genesis-Embodied-AI/Genesis
30. **neuraloperator (FNO / TFNO / SFNO / GINO)** (rel 4, library, gpu=cuda, diff=yes) — The canonical reference implementation of Fourier Neural Operators and relatives (Tensor-FNO, Spherical-FNO, GINO geometry-aware, GNO, Codano) for learning resolution-invariant maps between function s
   https://github.com/neuraloperator/neuraloperator
31. **jinns** (rel 4, library, gpu=jax, diff=yes) — JAX-native PINN library focused on inverse problems and meta-model learning, with refined architectures (SeparablePINN, HyperPINN), built on equinox/optax.
   https://gitlab.com/mia_jinns/jinns
32. **PETSc (petsc4py)** (rel 4, library, gpu=multi-backend, diff=no) — Mature MPI-parallel sparse-linear/nonlinear/timestepping solver suite with broad multi-vendor GPU backends and a first-class Python interface (petsc4py); the de-facto solver layer under Firedrake/FEni
   https://petsc.org/release/overview/gpu_roadmap/
33. **cuDOLFINx (cuda-dolfinx)** (rel 4, library, gpu=cuda, diff=no) — Drop-in CUDA assembly extension for FEniCSx/DOLFINx that makes existing UFL/FEniCSx codes GPU-accelerated with minimal changes — the most practical near-term path to GPU FEM from a Python conda stack.
   https://github.com/bpachev/cuda-dolfinx
34. **GPU-accelerated simulations of turbulence: Review of current applications and future perspectives** (rel 4, survey, gpu=multi-backend, diff=no) — 2026 Phys. Rev. Fluids review of GPU-accelerated turbulence/CFD solvers (NekRS, OpenSBLI v3, FluTAS, etc.), porting/optimization bottlenecks, and the road to exascale-on-GPU.
   https://journals.aps.org/prfluids/abstract/10.1103/vz9c-bbzm
35. **PICT — Differentiable GPU PISO solver (TUM, Thuerey group)** (rel 4, library, gpu=cuda, diff=yes) — Differentiable, GPU-accelerated multi-block PISO solver for INCOMPRESSIBLE Navier-Stokes in PyTorch, built for simulation-coupled learning (e.g. learned turbulence closures).
   https://github.com/tum-pbs/PICT
