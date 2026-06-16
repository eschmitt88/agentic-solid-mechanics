---
kind: candidates
topic: "LLM agents operating physics simulators for solid mechanics — agent-as-CAE-operator, differentiable FEM, inverse design, and the open-source solvers best suited to autonomous agent control"
discovered: 2026-06-16
source: discover
n_requested: 15
n_returned: 15
---

Triage merges three parallel sweeps: (A) agentic-LLM-operates-a-simulator
literature, (B) differentiable-FEM / inverse-design / surrogates, (C) the
open-source solver landscape for autonomous agent control. Ranked for this
project's two loops (agent-as-operator + differentiable/inverse-design), single
physics first. Recent arXiv IDs (2026) were WebFetch-checked by the sweep
agents; `/fetch-paper` will surface any that don't resolve.

## 1. MechAgents: LLM multi-agent collaborations can solve mechanics problems

- url: https://arxiv.org/abs/2311.08166
- type: paper
- summary: A multi-agent LLM team (planner, coder, executor, critic) autonomously writes, runs, and self-corrects finite element code to solve classical elasticity problems across varied BCs, geometries, and constitutive laws.
- reason: The seminal demonstration of this project's exact agent-operates-a-FEM-solver loop in solid mechanics — the direct conceptual ancestor.

## 2. ALL-FEM: Agentic LLMs Fine-tuned for Finite Element Methods

- url: https://arxiv.org/abs/2603.21011
- type: paper
- summary: An agentic system pairing fine-tuned open-weight LLMs (3B–120B) with a multi-agent orchestrator to formulate PDEs, generate/debug FEniCS code, and visualize results across solid, fluid, and multiphysics benchmarks.
- reason: Directly targets agent-driven solid-mechanics FEM with open-weight models on a 39-problem benchmark (incl. plasticity, FSI) — closest published analog to our setup, and FEniCS-based like our likely operator solver.

## 3. FEM-Bench: A Structured Scientific Reasoning Benchmark for Code-Generating LLMs

- url: https://arxiv.org/abs/2512.20732
- type: paper
- summary: A benchmark of computational-mechanics FEM and matrix-structural-analysis tasks scoring LLM-generated code; reliable on foundational routines but failing on geometric nonlinearity, buckling eigenvalue problems, and discriminative unit tests.
- reason: Ready-made mechanics-focused evaluation harness and a clear map of where simulator-operating agents currently break — informs our HCE benchmark design.

## 4. JAX-FEM: A differentiable GPU-accelerated 3D FEM solver for inverse design

- url: https://arxiv.org/abs/2212.00964
- type: paper
- summary: Pure-Python differentiable 3D FEM on JAX computing sensitivities by autodiff; demonstrates GPU-accelerated nonlinear topology optimization and inverse design.
- reason: The canonical differentiable-FEM solver for solid mechanics and the most likely backbone for the agent-orchestrated gradient-based inverse-design loop.

## 5. deepmodeling/jax-fem (repository)

- url: https://github.com/deepmodeling/jax-fem
- type: repo
- summary: Actively maintained reference JAX-FEM: differentiable FE assembly/solvers, mature solid mechanics (linear elasticity, hyperelasticity, macro & crystal plasticity), GPU-native, ships an LLM-oriented "JAX-FEM Express" interface.
- reason: The practical primary artifact for the differentiable loop — the codebase the agent calls and differentiates through; GPL-3.0, GPU, autodiff.

## 6. FEniCS/dolfinx (FEniCSx — repository)

- url: https://github.com/FEniCS/dolfinx
- type: repo
- summary: Next-gen FEniCS FE environment with a Python/UFL weak-form interface for PDEs incl. solid mechanics; conda/docker install; differentiable via dolfin-adjoint/pyadjoint.
- reason: Top candidate for the agent-as-operator loop — pure-Python scripting, abundant tutorials, mature solid mechanics; the solver the agent learns and drives.

## 7. Foam-Agent: Towards Automated Intelligent CFD Workflows

- url: https://arxiv.org/abs/2505.04997
- type: paper
- summary: Composable multi-agent LLM framework with RAG and dependency-aware scheduling driving the full OpenFOAM pipeline (mesh, config, HPC scripts, run, post-proc) from one prompt; 88.2% execution success on 110 tasks.
- reason: Strongest end-to-end "agent operates a real simulator" architecture; its retrieval/scheduling/MCP patterns transfer directly to a solid-mechanics solver.

## 8. Self-Refining Topology Optimization via an LLM-Based Multi-Agent Framework (TopOptAgents)

- url: https://arxiv.org/abs/2605.23273
- type: paper
- summary: Six collaborating LLM agents iteratively formulate, validate, code, execute, and quality-assess topology-optimization problems through self-refinement cycles.
- reason: A concrete instance of LLM-driven, simulator-grounded structural design optimization — the design-loop variant of our goal, bridging both project loops.

## 9. Toward Engineering AGI: Benchmarking Engineering Design Capabilities of LLMs (EngDesign)

- url: https://arxiv.org/abs/2509.16204
- type: paper
- summary: Simulation-based benchmark evaluating LLM engineering-design synthesis across nine disciplines by running designs through simulators rather than checking static answers.
- reason: Establishes simulation-in-the-loop scoring — directly applicable evaluation paradigm for an agent that designs against a physics simulator (grounds our HCE).

## 10. NVIDIA Warp (repository)

- url: https://github.com/NVIDIA/warp
- type: repo
- summary: Apache-2.0 Python framework JIT-compiling kernels to GPU with reverse-mode autodiff, interoperable with PyTorch/JAX; `warp.fem` module supports elasticity/diffusion/fluid FEM.
- reason: Easiest GPU install (`pip install warp-lang`) and a clean differentiable Python API — the general differentiable-GPU fallback when custom kernels beat a turnkey structural solver.

## 11. CalculiX (repository / project)

- url: https://github.com/Dhondtguido/CalculiX
- type: repo
- summary: GPL-2 implicit/explicit solver using the Abaqus .inp text format; strong nonlinear solid mechanics (contact, plasticity, large deformation); single `ccx` CLI, apt/conda install.
- reason: Plain-text Abaqus-style decks + one CLI + .frd output make problem generation and result parsing maximally mechanical for an agent — the cleanest "text-deck operator" baseline.

## 12. SfePy (repository)

- url: https://github.com/sfepy/sfepy
- type: repo
- summary: New-BSD lightweight pure-Python FE PDE solver (v2026.1) supporting linear and nonlinear elasticity; trivial `pip install sfepy`.
- reason: Lowest-friction option for an agent to install, drive, and parse end-to-end (CPU-only) — strong candidate for the first "hello-world" agentic trials.

## 13. Implicit differentiation with second-order derivatives and benchmarks in FE-based differentiable physics

- url: https://arxiv.org/abs/2505.12646
- type: paper
- summary: Extends FE differentiable physics (JAX-FEM lineage) with exact implicit second-order derivatives; benchmarks Newton-CG with exact Hessians on nonlinear PDE-constrained inverse problems.
- reason: Second-order sensitivities enable faster, more robust gradient-based inversion — directly upgrades the inverse-design loop's optimizer.

## 14. Differentiable programming across the PDE and Machine Learning barrier

- url: https://arxiv.org/abs/2409.06085
- type: paper
- summary: Firedrake/pyadjoint framework composing differentiable FE PDE solvers with ML models into end-to-end-differentiable coupled systems.
- reason: Canonical recent route to differentiating through FEM adjoints while interfacing neural surrogates — the hybrid an agentic loop would orchestrate.

## 15. AutoNumerics: An Autonomous, PDE-Agnostic Multi-Agent Pipeline for Scientific Computing

- url: https://arxiv.org/abs/2602.17607
- type: paper
- summary: Multi-agent framework where LLMs act as numerical architects generating transparent solver code from first principles, verified on a 200-PDE suite (1D–5D elliptic/parabolic/hyperbolic).
- reason: Demonstrates code-from-scratch (not just library-calling) numerical-PDE agents with built-in verification — an alternative design axis vs. the library-driving approach.

## Notes — solver landscape & honorable mentions

Full solver comparison from sweep C (license / interface / install / GPU /
differentiability):

| Solver | License | Interface | Install | GPU | Diff/adjoint |
|---|---|---|---|---|---|
| FEniCSx/DOLFINx | LGPL-3 | Python/UFL | conda/docker easy | partial (cuda-dolfinx) | via dolfin-adjoint |
| JAX-FEM | GPL-3 | Python | pip/source moderate | **yes** | **yes (autodiff)** |
| NVIDIA Warp | Apache-2 | Python | pip easy | **yes** | **yes** |
| CalculiX | GPL-2 | .inp text deck | apt/conda easy | no | no |
| SfePy | BSD | Python | pip easy | no | no |
| Kratos Multiphysics | BSD | Python+JSON | pip easy | no | via adjoint (structural) |
| MOOSE | LGPL-2.1 | HIT text deck | conda+build hard | partial (MFEM/NEML2) | forward AD |

**Sweep recommendation:** FEniCSx for the agent-as-operator loop (clean Python,
best docs), with CalculiX (text decks) and SfePy (lowest friction) as
alternative baselines; **JAX-FEM** for the differentiable/inverse-design loop
(GPU-native + autodiff + mature solid mechanics), Warp as the custom-kernel
fallback. All pair with **Gmsh** (meshing), **meshio** (formats), **PyVista**
(post-processing) — an all-Python stack an agent can drive end to end.

Honorable mentions not in the top 15: VFEAgent (arXiv 2605.28978 — multimodal
agent driving Abaqus from drawings; commercial solver, lower fit), DiffTaichi
(arXiv 1910.00935 — foundational differentiable sim), physics-encoded FNO for
stress fields (arXiv 2408.15408), Finite-PINN (arXiv 2412.09453 — PINN for
finite solid geometries), Kratos Multiphysics, MOOSE.
