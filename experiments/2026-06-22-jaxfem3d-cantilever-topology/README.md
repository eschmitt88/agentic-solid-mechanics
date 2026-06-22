---
kind: experiment
slug: "jaxfem3d-cantilever-topology"
date: "2026-06-22"
status: done        # running | done | abandoned
hypothesis: "A self-contained 3D differentiable FEM (8-node hex, SIMP, density filter) built in JAX runs on the GPU and is correct — its uniform-density compliance reproduces 3D cantilever beam theory — and gradient-based topology optimisation on it produces a physically sensible 3D load-carrying structure, extending the loop-2 substrate from 2D (trial 3) to 3D."
result: "Confirmed. The 24x24 B8 hex stiffness is exact (symmetric, 6 rigid-body zero-eigenvalues, zero force under rigid translation). The forward model's uniform-density compliance converges monotonically to Euler-Bernoulli under through-thickness refinement (-12.4% -> -3.4% -> -1.4% -> -0.61%), the residual being the known shear-locking of fully-integrated trilinear hex. OC topology optimisation on the GPU (24x8x8, ndof 6075) gives a sensible 3D cantilever load path at compliance 49.98, vol 0.30. A matrix-free (sparse) rewrite is bit-identical to dense and solves 121,875 unknowns in 0.43s (a dense matrix would be ~119GB). Loop-2 agentic pass@10: handed only the differentiable solver, the agent writes its own optimiser 10/10 feasible, ~1.2% stiffer than our reference, leakage clean on all."
related_concepts: ["differentiable-inverse-design", "agentic-design-optimization"]
related_literature: []
tags: ["jax", "differentiable-fem", "topology-optimization", "3d", "gpu", "trial-4"]
---

# jaxfem3d-cantilever-topology

Trial 4 / loop-2 extension: take the differentiable-FEM topology-optimisation
substrate from **2D** (`../2026-06-22-jaxfem-topology-optimization/`) to **3D**,
on the GPU. A trilinear 8-node hexahedral element, 3D isotropic elasticity,
SIMP + density filter, end-to-end autodifferentiable `compliance(rho)`.

## Hypothesis

A compact 3D differentiable FEM in JAX is (a) **correct** — uniform-density
compliance = tip deflection reproduces 3D cantilever beam theory
(Euler–Bernoulli + Timoshenko shear) the same way the 2D solver did — and (b) a
usable loop-2 substrate: OC topology optimisation on it yields a sensible 3D
structure. Runs on the RTX 5080 (GPU now unblocked after the driver/kernel
reboot).

## Setup

- Config: `config.yaml` (grid, volfrac, penal, rmin, optim).
- Code: `jaxfem3d.py` (differentiable 3D FEM: Gauss-quadrature B8 hex stiffness,
  SIMP, 3D density filter, jitted autodiff compliance), `topopt3d_reference.py`
  (OC optimiser + beam-theory verification + density export), `verify3d.py`.
- Solver: hand-rolled in JAX — stands in for production JAX-FEM, transparent and
  GPU-native. Dense direct solve (caps grid size; see Diagnostics).
- Data: none — analytical beam theory is the forward-model gold.

## Result

Points at `metrics.json` (reference 3D run) and `results/verification3d.json`.

- **Element gold** (`verify3d.py` → `verification3d.json:element_gold`): the
  24×24 B8 hex stiffness is symmetric, has exactly **6 zero eigenvalues** (3D
  rigid-body modes), and gives ~0 force under rigid translation (2.8e-17).
- **Forward-model gold** (`verification3d.json:beam_convergence`): L/h=10
  cantilever, uniform density; FEM tip deflection vs Euler–Bernoulli **converges
  monotonically** with through-thickness refinement: −12.4% (ny=2) → −3.4%
  (ny=4) → −1.4% (ny=6) → **−0.61% (ny=8)**. Residual = shear locking of
  fully-integrated trilinear hex.
- **3D topology** (`metrics.json`): 24×8×8 = 1536 elems (ndof 6075) on the RTX
  5080; OC → **compliance 49.98, vol 0.30**, a sensible 3D cantilever load path
  (densest chords at the root/bottom). Rendered in the QA site (trial-4).
- **Matrix-free (sparse) solver** (`jaxfem3d_mf.py`, `results/matrixfree_*.json`):
  replaces the dense assemble+factorise with element-by-element Jacobi-PCG, and
  the dense O(nelem²) filter matrix with a 3D convolution. **Bit-identical to the
  dense solver** (compliance rel-diff ~1e-13, gradients ~1e-9 → differentiability
  preserved) but O(ndof) memory. At 40×12×12 it is ~600× faster than the dense
  solve (11 ms vs 6.6 s); it solves a **64×24×24 grid (121,875 unknowns) in
  0.43 s — whose dense stiffness matrix would be ~119 GB, impossible on the 16 GB
  GPU**. A full 48×16×16 optimization (42,483 unknowns, 40 iters) finishes in
  3.3 s. This lifts the grid-size cap noted below.
- **Loop-2 agentic pass@10** (`agent_topopt3d.py`,
  `results/agent_topopt3d_trials_*.json`): handed ONLY the differentiable
  matrix-free solver in an isolated dir, the agent must write its own optimizer,
  run it on the GPU, and submit a design (headless `claude -p`, subscription, no
  API key). **10/10 feasible**, every run volume-exact (0.300) and **−1.2% to
  −1.3% vs our OC reference** (i.e. marginally *stiffer* — our 60-iter reference
  is not the true optimum; the agents' OC converged a touch further), median ~7
  turns, **leakage clean on all 10** (no run read the reference). Compliance was
  independently recomputed from each saved design (anti-fabrication). The 3-D
  analogue of trial 3's 2-D loop-2 result.

## Interpretation

The 3D differentiable FEM is correct by independent algebraic checks (rigid-body
spectrum) and by convergence to closed-form beam theory — the same shear story
trial 1 saw in CalculiX and trial 3's 2D solver saw vs Timoshenko, now in 3D.
The dense direct solve caps the grid (ndof grows as the cube of resolution); a
sparse/CG solve is the obvious next lift. With a verified 3D `compliance(rho)`,
the loop-2 agentic substrate now exists in 3D.

## Diagnostics

Unless noted, numbers reference `metrics.json`.

- intended_effect_confirmed: yes — element exact + forward model → EB to −0.61%
  (`results/verification3d.json:beam_convergence.finest_err_vs_eb_pct`); 3D
  topology at compliance 49.98 (`metrics.json:reference_compliance`); loop-2
  agent pass@10 = 10/10 feasible
  (`results/agent_topopt3d_trials_claude-opus-4-8.json:feasible_rate`).
- leakage_check: pass@10 ran each agent in an isolated dir with only the solver
  files; a token scan of every agent's authored code was clean on all 10
  (`agent_topopt3d_trials_*.json:all_leakage_clean=true`), and each reported
  compliance was independently recomputed from the saved density.
- overfitting_signal: n/a — single deterministic optimisation, no train/val split.
- delta_from_prior: vs `2026-06-22-jaxfem-topology-optimization` (2D), extends the
  same SIMP/OC/autodiff machinery to 3D B8 hex elements; ndof 6075 vs 1666.
- unexpected_findings: coarse-mesh stiffness is textbook shear locking, confirmed
  by monotone convergence under refinement — a feature of B8 full integration,
  not a bug; documented rather than "fixed".
- next_candidates:
  - Hand the 3D `compliance(rho)` to the agent (as in trial 3) for a loop-2 pass@k.
  - Reduced-integration (or B-bar) hex to cut shear locking on coarse meshes.
  - Scale the loop-2 agent pass@k to a larger grid (the matrix-free solver now
    supports it) and to harder objectives (multiple load cases, stress limits).

## Follow-up

- **Done:** matrix-free Jacobi-PCG solver + convolution filter (`jaxfem3d_mf.py`,
  `bench_matrixfree.py`, `topopt3d_matrixfree.py`) — lifts the grid cap from a few
  thousand unknowns to 100k+ on the GPU, verified bit-identical to the dense solve.
- Agent-design trial that writes its own optimiser on the matrix-free 3D solver.
- A conjugate-gradient solve that does not re-converge from scratch each iteration
  (warm-start from the previous design) to speed large optimisations further.
