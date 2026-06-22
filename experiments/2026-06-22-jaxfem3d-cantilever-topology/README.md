---
kind: experiment
slug: "jaxfem3d-cantilever-topology"
date: "2026-06-22"
status: done        # running | done | abandoned
hypothesis: "A self-contained 3D differentiable FEM (8-node hex, SIMP, density filter) built in JAX runs on the GPU and is correct — its uniform-density compliance reproduces 3D cantilever beam theory — and gradient-based topology optimisation on it produces a physically sensible 3D load-carrying structure, extending the loop-2 substrate from 2D (trial 3) to 3D."
result: "Confirmed. The 24x24 B8 hex stiffness is exact (symmetric, 6 rigid-body zero-eigenvalues, zero force under rigid translation). The forward model's uniform-density compliance converges monotonically to Euler-Bernoulli under through-thickness refinement (-12.4% -> -3.4% -> -1.4% -> -0.61%), the residual being the known shear-locking of fully-integrated trilinear hex. OC topology optimisation on the GPU (24x8x8, ndof 6075) gives a sensible 3D cantilever load path at compliance 49.98, vol 0.30. Verified loop-2 substrate in 3D."
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
  topology at compliance 49.98 (`metrics.json:reference_compliance`).
- leakage_check: n/a — no agent in this experiment yet; reference + verification only.
- overfitting_signal: n/a — single deterministic optimisation, no train/val split.
- delta_from_prior: vs `2026-06-22-jaxfem-topology-optimization` (2D), extends the
  same SIMP/OC/autodiff machinery to 3D B8 hex elements; ndof 6075 vs 1666.
- unexpected_findings: coarse-mesh stiffness is textbook shear locking, confirmed
  by monotone convergence under refinement — a feature of B8 full integration,
  not a bug; documented rather than "fixed".
- next_candidates:
  - Hand the 3D `compliance(rho)` to the agent (as in trial 3) for a loop-2 pass@k.
  - Replace the dense solve with a sparse/CG (or matrix-free) solver to lift the
    grid-size cap and use reduced integration to cut shear locking.

## Follow-up

- Loop-2 agentic pass@k on the 3D substrate (agent writes its own 3D optimiser).
- Sparse solver + larger 3D grids; compare to a published 3D topology benchmark.
