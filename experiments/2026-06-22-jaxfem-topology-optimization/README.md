---
kind: experiment
slug: "jaxfem-topology-optimization"
date: "2026-06-22"
status: done
hypothesis: "Given a differentiable FEM (autodiff sensitivities), an LLM agent can run the LOOP-2 differentiable inverse-design loop — write and drive a gradient-based optimizer — to solve minimum-compliance topology optimization, reaching a near-optimal, volume-feasible structure."
result: "Confirmed. Given only the differentiable solver (jaxfem.py), the agent wrote its own Optimality-Criteria optimizer using jax autodiff sensitivities and drove compliance 1838 → 296.6 at exactly 40% volume in 76 iterations — within 0.31% of the reference optimum (295.7). Independent recompute of its saved density matches its reported compliance exactly. Ran on CPU (GPU blocked by a driver/kernel mismatch — see note)."
related_concepts: ["differentiable-inverse-design", "agentic-design-optimization", "physics-grounded-evaluation"]
related_literature: ["xue2022jaxfem", "deepmodeling-jax-fem", "nvidia-warp", "park2026self"]
tags: ["jax", "differentiable-fem", "topology-optimization", "trial-3", "loop-2", "autodiff"]
---

# jaxfem-topology-optimization

Trial 3 — the first **loop-2** trial: **differentiable inverse design**. In
loops 1/2 the agent operated/designed via an opaque solver (CalculiX). Here the
agent's role shifts to *orchestrating gradient-based optimization* through a
**differentiable** solver — the [[differentiable-inverse-design]] mechanism.

## Setup

- **Differentiable solver (`jaxfem.py`):** a compact, self-contained 2D
  plane-stress FEM in JAX — bilinear Q4 elements, SIMP material interpolation,
  density filter — exposing an end-to-end autodifferentiable `compliance(rho)`.
  `jax.grad(compliance)` yields the sensitivities that drive optimization. This
  stands in for **JAX-FEM** ([[xue2022jaxfem]] / [[deepmodeling-jax-fem]], the
  project's production differentiable solver) so the trial is transparent and
  self-contained; the agentic loop is identical. Device-agnostic (GPU when
  available, else CPU).
- **Problem:** 2D cantilever, 48×16 elements (ndof=1666), left edge clamped,
  unit downward load at the middle of the right edge. Minimise compliance fᵀu
  subject to mean physical density ≤ **0.40**, ρ∈[0,1]. SIMP penal=3, filter
  rmin=1.5.
- **Code:** `topopt_reference.py` (reference Optimality-Criteria optimizer using
  the autodiff sensitivities; saves density npy/pgm/ascii), `agent_topopt`
  harness pattern (the agent writes its own optimizer).
- **Grading:** agent's final compliance within 5% of the reference, volume ≤
  0.42. Compliance is recomputed independently from the agent's saved density.

## GPU status — RESOLVED (2026-06-22, post-reboot)

The earlier driver/kernel-module mismatch (NVML 580.167 vs NVRM 580.159.03,
`CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`) is **fixed by the reboot**.
`nvidia-smi` is healthy (RTX 5080, 580.167.08) and JAX reports
`CudaDevice(id=0)`. The GPU re-run is **bit-identical to CPU** (C=295.69531 →
determinism confirmed) and ~4× faster at 48×16; grid scaling to 120×40
(ndof 9922) runs on GPU in ~70 s — see the **GPU & grid scaling** and
**external-gold verification** sections below / on the QA page.

## Reference optimum (`results/reference.json`, `reference_density.pgm`)

OC converged to **compliance 295.7** at volume 0.40 — the classic cantilever
truss (top/bottom chords + diagonal bracing toward the load):

```
@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%#-.
##%@@@@%##########%%#%%#*+++++++++*###*:
.:*###%#=........=#+=*##=:.......-*+-*##+:
  ...-*%*:.     :**. .-*#*:     .+*: .:*#*-.
      :*##*:   .+*:    :*#*-.   :*-    :=*#*:
       .-*%*=. -*-      .-*#*:.:**.      :*##+:
         :=*#*=**.        :+#*++*:        .:*#*=
           :*%%#:          .:#%#=           :+%#
           :*%%#:          .:#%#=           :+%#
         :=*#*=**.        :+#*++*:        .:*#*=
       .-*%*=. -*-      .-*#*:.:**.      :*##+:
      :*##*:   .+*:    :*#*-.   :*-    :=*#*:
  ...-*%*:.     :**. .-*#*:     .+*: .:*#*-.
.:*###%#=........=#+=*##=:.......-*+-*##+:
##%@@@@%##########%%#%%#*+++++++++*###*:
@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%#-.
```

## Result

Points at `metrics.json`. Agent = in-session subagent, isolated in
`results/agent_demo/` with `jaxfem.py` (the solver) but **not** the reference
optimizer/solution (verified clean).

- Wrote **its own Optimality-Criteria optimizer** (bisection on the volume
  multiplier, move limit 0.2) using `jax.value_and_grad` sensitivities.
- Compliance **1838 → 296.6** in **76 iterations**, volume exactly 0.40.
- **+0.31% vs the reference 295.7 → PASS.** Independent recompute of its saved
  `agent_density.npy` gives 296.6048, exactly its reported value (no fabrication).

## Interpretation

This closes the loop-2 mechanism: the agent didn't operate an opaque solver, it
*orchestrated a gradient-based optimization* — chose an algorithm (OC), used the
differentiable solver's autodiff sensitivities, enforced the volume constraint,
and converged to within 0.3% of the reference optimum. Combined with trials 1/2,
the agent now demonstrably handles all three agentic axes the project set out:
drive a solver, design with a solver, and **differentiate** a solver.

## Diagnostics

Unless noted, numbers reference `metrics.json`.

- intended_effect_confirmed: yes — agent reached compliance 296.6, +0.31% vs
  reference (`metrics.json:agent_optimizer_demo`), volume-feasible.
- leakage_check: `grep` over `results/agent_demo/` clean (no `reference.json` /
  `topopt_reference` reads); compliance recomputed independently from its density.
- overfitting_signal: n/a — single optimization instance.
- delta_from_prior: first loop-2 trial; vs loops 1/2 (operate/design a solver),
  this orchestrates autodiff-based optimization.
- unexpected_findings: agent independently chose OC (the field-standard method)
  and matched the reference to 0.3% — and the autodiff FEM ran the full topopt
  in ~6 s on CPU even jit-compiled.
- seeds_run: 1 live subagent demo.
- metric_aggregation: single run; reference is a 60-iteration OC solve.
- next_candidates:
  - Re-run on GPU after the driver reboot; scale the grid (e.g. 120×40) to show
    the GPU speedup and move to JAX-FEM proper ([[deepmodeling-jax-fem]]).
  - Parameter-inversion variant (recover a material field from observed
    displacements) — a different loop-2 task.
  - Have the agent compare optimizers (OC vs Adam vs MMA) and self-select.

## External-gold verification (`verify_benchmark.py` → `results/verification.json`)

Added to answer "how do we know the reference is right" — checks against targets
defined **outside** this repo:

1. **Forward-model gold (closed form):** the differentiable FEM, uniform density,
   reproduces analytical beam theory. compliance(ones) = tip deflection; **max
   |error vs Timoshenko| = 0.86%** across L/h ∈ {3,10,20}, and the L/h=3 case
   shows Euler–Bernoulli off by +8.9% (shear) while Timoshenko holds to +0.2% —
   the same shear correction CalculiX showed in trial 1.
2. **Optimizer gold (published problem):** reproduces the **canonical MBB beam**
   [Sigmund 2001; Andreassen et al. 2011] — the textbook topology (top + bottom
   chords, triangulated web). Our converged compliance ≈ 218.5; the published
   *scalar* is filter/rmin-dependent (~200–220), so the rigorous match is the
   reproduced topology, not a single number we can't re-derive offline.

## GPU & grid scaling

GPU re-run is bit-identical to CPU (determinism) and ~4× faster at 48×16. Scaled
to **120×40 (ndof 9922)** on the GPU: compliance 233.9 in ~70 s — the dense
direct solve is the wall (ndof grows as resolution³), motivating a sparse / real
JAX-FEM solver. `results/scaled_density.npy` rendered on the QA page.

## Follow-up

- **Trial 4 (done):** 3D differentiable FEM —
  `../2026-06-22-jaxfem3d-cantilever-topology/`.
- Sparse / matrix-free solve to lift the grid-size cap on GPU.
- An `agent_topopt.py` headless harness (like trials 1/2's `agent_design.py`)
  for loop-2 pass@k.
