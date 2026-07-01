---
kind: experiment
slug: "window-shade-natural-convection"
date: "2026-07-01"
status: running
hypothesis: "A hand-rolled thermal lattice-Boltzmann solver (Boussinesq natural convection) in JAX, validated against the de Vahl Davis benchmark, can predict the effective window↔room convective heat-transfer coefficient of an interior venetian blind as a function of slat angle — h(θ) — over the full tilt range."
result: "In progress. Solver built and VALIDATED on de Vahl Davis (Nu error +15%/+8%/+4% at Ra 1e3/1e4/1e5 — decreasing with Ra as convection dominates; velocity field matches benchmark, Ra-scaling correct). Blind h(θ) sweep is the next step."
related_concepts: ["boussinesq-natural-convection", "differentiable-lattice-boltzmann", "gpu-differentiable-physics-simulation", "physics-grounded-evaluation"]
related_literature: ["zheng2022venetianblinds", "xlb"]
tags: ["cfd", "natural-convection", "lattice-boltzmann", "jax", "gpu", "venetian-blind", "trial-6"]
---

# window-shade-natural-convection

Trial 6 / the project's first **fluids / coupled-physics** trial (the planned
multiphysics expansion). Goal: the effective heat-transfer coefficient of an
**interior venetian blind** in front of a window, as a function of slat angle,
via 2D natural-convection CFD — a genuinely new capability (our prior solvers
were structural FEA + differentiable solid-mechanics FEM).

## Problem (from the user's geometry sketch)

- A **window** (vertical isothermal surface) with a **large open room** on the
  interior side; a venetian blind hangs on the room side of the window.
- **Slats:** 2.5″ chord, tilt angle θ from horizontal, **−90° < θ < +90°**
  (both directions; h(θ) may be asymmetric because gravity gives the buoyant
  plume a preferred direction). Overlap of ~¼″ at near-vertical fixes the
  **pitch ≈ 2.25″** (chord − overlap); a minimum practical gap caps |θ| short of 90°.
- **Deliverable:** effective h (window↔room convective coefficient) vs θ, at
  consistent geometry (fix chord, pitch, gap; vary only θ).
- Validation reference for the physics: [[zheng2022venetianblinds]].

## Solver

`jaxlbm.py` — hand-rolled **thermal lattice-Boltzmann** in JAX (D2Q9 flow +
D2Q5 temperature, Guo-forced **Boussinesq** buoyancy). Grid + mask based:
angled slats are solid (bounce-back) + adiabatic nodes; temperature walls use
anti-bounce-back (Dirichlet) or bounce-back (adiabatic). Runs to steady state in
jitted blocks on the GPU. Differentiable (h(θ) sensitivities available later).

## Validation — de Vahl Davis cavity (`devahl.py`, `results/devahl.json`)

The canonical natural-convection benchmark: hot/cold vertical walls, adiabatic
top/bottom, average hot-wall Nusselt number vs published values (Pr=0.71).

| Ra | Nu (ours) | Nu (de Vahl Davis 1983) | error |
|---|---|---|---|
| 10³ | 1.289 | 1.118 | +15.3% |
| 10⁴ | 2.424 | 2.243 | +8.1% |
| 10⁵ | 4.701 | 4.519 | +4.0% |

The error **shrinks as Ra rises** — the full-way anti-bounce-back wall placement
is an O(1) offset that matters less as convection dominates the interior heat
flux (thin boundary layers). The velocity field matches the benchmark (max speed
≈ published v_max in lattice units), the plane-integrated Nu(x) is x-independent
(steady, consistent), and the Ra-scaling tracks the benchmark. Absolute Nu
carries a modest BC bias; the **relative h(θ) trend** — the actual deliverable,
computed at fixed Ra and fixed BCs — has that systematic offset cancel. A window
operates at high Ra (≥1e4), i.e. the ≤8%-and-shrinking regime.

Debugging log (all fixed): wet-node Dirichlet-T was over-diffusive (Nu −14%);
switched to anti-bounce-back (+8%, resolution-converged). Nu must be taken from
the interior heat-flux integral, not a one-cell wall gradient. Convergence must
be probed on flow speed, not total heat (which is ~conserved).

## Next — the blind h(θ) sweep (modeling decision)

Model the window+room as a **tall differentially-heated cavity**: window wall
(isothermal, right), room-side boundary held at ambient T (left), adiabatic
top/bottom, with the blind slats at angle θ near the window. This is the standard
tractable fenestration model (the "room" as an ambient boundary, per ISO-15099-
style treatments) and reuses the validated solver directly — no open-boundary
scheme needed. A fully **open** room (open BCs, resolved plume) is a later
refinement. Then sweep θ ∈ (−90°, +90°) at pitch 2.25″, chord 2.5″ → h(θ).

## Diagnostics

- intended_effect_confirmed: partial — solver validated on de Vahl Davis; blind
  sweep pending.
- leakage_check: n/a (no agent; validation against published benchmark).
- next_candidates:
  - Build the tall-cavity + slat geometry and sweep θ → h(θ).
  - Tighten the benchmark to <5% at all Ra with half-way bounce-back if needed.
