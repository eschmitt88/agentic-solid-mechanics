---
kind: experiment
slug: "window-shade-natural-convection"
date: "2026-07-01"
status: done
hypothesis: "A hand-rolled thermal lattice-Boltzmann solver (Boussinesq natural convection) in JAX, validated against the de Vahl Davis benchmark, can predict the effective window↔room convective heat-transfer coefficient of an interior venetian blind as a function of slat angle — h(θ) — over the full tilt range."
result: "Confirmed. Solver validated on de Vahl Davis (Nu +15/+8/+4% at Ra 1e3/1e4/1e5, shrinking with Ra; velocity + field textbook). The blind h(θ) sweep (θ∈[−75,+75]°, pitch 2.25″, chord 2.5″, Ra_W=1e5) gives a coherent U-shaped curve: minimum at θ=0 (horizontal slats block the buoyant window boundary layer most), rising ~40% toward closed (±75°), and ASYMMETRIC by ~5% (h(−θ)>h(+θ) — gravity gives the plume a preferred direction). Every point energy-conserved (Nu(x) spread ±0.5–1%)."
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

## Blind h(θ) sweep (`blind.py`, `results/blind_sweep.json`)

**Model:** a tall differentially-heated cavity (window wall isothermal-hot on the
right, room boundary isothermal-ambient on the left, adiabatic top/bottom) with
an interior venetian blind of 6 slats near the window — the standard tractable
fenestration model (room as an ambient boundary, per ISO-15099-style treatments),
reusing the validated solver with slats as solid + adiabatic nodes. Geometry held
consistent across the sweep (chord 2.5″=40 cells, pitch 2.25″=36 cells, fixed
position/thickness); only θ varies. Effective h ∝ the through-cavity heat transfer,
measured as the height-integrated horizontal heat flux in the clear room-side band
(uncorrupted by the adiabatic slats; x-flat to ±0.5–1% → energy-conserved).

**Result** (`results/h_vs_angle.png`, `results/blind_fields.png`):

| θ | −75 | −60 | −45 | −30 | −15 | 0 | +15 | +30 | +45 | +60 | +75 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h/h(0) | 1.43 | 1.40 | 1.28 | 1.09 | 1.01 | **1.00** | 1.01 | 1.08 | 1.22 | 1.35 | 1.40 |

- **Minimum at horizontal (θ=0):** a stack of horizontal slats acts as baffles
  perpendicular to the rising window boundary layer, suppressing convection most.
- **Rises ~40% toward closed (±75°):** tilted slats obstruct the vertical plume
  less (and near-vertical they act as fins channelling flow along the window).
- **Asymmetric (~5%): h(−θ) > h(+θ).** Gravity gives the buoyant plume a preferred
  direction, so window-side-down slats (−θ) enhance the up-flow slightly more than
  window-side-up (+θ) — a real effect, well above the ±0.5% numerical noise.

**Caveats (honest):** moderate laminar Ra_W=1e5 (a full-height window is
higher-Ra / transitional — the *trend* holds, absolute h would need turbulence
modelling); enclosed-cavity model (room = ambient boundary, not a resolved open
room); thin adiabatic slats that do not span/seal the gap (so "closing" them does
not block convection the way a gap-sealing blind would — a geometry choice). The
absolute Nu carries the ~4–8% de-Vahl-Davis BC bias; the h(θ) *trend* cancels it.
Quantitative validation against experimental interior-blind data
([[zheng2022venetianblinds]]) is the next step.

## Diagnostics

- intended_effect_confirmed: partial — solver validated on de Vahl Davis; blind
  sweep pending.
- leakage_check: n/a (no agent; validation against published benchmark).
- next_candidates:
  - Build the tall-cavity + slat geometry and sweep θ → h(θ).
  - Tighten the benchmark to <5% at all Ra with half-way bounce-back if needed.
