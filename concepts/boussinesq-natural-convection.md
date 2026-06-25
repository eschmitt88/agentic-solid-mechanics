---
kind: concept
name: "boussinesq-natural-convection"
status: seedling
added: "2026-06-25"
sources: ["(workflow research 2026-06-25) — GPU/differentiable physics-sim landscape sweep"]
related_concepts:
  - gpu-differentiable-physics-simulation
  - differentiable-lattice-boltzmann
related_experiments: []
tags: [gpu, differentiable-simulation]
---

# boussinesq-natural-convection

## Definition

Buoyancy-driven incompressible flow modeled by coupling the incompressible Navier-Stokes momentum equation (with a temperature-dependent buoyancy body force under the Boussinesq approximation) to an energy/advection-diffusion equation for temperature.

## Why it matters here

The exact physics of the window-shade h(angle) problem and the capability missing from every surveyed CFD tool; must be a tracked concept since it drives solver selection.

## Connections

- [[gpu-differentiable-physics-simulation]]
- [[differentiable-lattice-boltzmann]]
