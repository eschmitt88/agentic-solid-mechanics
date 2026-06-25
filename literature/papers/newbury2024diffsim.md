---
kind: paper
title: "A Review of Differentiable Simulators"
authors: []
institutions: []
year: 2024
venue: IEEE Access
peer_reviewed: true
url: https://arxiv.org/abs/2407.05560
code_url:
citations:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 3
credibility: 3
status: skimmed
related_experiments: []
related_concepts:
  - gpu-differentiable-physics-simulation
  - differentiable-inverse-design
tags: [survey, differentiable-simulation]
---

# A Review of Differentiable Simulators

## TL;DR

A 2024 IEEE Access survey (Newbury et al.) mapping the landscape of differentiable physics simulators: it lays out foundational concepts, organizes simulators by gradient method (autodiff vs analytical/adjoint), dynamical and contact model, and integrator, and catalogs ~8 named open-source engines (TDS, PhiFlow, Brax, GradSim, Nimble, Dojo, JAX-Fluids, daX). Its central framing is the trade-off triangle of versatility, computational speed, and gradient accuracy, illustrated through applications in robotics, computational physics, and ML. For this project it serves as an orientation reference f…

## Claims

- Taxonomy of differentiable physics simulators organized by gradient method (autodiff, analytical/adjoint), dynamical model, contact model, and integrator
- Practical catalog of ~8 named open-source diff simulators: TDS, PhiFlow, Brax, GradSim, Nimble, Dojo, JAX-Fluids, daX (plus FluidLab, Warp, DiffTaichi in references)
- Discussion of the core design trade-off triangle: versatility vs computational speed vs gradient accuracy
- Survey of applications across robotics (system ID, control, sim-to-real), computational physics, and ML
- Discussion of known gradient pathologies, especially for contact-rich/discontinuous dynamics

## Critique / open questions

Real and legitimately peer-reviewed (IEEE Access), not vaporware — but it is a robotics-flavored survey, not the CFD/differentiable-physics map our window-shade problem needs. The "comprehensive" claim is fair for robotics/contact diff-sim and modest for fluids: only ~3 fluid entries and zero thermal/convection coverage. The catalog is also already ~2 years stale relative to today (2026) and predates much sm_120-era GPU work. Value is as a conceptual taxonomy and citation anchor, not as an actionable tool guide for our specific need.
- Robotics/contact-dynamics centric — the comparison table's axes (soft-body, contact model, integrator) are rigid/soft-body oriented; CFD and heat transfer are thin
- Fluid coverage is shallow: only PhiFlow, JAX-Fluids, and FluidLab are listed; no natural convection, Boussinesq buoyancy, or coupled thermal-fluid treatment
- Comparison table does NOT systematically tabulate GPU support or programming language — those must be checked per-simulator elsewhere
- Snapshot frozen at mid-2024; misses later/ongoing GPU-diff-physics work; a static reference, not a living survey

## Trust signals

- **Credibility:** active (as a published reference). Submitted Jul 8 2024, accepted/published in IEEE Access vol. 12, pp. 97581-97604, 2024. arXiv has at least a v2. A survey is "done" rather than maintained; content r…

## Relevance here

- **Project:** Moderate-to-good as a framing/orientation reference. It directly supports the loop-2 (differentiable/inverse design) thesis: it catalogs the tools an agent could operate, names gradient-method trade-offs an agent must reason about, and documents gradient pathologies (contact discontinuities, etc.) that an agentic optimizer would hit. Good background citation; not a tool we use.
- **Window-shade:** Low-to-moderate. The survey confirms the differentiable-CFD landscape is sparse and points to JAX-Fluids (compressible two-phase, high-order) and PhiFlow as the main differentiable fluid options — but neither is a natural-convection/Boussinesq incompressible buoyancy solver, and the paper offers no …

## Follow-up

- ...
