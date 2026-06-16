---
adr: 0001
title: CalculiX as the primary agent-as-operator solver
date: 2026-06-16
status: accepted
---

# 0001 — CalculiX as the primary agent-as-operator solver

## Context

This project tests two agentic-simulation loops on solid mechanics:
(1) agent-as-operator (write/run/debug solver setups toward a goal) and
(2) differentiable / inverse design. We need a primary solver for loop (1)
that an LLM agent can drive autonomously on this Linux box (Ubuntu, RTX 5080,
32 cores, 91 GB RAM, `uv`-managed Python; no conda installed).

A three-way web sweep (2026-06-16, see
`raw/_candidates/2026-06-16-llm-agents-solid-mechanics-solvers.md`) compared
FEniCSx/DOLFINx, JAX-FEM, NVIDIA Warp, CalculiX, SfePy, Kratos, and MOOSE on
license, interface, install friction, GPU, and differentiability.

## Decision

**CalculiX** (`ccx`, GPL-2) is the primary solver for the agent-as-operator
loop. Problems are defined as plain-text Abaqus-style `.inp` decks, solved by a
single CLI executable, with `.frd`/`.dat` text output.

Rationale specific to *agent control*:

- **Text decks are maximally mechanical to generate and parse.** No Python API
  surface to learn; the agent emits/edits a structured text file and reads
  structured text back — the cleanest possible read/write/debug loop.
- **Single CLI executable** (`ccx job`) — trivial to invoke and capture
  exit/stderr; no environment/import fragility.
- **Easy install** (apt `calculix-ccx` or conda-forge) — no conda dependency,
  unlike FEniCSx.
- **Mature nonlinear solid mechanics** (contact, plasticity, large deformation)
  — enough physics depth for the full trial roadmap.
- The Abaqus `.inp` format is widely represented in LLM training data.

## Consequences

- Loop (1) trials target CalculiX `.inp` decks. Pairs with **Gmsh** (meshing →
  `.inp`), **ccx2paraview** / **PyVista** (post-processing).
- No GPU and no autodiff in CalculiX — those are out of scope for loop (1) and
  belong to loop (2).
- **Differentiable / inverse-design loop (2)** solver remains **JAX-FEM**
  (GPU-native + autodiff), to be confirmed when that loop is built. Recorded
  here as the working assumption, not yet committed via its own ADR.
- FEniCSx/SfePy remain available as alternative operator baselines if a Python
  API turns out to matter for a specific trial; revisit with a new ADR if we
  switch the primary.
- Trade-off accepted: text decks are less expressive than a full programming
  API, which may itself be an interesting axis to study (does deck-vs-API
  change agent success rate?).
