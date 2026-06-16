---
kind: repo
name: "sfepy"
url: https://github.com/sfepy/sfepy
commit: release_2026.1
source: "raw/repos/sfepy.md"
added: "2026-06-16"
relevance: 3
status: skimmed
related_experiments: []
related_concepts: [agent-as-solver-operator, physics-grounded-evaluation]
tags: [finite-element, pde-solver, pure-python, solid-mechanics, elasticity, baseline, cpu-only]
---

# sfepy

## Purpose

SfePy (Simple Finite Elements in Python) solves systems of coupled PDEs by the
finite element method in 1D/2D/3D. It is usable two ways: as a black-box PDE
solver driven by declarative "problem definition files," or as a Python package
(NumPy/SciPy under the hood) for building custom solver applications. For this
project's purposes it is the lowest-friction, pure-Python operator-solver
baseline — a trivial `pip install sfepy` on the uv box, no compiled
toolchain or container fuss — making it a strong candidate for a fast
"hello-world" agentic trial (e.g. linear/nonlinear elasticity).

## Shape

- **Core stack:** NumPy + SciPy. No GPU, no autodiff — assembly and sparse
  linear/nonlinear solves run on CPU only.
- **Two interfaces:**
  - *Declarative* — "problem definition files" (a.k.a. input/problem
    description files) specifying PDEs in weak form (composed of named
    *terms*), boundary conditions, function spaces, materials. Copy an example
    from `sfepy/examples/` and edit.
  - *Interactive Python API* — full scripting access for advanced users; a
    documented interactive linear-elasticity tutorial exists.
- **Meshes:** must be supplied externally (no built-in mesher); reads several
  standard formats, notably legacy VTK. Results write to VTK or custom HDF5;
  visualize with ParaView or pyvista (pyvista supported directly).
- **License:** New BSD. **Authors:** Robert Cimrman and contributors.
- **Latest release:** `release_2026.1` (active, regular quarterly-style
  cadence — 2023.x through 2026.1 tags present).
- **Docs:** good — manual PDF (`doc/sfepy_manual.pdf`), online terms overview,
  tutorials, and a long-lived dev site at sfepy.org.

## Useful bits

- **Agent ergonomics (the main draw):** trivial `pip install sfepy` into a uv
  venv, no system dependencies beyond the NumPy/SciPy stack. Both the
  declarative input-file format and the pure-Python API are scriptable by an
  agent, so an LLM can author or mutate a problem definition and run it
  end-to-end without leaving Python. Ideal for a quick agentic
  [[agent-as-solver-operator]] baseline on linear/nonlinear elasticity.
- **Physics-grounded check:** VTK/HDF5 outputs and a real FE weak-form solve
  give a concrete, verifiable signal for [[physics-grounded-evaluation]] —
  the agent's proposed solution can be scored against an actual mechanics
  solve rather than a surrogate.
- Example library (`sfepy/examples/`) is the natural seed corpus for an agent
  to copy-and-modify.

## Follow-up

- **CPU-only / no-autodiff limitation:** SfePy does the forward solve well but
  offers no gradients through the solver, so it is unsuitable as-is for
  gradient-based [[agent-as-solver-operator]] inverse loops; it serves the
  forward-baseline role, not the differentiable one. Note this when comparing
  against autodiff/GPU FE backends.
- Trial idea: stand up the documented interactive linear-elasticity example as
  the project's first "hello-world" agentic run and measure how cleanly an
  agent can drive it.
