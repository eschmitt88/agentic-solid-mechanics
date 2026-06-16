---
kind: repo
name: "FEniCSx / DOLFINx"
url: https://github.com/FEniCS/dolfinx
commit:
source: "raw/repos/fenics-dolfinx.md"
added: "2026-06-16"
relevance: 3
status: skimmed
related_experiments: []
related_concepts: [agent-as-solver-operator, differentiable-inverse-design, agentic-design-optimization, physics-grounded-evaluation]
tags: [fem, solver, solid-mechanics, python, ufl, differentiable, reference]
---

# FEniCSx / DOLFINx

## Purpose

DOLFINx is the computational core of the [FEniCSx](https://fenicsproject.org)
project — a general finite-element PDE solver implementing the FEniCS Problem
Solving Environment in C++ with a first-class Python interface. It is the
leading open-source *alternative operator solver* to CalculiX for this
project, and is directly relevant because [[deotale2026allfem]] (ALL-FEM)
built its agentic FEM pipeline on FEniCS. We committed to CalculiX, but
FEniCSx remains the reference point for the agent-as-solver-operator pattern.

## Shape

- **Languages:** C++ core + Python interface (the layer an agent would drive).
- **Weak-form interface:** problems are expressed symbolically in **UFL**
  (Unified Form Language) — the variational/weak form is written almost
  verbatim from the math (e.g. `inner(sigma(u), eps(v)) * dx`), then
  JIT-compiled to C kernels via FFCx/Basix. This is the defining ergonomic:
  the model *is* the math, not a deck of cards.
- **Solid mechanics:** fully capable — linear/nonlinear elasticity,
  hyperelasticity, plasticity, contact (via add-ons), thermomechanics. The
  official tutorials include a complete linear-elasticity walkthrough.
- **Parallelism:** distributed-memory **MPI** by default; linear algebra and
  solvers via **PETSc**. Default execution is **CPU/MPI**.
- **GPU:** not in the default build. GPU acceleration is available via
  `cuda-dolfinx` (and PETSc GPU backends), but it is an add-on, not stock.
- **Differentiability:** adjoint/inverse-design support comes from
  **dolfin-adjoint / pyadjoint**, which records the UFL operations on a tape
  and replays them for gradients — enabling PDE-constrained optimization and
  topology/shape inverse design without hand-deriving adjoints.
- **License:** **LGPL-3**.
- **Docs:** excellent — `docs.fenicsproject.org`, plus the widely-used
  "FEniCSx tutorial" with runnable Jupyter notebooks. Active Discourse/Slack.

## Useful bits

- **Agent ergonomics are the headline.** A clean, well-typed Python API plus
  a symbolic weak-form DSL (UFL) and a large corpus of high-quality tutorial
  notebooks make FEniCSx unusually learnable for an LLM agent — the agent can
  pattern-match the math-to-UFL mapping and the docs are dense enough to
  ground generation. This is the case *for* it as an
  [[agent-as-solver-operator]] backend.
- **Differentiable by design via pyadjoint** — a natural fit for
  [[differentiable-inverse-design]] and [[agentic-design-optimization]],
  where the agent proposes a design and gradients flow back through the
  solve.
- **Physics-grounded evaluation:** a real FE solver gives objective,
  reproducible residuals/energies an agent loop can score against
  ([[physics-grounded-evaluation]]).

## Follow-up

- **Dependency friction (the case against it on this box):** the supported
  install paths are **conda-forge (`fenics-dolfinx`), apt/Debian, Docker, or
  Spack**. Pip-from-source requires building the C++ core + PETSc/petsc4py and
  is hard. **This box is uv-only — no conda.** That makes FEniCSx awkward to
  stand up here versus CalculiX, and is a concrete reason we did not pick it,
  despite its superior agent ergonomics. A Docker image (`dolfinx/dolfinx`)
  would be the least-painful route if we ever want a side-by-side comparison.
- If we want to validate the agent-as-solver pattern against the ALL-FEM
  baseline directly, FEniCSx-in-Docker is the closest reproduction of
  [[deotale2026allfem]]'s stack.
</content>
