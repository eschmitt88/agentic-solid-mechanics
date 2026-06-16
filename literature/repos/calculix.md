---
kind: repo
name: "CalculiX (CCX/CGX)"
url: http://www.calculix.de/
commit:
source: "raw/repos/calculix.md"
added: "2026-06-16"
relevance: 5
credibility: 5
status: skimmed
related_experiments: []
related_concepts:
  - "[[agent-as-solver-operator]]"
  - "[[differentiable-inverse-design]]"
  - "[[agentic-design-optimization]]"
  - "[[multi-agent-self-correction]]"
  - "[[physics-grounded-evaluation]]"
tags: [fem, solver, calculix, solid-mechanics, abaqus-inp, operator]
---

# CalculiX (CCX/CGX)

This project's committed operator solver (ADR 0001). This note documents what
an LLM agent needs to drive it: how to invoke it, what it consumes and emits,
and where its ground-truth reference lives.

## Purpose

CalculiX is a free (GPL-2.0+) three-dimensional structural finite element
program. It has two parts:

- **CCX (CalculiX CrunchiX)** — the FE solver. Linear and nonlinear static,
  dynamic, and thermal analysis.
- **CGX (CalculiX GraphiX)** — an interactive OpenGL pre/post-processor (not
  needed for headless agent control; the agent uses CCX directly).

For [[agent-as-solver-operator]], CCX is the textbook "tool an agent operates
by writing text and reading text" — it grounds [[physics-grounded-evaluation]]
in a real PDE solve rather than a learned surrogate.

## Shape

- **Entry point**: the `ccx` CLI. Convention is `ccx jobname`, which reads
  `jobname.inp` and writes `jobname.frd` (full results) and `jobname.dat`
  (requested tabular output).
- **Input**: Abaqus-style `.inp` keyword deck — ASCII, keyword-driven
  (`*NODE`, `*ELEMENT`, `*MATERIAL`, `*STEP`, `*STATIC`, `*BOUNDARY`,
  `*CLOAD`, `*NODE FILE`, `*EL FILE`, ...).
- **Output**: `.frd` (binary/ASCII results consumed by CGX and converters)
  and `.dat` (ASCII tabular results from `*NODE PRINT` / `*EL PRINT`).
- **Implementation**: Fortran (~75%) + C (~25%), GPL-2.0+. Source on the
  GitHub mirror Dhondtguido/CalculiX (`src/`, `doc/`, `test/`).
- **Version**: 2.23 (official site / www.dhondt.de); GitHub mirror at 2.22.
- **Install**: `apt install calculix-ccx`, or `conda install -c conda-forge
  calculix`, or source build from the mirror.

## Useful bits — agent-control ergonomics

CalculiX is unusually well-suited to agentic control:

- **Text-in / text-out.** The model is a plain-text `.inp` deck and the
  agent-readable results are plain-text `.dat`. An LLM can author, edit, and
  diff the entire problem definition as text — no binary model database, no
  GUI in the loop.
- **Single CLI, no orchestration.** One invocation, `ccx jobname`. The whole
  solve is a single subprocess call with a deterministic exit code and log —
  trivial for an agent to launch, monitor, and retry. Supports
  [[agentic-design-optimization]] and [[multi-agent-self-correction]] loops
  where each iteration is one `ccx` call on an edited deck.
- **Parseable output.** `.dat` is column tabular ASCII keyed by the
  `*PRINT` requests in the deck — directly parseable for a scalar objective
  or constraint. `.frd` field results convert cleanly to VTK/VTU via
  `ccx2paraview` for richer field extraction.
- **Training-data familiarity.** The `.inp` deck is the Abaqus input format,
  one of the most widely documented FE formats in existence. An LLM has seen
  enormous amounts of Abaqus/CalculiX deck text and keyword documentation, so
  it can author valid decks and self-correct keyword errors with high prior —
  a concrete advantage for [[agent-as-solver-operator]] over niche solvers.
- **Authoritative reference.** The CCX keyword manual (`ccx_2.23.pdf` /
  `ccx_2.23.htm` from www.dhondt.de) documents every keyword, element type,
  and material model — the ground-truth spec an agent should consult/retrieve
  when constructing a deck.

## Gaps — what CalculiX does NOT give an agent

- **No GPU.** CCX is CPU-only (Fortran/C, direct + iterative CPU solvers).
  No GPU acceleration — solve wall-time bounds the iteration rate of any
  agentic optimization loop.
- **No autodiff / no gradients.** CCX is a forward solver; it exposes no
  analytic sensitivities of outputs w.r.t. design parameters. This is the
  core tension with [[differentiable-inverse-design]]: gradient-based inverse
  design must come from finite differences, adjoint code outside CCX, or a
  differentiable surrogate — CalculiX itself is a black-box forward map.
- Headless-only for agents: CGX is interactive (OpenGL), so post-processing
  in an agent loop relies on `.dat` parsing or `.frd` conversion, not CGX.

## Follow-up

- Pin the exact `ccx` CLI flag set (`-i`, threading env like
  `OMP_NUM_THREADS` / `CCX_NPROC_EQUATION_SOLVER`) from the manual —
  invocation shape is verified, specific flags are **unverified** this pass.
- Confirm a robust `.frd`/`.dat` parsing path: evaluate `ccx2paraview` vs a
  thin custom `.dat` parser for objective extraction in the operator loop.
- Decide the gradient strategy for [[differentiable-inverse-design]] given
  CCX has no autodiff (finite-difference vs surrogate).
- Verify Python wrappers (PyCCX, pycalculix) against current CCX 2.23 if a
  programmatic deck-builder is wanted instead of raw text authoring.
