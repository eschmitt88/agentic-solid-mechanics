---
kind: paper
title: "Differentiable Programming for Differential Equations: A Review"
authors: []
institutions: []
year: 2024
venue: arXiv
peer_reviewed: false
url: https://arxiv.org/abs/2406.09699
code_url:
citations:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 4
credibility: 3
status: skimmed
related_experiments: []
related_concepts:
  - implicit-differentiation-through-solve
  - differentiable-inverse-design
tags: [survey, differentiable, adjoint, sensitivity]
---

# Differentiable Programming for Differential Equations: A Review

## TL;DR

A 75-page review (Sapienza, Bolibar, Schäfer, Pal, Boussange, Heimbach, Pérez, Persson, Rackauckas et al.) that systematizes how to compute gradients of differential-equation solvers: the full taxonomy of sensitivity methods — finite differences, complex-step, forward- and reverse-mode AD, and discrete vs. continuous adjoints (including their checkpointing and stability trade-offs). It is a methods/reference paper rather than a tool, with a companion open-source repo (ODINN-SciML/DiffEqSensitivity-Review) holding the LaTeX source plus Julia demonstration code. The work is grounded primarily in…

## Claims

- Conceptual taxonomy of all major sensitivity/differentiation methods for ODEs/PDEs
- Trade-off analysis: cost, accuracy, memory, numerical stability of forward AD vs reverse AD vs discrete vs continuous adjoint vs finite/complex-step differences
- Cross-ecosystem survey (Julia SciML, JAX, PyTorch, C++) with practitioner recommendations
- Demonstration Julia/Jupyter code in the companion repo

## Critique / open questions

Claims hold up and are not overstated — it genuinely is a ~75-page CC BY review covering forward/reverse AD and discrete/continuous adjoints across Julia/JAX/PyTorch/C++, verified directly from the PDF and arXiv page (v2 Oct 2025). The honest caveat is the framing: 'gpu=n/a' is correct precisely because this is a paper, not a tool — anyone expecting a runnable differentiable solver will be disappointed. It is also visibly Julia-SciML-flavored (Rackauckas + ODINN-SciML authorship), so treat its Python/JAX recommendations as a starting map rather than gospel. No vaporware; high-quality, well-mai…
- Not software — provides no GPU kernels, no Python API, no convection/CFD solver you can call; it is reference knowledge only.
- Julia-SciML-centric in its examples and code; Python/JAX coverage is descriptive, not a porting guide.
- No worked natural-convection or Boussinesq example — CFD coverage is limited to scattered aerodynamic citations, so the window-shade problem is not directly addressed.
- Companion repo code is illustrative figures/demos, not a maintained library; commits stalled late 2024.

## Trust signals

- **Credibility:** active — arXiv v2 revised 2025-10-29 (v1 2024-06-14); companion repo last commit 2024-11-26 (facusapienza21). The paper is the maintained artifact and was updated within the last ~8 months.

## Relevance here

- **Project:** High for loop 2 (differentiable/inverse design) as the canonical reference for differentiating physics solvers; lower for loop 1 (operating simulators).
- **Window-shade:** Indirect but real. It contains no convection/Boussinesq example, so it won't give us h(angle) for the venetian blind. Its value is upstream: when we build or wrap a differentiable transient natural-convection solver to get gradients of the effective heat-transfer coefficient w.r.t. slat angle, this …

## Follow-up

- ...
