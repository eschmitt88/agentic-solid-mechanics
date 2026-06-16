---
kind: paper
title: "JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science"
authors: ["Tianju Xue", "Shuheng Liao", "Zhengtao Gan", "Chanwook Park", "Xiaoyu Xie", "Wing Kam Liu", "Jian Cao"]
institutions: ["Northwestern University"]
year: 2022
venue: "Computer Physics Communications"
peer_reviewed: true
url: https://arxiv.org/abs/2212.00964
code_url: https://github.com/deepmodeling/jax-fem
citations: null
source: "raw/papers/xue2022jaxfem.pdf"
added: "2026-06-16"
relevance: 5
credibility: 5
status: skimmed
related_experiments: []
related_concepts: ["differentiable-inverse-design"]
tags: ["differentiable-fem", "jax", "gpu", "topology-optimization", "inverse-design", "solver"]
---

# JAX-FEM

## TL;DR

The canonical differentiable, GPU-accelerated 3D FEM library (pure Python, on
JAX) — the planned solver for this project's loop 2: it computes sensitivities
by autodiff so an agent can run gradient-based inverse design and topology
optimization without hand-deriving adjoints.

## Claims

- Pure-Python yet scalable: ~**10× GPU speedup** vs. commercial FEM on a 3D
  tensile problem with 7.7M DOF.
- Automatic differentiation solves inverse problems with no manual sensitivity
  derivation.
- Demonstrated on 3D topology optimization of nonlinear materials (optimal
  compliance) and data-driven multi-scale composite computations.

## Methods

- FEM assembly + solvers implemented on Google JAX; autodiff provides exact
  gradients through the solve.
- Mature solid-mechanics element/material support (linear elasticity,
  hyperelasticity, macro and crystal plasticity per the repo).

## Results

- Large 3D problems solved on GPU; inverse-design and topology-optimization
  examples shipped.

## Critique / open questions

- GPL-3.0; commercial use needs author contact (fine for research).
- JAX + CUDA on Blackwell (RTX 5080, sm_120) needs a recent JAX build — verify
  install before committing loop-2 trials.
- The agent's role here is orchestration (define objective, drive the optimizer,
  interpret results), not deck-writing — a genuinely different loop from
  [[agent-as-solver-operator]].

## Trust signals

- **Credibility:** 5 — Northwestern (Cao, W.K. Liu) flagship CM group; published
  in Computer Physics Communications; widely used open-source code.

## Follow-up

- The substrate for loop-2 trial 3 (inverse design / topology optimization).
  Confirm via its own ADR once loop-2 build begins. See
  [[differentiable-inverse-design]].
