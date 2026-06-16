---
kind: paper
title: "Implicit differentiation with second-order derivatives and benchmarks in finite-element-based differentiable physics"
authors: ["Tianju Xue"]
institutions: []
year: 2025
venue: arXiv preprint
peer_reviewed: false
url: https://arxiv.org/abs/2505.12646
code_url: null
citations: null
source: "raw/papers/xue2025implicit.pdf"
added: "2026-06-16"
relevance: 3
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["differentiable-inverse-design"]
tags: ["differentiable-physics", "finite-element", "implicit-differentiation", "second-order-optimization", "inverse-problems", "shape-optimization"]
---

# Implicit differentiation with second-order derivatives and benchmarks in finite-element-based differentiable physics

## TL;DR

Develops an algorithm to compute second-order derivatives (Hessian-vector products) for implicit functions arising in finite-element-based differentiable physics, enabling exact-Hessian Newton-CG optimization for nonlinear inverse problems.

## Claims

- Second-order derivatives (Hessian-vector products) of implicit functions in finite-element simulations can be computed using primitive automatic differentiation tools.
- Newton-CG with exact Hessians accelerates convergence for nonlinear inverse problems compared to first-order methods.
- For linear cases, first-order L-BFGS-B is sufficient.
- The framework offers a foundation for integrating second-order implicit differentiation into differentiable physics engines.

## Methods

- Algorithm for computing Hessian-vector products for implicit functions, built on primitive AD tools.
- Accuracy validation via comparison against finite-difference approximations.
- Four benchmark problems, including traction force identification and shape optimization.

## Results

- Newton-CG with exact Hessians showed accelerated convergence on nonlinear inverse problems (e.g., traction force identification, shape optimization).
- L-BFGS-B was reported as sufficient for linear cases.

## Critique / open questions

- No code repository URL surfaced in the abstract/landing page; reproducibility unclear from what was reviewed.
- Single-author arXiv preprint, not yet peer-reviewed at time of ingest.
- Specific numerical results (convergence rates, problem sizes) not extracted; would require reading the full PDF.

## Trust signals

- **Credibility:** 4 — Builds on well-established implicit-differentiation and AD theory; validated against finite differences; concrete benchmarks. Tempered by being a non-peer-reviewed single-author preprint with no surfaced code link.

## Follow-up

- Read full PDF for exact benchmark setups and convergence numbers.
- Check whether code is released elsewhere (e.g., JAX-FEM ecosystem, which the author is associated with).
- Relevant to differentiable inverse design where exact second-order information could improve optimization for solid-mechanics problems.
