---
kind: paper
title: "Differentiable programming across the PDE and Machine Learning barrier"
authors: ["Nacime Bouziani", "David A. Ham", "Ado Farsi"]
institutions: ["Imperial College London"]
year: 2024
venue: arXiv
peer_reviewed: false
url: https://arxiv.org/abs/2409.06085
code_url: null
citations: null
source: "raw/papers/bouziani2024differentiable.pdf"
added: "2026-06-16"
relevance: 3
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["differentiable-inverse-design"]
tags: ["differentiable-programming", "pde", "finite-element", "firedrake", "machine-learning", "pytorch", "jax"]
---

# Differentiable programming across the PDE and Machine Learning barrier

## TL;DR

A generic differentiable programming framework that lets scientists and engineers specify end-to-end differentiable models coupling machine learning and PDE-based components, relying on code generation for high performance. It is integrated into the Firedrake finite-element library and supports the PyTorch and JAX ecosystems.

## Claims

- Provides a generic abstraction for composing machine learning operators with PDE-based components in a fully differentiable, end-to-end pipeline.
- Bridges the "barrier" between PDE solvers and ML frameworks without sacrificing performance, by relying on code generation.
- Integrated into the Firedrake finite-element library and interoperable with the PyTorch and JAX ecosystems.

## Methods

- Differentiable programming abstraction layered on a finite-element library (Firedrake).
- Code generation for high-performance evaluation of coupled ML/PDE models.
- Interoperability with PyTorch and JAX autodiff ecosystems for gradient flow across the PDE/ML boundary.

## Results

- Demonstrates that ML and PDE components can be coupled and differentiated end-to-end within a single productive framework.

## Critique / open questions

- arXiv preprint; not peer-reviewed at time of ingest.
- No code repository surfaced from the abstract page.
- Relevance to agentic solid mechanics is via the differentiable PDE/ML coupling that underpins differentiable inverse design, rather than agentic methods themselves.

## Trust signals

- **Credibility:** 4 — Imperial College London authors including David A. Ham (a core Firedrake developer); builds on the established, well-regarded Firedrake finite-element library. arXiv preprint, not yet peer-reviewed.

## Follow-up

- Check whether code/artifacts have been released since ingest.
- Assess applicability of the Firedrake + PyTorch/JAX coupling to differentiable inverse-design experiments in solid mechanics.
