---
kind: paper
title: "AutoNumerics: An Autonomous, PDE-Agnostic Multi-Agent Pipeline for Scientific Computing"
authors: ["Jianda Du", "Youran Sun", "Haizhao Yang"]
institutions: []
year: 2026
venue: arXiv preprint
peer_reviewed: false
url: https://arxiv.org/abs/2602.17607
code_url: null
citations: null
source: "raw/papers/du2026autonumerics.pdf"
added: "2026-06-16"
relevance: 3
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["code-from-scratch-numerical-solver-agent", "agent-as-solver-operator", "multi-agent-self-correction"]
tags: ["multi-agent", "pde-solver", "code-generation", "scientific-computing", "self-verification"]
---

# AutoNumerics: An Autonomous, PDE-Agnostic Multi-Agent Pipeline for Scientific Computing

## TL;DR

A multi-agent LLM framework that autonomously designs, implements, debugs, and
verifies numerical PDE solvers directly from natural-language problem
descriptions, generating transparent solver code grounded in classical
numerical analysis rather than neural surrogates.

## Claims

- A multi-agent pipeline can autonomously design, implement, debug, and verify
  numerical solvers for general PDEs from natural-language descriptions.
- Unlike neural-based approaches, the framework generates transparent solvers
  grounded in classical numerical analysis.
- A coarse-to-fine execution strategy and a residual-based self-verification
  mechanism improve robustness.
- On 24 PDE problems, the approach achieves competitive or superior accuracy
  compared to existing neural and LLM-based baselines.

## Methods

- Multi-agent framework that takes a PDE problem stated in natural language and
  produces a numerical solver end-to-end (design → implementation → debugging →
  verification).
- Coarse-to-fine execution strategy.
- Residual-based self-verification mechanism for checking solver correctness.
- Solvers are classical numerical-analysis code (not neural networks).

## Results

- Evaluated on 24 PDE problems.
- Reported competitive or superior accuracy versus neural and LLM-based
  baselines.

## Critique / open questions

- arXiv preprint, not peer-reviewed; results based on the authors' own 24-problem
  benchmark.
- No code repository linked on the abstract page.
- The paper targets general PDEs broadly; its specific applicability to
  solid-mechanics PDEs and the relevance of its verification mechanism there are
  not yet assessed.

## Trust signals

- **Credibility:** 4 — established authors in scientific machine learning
  (Haizhao Yang has a substantial track record), coherent and concrete
  methodology, but a non-peer-reviewed preprint with no released code and a
  self-constructed benchmark.

## Follow-up

- Inspect the residual-based self-verification mechanism in detail — it may
  generalize to verifying solid-mechanics solver outputs.
- Note the "agent generates numerical solver code from scratch" framing,
  captured as [[code-from-scratch-numerical-solver-agent]] — the third agentic
  design axis alongside driving and differentiating a library.
