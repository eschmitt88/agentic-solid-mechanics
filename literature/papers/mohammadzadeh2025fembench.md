---
kind: paper
title: "FEM-Bench: A Structured Scientific Reasoning Benchmark for Evaluating Code-Generating LLMs"
authors: ["Saeed Mohammadzadeh", "Erfan Hamdi", "Joel Shor", "Emma Lejeune"]
institutions: ["Boston University"]
year: 2025
venue: "arXiv preprint"
peer_reviewed: false
url: https://arxiv.org/abs/2512.20732
code_url: null
citations: null
source: "raw/papers/mohammadzadeh2025fembench.pdf"
added: "2026-06-16"
relevance: 5
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["physics-grounded-evaluation", "agent-as-solver-operator"]
tags: ["agentic-simulation", "fem", "benchmark", "evaluation", "computational-mechanics"]
---

# FEM-Bench

## TL;DR

A computational-mechanics benchmark that scores LLM-generated FEM code on
objectively verifiable tasks from a first graduate course — giving us a
ready-made, physics-grounded evaluation harness and a clear map of where
simulator-operating LLMs currently break.

## Claims

- Computational mechanics is an ideal substrate for structured scientific-
  reasoning evaluation: clear math structure, strict physical/numerical
  constraints, objective verification.
- FEM-Bench 2025 = 33 introductory-but-nontrivial tasks; SOTA models do **not**
  reliably solve all of them.
- Best function-writer (Gemini 3 Pro): 30/33 solved at least once, 26/33 all
  five of five attempts. Best unit-test writer (GPT-5): 73.8% Average Joint
  Success Rate. Broad variance across models.

## Methods

- Tasks aligned to a first graduate computational-mechanics course.
- Two evaluation modes: function writing and discriminative unit-test writing.
- Five-attempt (pass@5-style) runs with objective verification.

## Results

- Even introductory tasks expose unreliability; geometric nonlinearity,
  buckling/eigenvalue problems, and writing discriminative tests are weak spots.

## Critique / open questions

- Tasks are Python/FEM-code oriented; our operator loop uses CalculiX decks —
  but the *evaluation philosophy* (objective verification, pass@k) ports
  directly to our HCE design.
- "Introductory but nontrivial" is exactly the right altitude for our trial 1.

## Trust signals

- **Credibility:** 4 — Emma Lejeune (BU) is a respected computational-mechanics
  ML group; careful benchmark methodology; preprint.

## Follow-up

- Use FEM-Bench's verification methodology as the template for our held-out
  scoring (see [[physics-grounded-evaluation]] and project HCE rule).
- Mirror its pass@5 protocol for our CalculiX trials (the `/implement --seeds N`
  path).
