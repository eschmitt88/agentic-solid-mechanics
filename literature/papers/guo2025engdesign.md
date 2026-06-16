---
kind: paper
title: "Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs (EngDesign)"
authors: ["Xingang Guo", "Yaxin Li", "Xiangyi Kong", "et al."]
institutions: ["UIUC", "UC San Diego (multi-institution)"]
year: 2025
venue: "arXiv preprint"
peer_reviewed: false
url: https://arxiv.org/abs/2509.16204
code_url: null
citations: null
source: "raw/papers/guo2025engdesign.pdf"
added: "2026-06-16"
relevance: 4
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["physics-grounded-evaluation", "agentic-design-optimization"]
tags: ["benchmark", "engineering-design", "simulation-in-the-loop", "evaluation"]
---

# EngDesign

## TL;DR

A benchmark across nine engineering domains that scores LLMs on **functional
design** by running their designs through simulators — pioneering the
simulation-in-the-loop evaluation paradigm we want for grading an agent that
designs against physics, not against a textbook answer key.

## Claims

- Engineering design (synthesize knowledge, reason under constraints, produce
  functional designs) is distinct from textbook QA and unaddressed by prior
  benchmarks.
- Each task has explicit goals, constraints, and performance requirements;
  scoring is **dynamic simulation-driven functional verification**, not static
  answer checking.

## Methods

- Nine engineering domains; each task wraps a real design problem in a simulator
  that verifies whether the LLM's design meets requirements.

## Results

- Establishes simulation-based scoring as a viable, discriminative evaluation
  paradigm (headline: current LLMs are far from "engineering AGI").

## Critique / open questions

- Breadth over depth (nine domains) — our project is depth-first on solid
  mechanics, but the *scoring methodology* is exactly what we adopt.
- Simulation-in-the-loop verification = the natural grader for loop-2 design
  tasks (see [[agentic-design-optimization]]).

## Trust signals

- **Credibility:** 4 — large multi-institution effort (UIUC-led); preprint;
  methodologically aligned with the field's move to functional verification.

## Follow-up

- Adopt simulation-driven functional verification for trial 2 (operator design
  loop) and trial 3 (inverse design). Connects to
  [[physics-grounded-evaluation]].
