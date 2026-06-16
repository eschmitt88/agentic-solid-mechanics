---
kind: concept
name: "Agentic design optimization"
status: seedling
added: "2026-06-16"
sources: ["guo2025engdesign"]
related_concepts: ["differentiable-inverse-design", "agent-as-solver-operator", "physics-grounded-evaluation"]
related_experiments: []
tags: ["agentic-simulation", "design-optimization", "core-loop"]
---

# Agentic design optimization

## Definition

An LLM agent driving an iterative design-improvement loop against a simulator:
propose a design (geometry, sizing, topology, material), simulate it, read the
performance/constraint outcomes, and revise — toward an explicit objective such
as minimum mass subject to a stress constraint. The optimization can be
gradient-free (agent edits decks, loop 1 style) or gradient-based
([[differentiable-inverse-design]], loop 2 style).

## Why it matters here

It is the shared target of both project loops' more advanced trials (roadmap
trials 2 and 3) and the point where "operate the solver" becomes "design with
the solver." TopOptAgents (arXiv 2605.23273, in the candidate triage) is the
LLM-multi-agent instance for topology optimization; EngDesign supplies the
simulation-in-the-loop scoring.

## Connections

- Gradient-based realization: [[differentiable-inverse-design]] on JAX-FEM.
- Gradient-free realization: [[agent-as-solver-operator]] editing CalculiX decks
  across iterations.
- Scored by [[physics-grounded-evaluation]] / simulation-in-the-loop
  verification ([[guo2025engdesign]]).
