---
kind: concept
name: "Agent as solver operator"
status: seedling
added: "2026-06-16"
sources: ["ni2023mechagents"]
related_concepts: ["multi-agent-self-correction", "physics-grounded-evaluation"]
related_experiments: []
tags: ["agentic-simulation", "core-loop"]
---

# Agent as solver operator

## Definition

The agentic loop in which an LLM agent treats an external simulator as a tool:
it authors/edits the solver's input (a text deck or API script), runs the
solver, parses the results, diagnoses failures, and iterates toward an analysis
or design goal — without a human in the inner loop.

## Why it matters here

This is loop (1) of the project, and the primary thing we are testing. The
agent's competence is graded against ground truth (analytical solutions, mesh
convergence, benchmarks), not its self-report. Our committed substrate is
**CalculiX** `.inp` text decks (see ADR 0001), chosen because text in / text
out is the most mechanical read-write-debug surface for an agent.

## Connections

- First demonstrated for FEM elasticity by [[ni2023mechagents]] (MechAgents).
- Distinct from [[differentiable-inverse-design]] (loop 2), where the agent
  orchestrates gradient-based optimization through a differentiable solver
  rather than treating the solver as an opaque tool.
- Reliability of the loop hinges on [[multi-agent-self-correction]] (or
  single-agent retry) and is measured via [[physics-grounded-evaluation]].
