---
kind: concept
name: "Multi-agent self-correction"
status: seedling
added: "2026-06-16"
sources: ["ni2023mechagents"]
related_concepts: ["agent-as-solver-operator"]
related_experiments: []
tags: ["agentic-simulation", "multi-agent"]
---

# Multi-agent self-correction

## Definition

A pattern where multiple LLM agents with distinct roles (e.g. planner,
formulator, coder, executor, critic) review and correct each other's work over
iterations, using execution errors and physics-based critiques as the
correction signal — as opposed to a single agent emitting a one-shot answer.

## Why it matters here

It is the leading mechanism by which [[agent-as-solver-operator]] loops recover
from buggy decks/scripts and physically implausible results. An open question
for this project: how much of the success is the multi-agent decomposition
versus a strong base model with a simple retry loop? Worth isolating as an
experimental axis.

## Connections

- Introduced for mechanics FEM by [[ni2023mechagents]] (two-agent and
  larger-team variants).
- The correction signal connects to [[physics-grounded-evaluation]]: a critic
  grounded in ground-truth physics is stronger than one grounded in
  self-consistency.
