---
kind: paper
title: "Self-Refining Topology Optimization via an LLM-Based Multi-Agent Framework"
authors: ["Hyunjee Park", "Hayoung Chung"]
institutions: []
year: 2026
venue: arXiv
peer_reviewed: false
url: https://arxiv.org/abs/2605.23273
code_url: null
citations: null
source: "raw/papers/park2026self.pdf"
added: "2026-06-16"
relevance: 4
credibility: 4
status: skimmed
related_experiments: []
related_concepts: [agentic-design-optimization, multi-agent-self-correction]
tags: [topology-optimization, llm-agents, multi-agent, design-automation]
---

# Self-Refining Topology Optimization via an LLM-Based Multi-Agent Framework

## TL;DR

TopOptAgents is a system of six collaborating LLM-based agents that iteratively
automates the topology-optimization workflow — problem formulation, validation,
code generation, and quality assessment — and is reported to converge on problem
classes where a single standard LLM approach struggles.

## Claims

- A multi-agent LLM framework (six agents) can automate the human-in-the-loop
  decisions of a topology-optimization workflow, including parameter adjustment
  and feasibility assessment.
- The framework is especially effective on problem classes with limited
  pretrained-language-model exposure, enabling convergence in cases where a
  standard single-LLM approach fails.
- The approach extends the practical applicability of automated topology
  optimization.

## Methods

- Six LLM-based agents collaborate iteratively across four stages: problem
  formulation, validation, code generation, and quality assessment.
- The agents perform the decisions engineers normally make across the workflow,
  including parameter adjustment and feasibility assessment.
- Topology optimization itself is carried out by well-established numerical
  algorithms; the agents orchestrate and self-refine around that solver.

## Results

- Reported convergence on problem classes that standard LLM approaches struggle
  with (limited pretrained exposure). Specific quantitative results are not
  extracted from the abstract alone.

## Critique / open questions

- Abstract provides no quantitative benchmarks (number of test problems,
  convergence/success rates, compute cost) — these must be read from the full
  PDF.
- No code release noted; reproducibility unclear.
- "Six agents" division of labor and how self-refinement is grounded in the
  numerical solver's feedback needs verification from the body.
- How performance scales with problem complexity, and failure modes, are open.

## Trust signals

- **Credibility:** 4 — concrete, well-scoped engineering contribution on a
  classical, well-defined numerical problem (topology optimization), with a
  clear multi-stage agent architecture. arXiv preprint, not yet peer-reviewed,
  and no code release noted, so held just below maximum.

## Follow-up

- Read the full PDF for the agent roles, the self-refinement loop, and any
  quantitative convergence/success-rate results.
- Relates to [[agentic-design-optimization]] (LLM agents driving a design
  optimization loop) and [[multi-agent-self-correction]] (iterative
  validation/quality-assessment agents correcting each other's outputs).
- Consider an experiment replicating the validation/quality-assessment agent
  loop on a small topology-optimization benchmark.
