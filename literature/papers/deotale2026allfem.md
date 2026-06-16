---
kind: paper
title: "ALL-FEM: Agentic Large Language models Fine-tuned for Finite Element Methods"
authors: ["Rushikesh Deotale", "Adithya Srinivasan", "Yuan Tian", "Tianyi Zhang", "Pavlos Vlachos", "Hector Gomez"]
institutions: ["Purdue University"]
year: 2026
venue: "arXiv preprint"
peer_reviewed: false
url: https://arxiv.org/abs/2603.21011
code_url: null
citations: null
source: "raw/papers/deotale2026allfem.pdf"
added: "2026-06-16"
relevance: 5
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["agent-as-solver-operator", "multi-agent-self-correction", "physics-grounded-evaluation"]
tags: ["agentic-simulation", "fem", "fenics", "fine-tuning", "benchmark"]
---

# ALL-FEM

## TL;DR

The closest published analog to this project: an agentic system pairing
**fine-tuned open-weight LLMs (3B–120B)** with a multi-agent orchestrator to
generate, correct, and validate **FEniCS** code across solid mechanics, fluids,
and coupled physics — beating non-agentic GPT-5 Thinking when deployed inside
the agentic loop.

## Claims

- A fine-tuned model reaches **71.79% code success** on a 39-benchmark suite
  spanning elasticity, plasticity, fluids, thermodynamics, FSI, transport.
- Agentic structure + runtime validation lets a smaller fine-tuned model exceed
  a much larger non-agentic frontier model (GPT-5 Thinking).
- A corpus of **1000+ validated FEniCS scripts** (500+ expert + LLM-retrieval
  generated) makes the fine-tuning possible.

## Methods

- Multi-agent framework: agents translate problem → PDE, produce + correct code,
  display results.
- Fine-tuning across a 3B–120B model range on the validated FEniCS corpus.
- Runtime validation as the correction/grading signal.

## Results

- Best model 71.79% success across the 39 tasks; agentic > non-agentic at equal
  or smaller model size.

## Critique / open questions

- Uses FEniCS (Python API); we commit to **CalculiX text decks** — does the
  deck-vs-API axis change fine-tuning value and success rate?
- "Code success" (runs + plausible) vs. graded-against-ground-truth accuracy:
  how strict is the validation? Relates to [[physics-grounded-evaluation]].
- Fine-tuning a corpus is heavy; can prompt-only agentic loops close the gap on
  our hardware?

## Trust signals

- **Credibility:** 4 — Purdue mechanics/CS group (Gomez, Vlachos, Tianyi Zhang);
  preprint, not yet peer-reviewed, but methodologically detailed with a real
  benchmark.

## Follow-up

- Mine its 39-task benchmark structure for our CalculiX trial design.
- Consider whether a small local fine-tune is worth it vs. prompt-only on this
  box (RTX 5080) — note for a future experiment.
