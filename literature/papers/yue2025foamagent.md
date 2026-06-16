---
kind: paper
title: "Foam-Agent: Towards Automated Intelligent CFD Workflows"
authors: ["Ling Yue", "Nithin Somasekharan", "Tingwen Zhang", "Yadi Cao", "Zhangze Chen", "Shimin Di", "Shaowu Pan"]
institutions: ["Rensselaer Polytechnic Institute"]
year: 2025
venue: "arXiv preprint"
peer_reviewed: false
url: https://arxiv.org/abs/2505.04997
code_url: https://github.com/csml-rpi/Foam-Agent
citations: null
source: "raw/papers/yue2025foamagent.pdf"
added: "2026-06-16"
relevance: 4
credibility: 4
status: skimmed
related_experiments: []
related_concepts: ["agent-as-solver-operator", "multi-agent-self-correction"]
tags: ["agentic-simulation", "cfd", "openfoam", "rag", "mcp", "architecture"]
---

# Foam-Agent

## TL;DR

The strongest end-to-end "agent operates a real, CLI-driven simulator"
architecture to date: a multi-agent framework that runs the full OpenFOAM
pipeline (mesh → config → HPC scripting → run → post-processing) from one
prompt, hitting **88.2% execution success** on 110 tasks — directly transferable
patterns even though the physics is CFD.

## Claims

- RAG + dependency-aware scheduling synthesizes high-fidelity simulation
  configurations from natural language.
- Exposes its core functions as **MCP tools**, so other agentic systems can call
  it.
- 88.2% execution success without expert intervention on 110 tasks (SOTA).

## Methods

- Multi-agent orchestration over the OpenFOAM workflow stages.
- Retrieval-augmented generation grounded in OpenFOAM case files/docs.
- Dependency-aware scheduling to order interdependent config steps.

## Results

- 88.2% end-to-end execution success on 110 CFD tasks.

## Critique / open questions

- OpenFOAM, like CalculiX, is **text-config + CLI** driven — the closest
  workflow analog to our operator loop despite being fluids, not solids.
- "Execution success" measures runs-without-error, not physical correctness vs.
  ground truth — the gap [[physics-grounded-evaluation]] targets.
- RAG over solver docs/cases is a concrete, portable idea for a CalculiX agent.

## Trust signals

- **Credibility:** 4 — RPI group; code released; strong, clearly-measured
  results; preprint.

## Follow-up

- Adopt RAG-over-CalculiX-keyword-manual and dependency-aware deck assembly for
  our operator trials. Consider exposing our CalculiX runner as an MCP tool.
