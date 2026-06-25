---
kind: paper
title: "Effect of interior venetian blinds on natural convective heat exchange at a full-scale window glazing"
authors: []
institutions: []
year: 2022
venue: Applied Thermal Engineering
peer_reviewed: true
url: https://www.sciencedirect.com/science/article/abs/pii/S1359431122013928
code_url:
citations:
source: "(workflow research 2026-06-25)"
added: "2026-06-25"
relevance: 4
credibility: 3
status: skimmed
related_experiments: []
related_concepts:
  - boussinesq-natural-convection
tags: [natural-convection, venetian-blind, fenestration, validation]
---

# Effect of interior venetian blinds on natural convective heat exchange at a full-scale window glazing

## TL;DR

A 2D CFD study (Applied Thermal Engineering, 2022; Zheng, Jiang, Bisengimana, Zhao, Yuan; preprint DOI 10.2139/ssrn.4188691) of buoyancy-driven natural convection at a full-scale interior window with horizontal venetian-blind slats. It maps how slat angle, slat-to-window spacing, and air-to-window temperature difference reshape the near-window flow, identifying three flow regimes — Duct-Directed Flow (DDF), Transition Flow (TF), and Louver-Directed Flow (LDF) — and ties the local and average convective heat-exchange rate directly to which regime dominates. It is a domain-physics reference, not…

## Claims

- 2D finite-volume CFD of natural (buoyancy-driven) convection at a window with venetian-blind slats
- Parametric sweep over slat angle, slat-to-window spacing, and temperature difference
- Identification/classification of near-window convective flow regimes (DDF/TF/LDF) and their heat-transfer consequences

## Critique / open questions

Claims as tagged hold up: it is a 2D CFD + parametric study of venetian-blind natural convection at a window, with no GPU and no differentiability — no overstatement in the supplied tags. The substantive findings (DDF/TF/LDF regimes, ~1.7x LDF/DDF ratio, up to 45%/70% sensitivities) are corroborated across SSRN, ResearchGate, and ScienceDirect listings. Caveats: it is paywalled, so the numerical fidelity (turbulence model, validation, Boussinesq specifics) is unverified from open sources; treat the quantitative targets as approximate until the full text is obtained. The earlier phrasing 'exper…
- Paywalled journal article (ScienceDirect/ResearchGate return HTTP 403); only the abstract and headline results are openly accessible without institutional access or the SSRN preprint.
- Not a tool or code release — no software artifact, no repository, nothing installable or runnable; it is a physics-result reference only.
- Exact numerics (mesh, turbulence/laminar treatment, solver name, Boussinesq vs full property variation, validation against experiment) were not extractable from open sources and could not be independently verified here.
- Findings are reported for the slat-close-to-window regime (n<=10 mm emphasized); generality to large blind-window gaps is not established from the accessible material.

## Trust signals

- **Credibility:** N/A — static 2022 journal publication (Applied Thermal Engineering); not a maintained software project, so no release/commit cadence applies.

## Relevance here

- **Project:** High for the immediate CFD/natural-convection thrust as a ground-truth/benchmark source; low for the broader loop-1/loop-2 agentic and differentiable-design machinery, since it offers no method or tooling — only target physics.
- **Window-shade:** Directly on-target. This is essentially the canonical physics reference for the planned 2D venetian-blind window-shade natural-convection problem (effective h vs slat angle, Boussinesq buoyancy): same geometry, same driving question, with quantitative regime-based h(angle) behavior to validate again…

## Follow-up

- ...
