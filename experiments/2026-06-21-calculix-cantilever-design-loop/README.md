---
kind: experiment
slug: "calculix-cantilever-design-loop"
date: "2026-06-21"
status: done
hypothesis: "Given a sizing objective (minimise mass) under deflection + stress constraints, an LLM agent can run the design loop — propose (b,h) → mesh → solve → measure → revise — to reach a feasible design near the FE-true optimum, including discovering the non-obvious corner solution (minimise width) and the shear correction the beam formula misses."
result: "Confirmed. The designer agent drove b to its lower bound (correct corner solution), seeded h with Euler-Bernoulli, observed the FE shear correction and nudged h up, and landed at a FEASIBLE design 8.525 kg = +0.07% above the FE-true optimum (8.519 kg), in 3 FE-evaluated candidates. Independent n=8 grade: deflection 2.993 mm ≤ 3.0, nominal stress 101.7 MPa ≤ 200, within bounds."
related_concepts: ["agentic-design-optimization", "agent-as-solver-operator", "physics-grounded-evaluation"]
related_literature: ["park2026self", "guo2025engdesign", "calculix"]
tags: ["calculix", "design-loop", "trial-2", "optimization", "cantilever-sizing"]
---

# calculix-cantilever-design-loop

Trial 2 of the roadmap: the **operator design loop** — the agent *sizes* a part
to meet constraints, iterating geometry → mesh → solve → check → revise. Builds
on trial 1 (`../2026-06-16-calculix-cantilever-baseline/`).

## Hypothesis

Given a minimise-mass objective under a deflection + stress constraint, an LLM
agent can run the design loop to a feasible, near-optimal section — including
(a) the non-obvious corner solution (minimise *width*, since stiffness-per-mass
favours height) and (b) the shear correction the Euler-Bernoulli formula misses
(so a formula-only design lands slightly off). Tests multi-step agentic
iteration ([[agentic-design-optimization]]), not just one-shot deck authoring.

## Setup

- **Problem (`config.yaml`):** cantilever L=1.0 m, tip load P=2000 N, steel
  (E=210 GPa, ρ=7850). Design vars: b∈[10,80] mm, h∈[10,200] mm. Minimise
  mass = ρ·L·b·h subject to: tip deflection ≤ **3.0 mm** (graded by FE) and
  nominal bending stress 6PL/(bh²) ≤ **200 MPa**.
- **Why nominal stress, not the FE peak:** trial 1 showed the FE von Mises peak
  at a fully-encastre face is a mesh-dependent singularity — ill-posed as a
  constraint. So the stress constraint is graded on the closed-form nominal
  stress; the deflection constraint (which converges cleanly) is graded by FE.
  Direct application of [[physics-grounded-evaluation]].
- **Code:** `fea.py` (C3D20R deck gen + ccx runner + parser + mass/stress
  helpers), `design.py` (problem, FE-true reference optimum via b-scan +
  h-bisection, grader on a fixed n=8 mesh), `reference_design.py`,
  `agent_design.py` (automated `claude -p` harness, subscription auth, pass@k).
- **Solver:** CalculiX 2.23 via micromamba env `solidmech`. See [[calculix]].
- **Data:** none — reference optimum is computed by FE; no HCE split (single
  problem instance).

## Reference optimum (FE-true, `results/reference.json`)

The b-scan (min h meeting the deflection cap at each b) confirms mass is
**monotone increasing in b** → optimum at the width lower bound:

| b (mm) | min h (mm) | mass (kg) | nominal σ (MPa) |
|---|---|---|---|
| **10** | **108.52** | **8.519** | 101.9 |
| 24 | 80.94 | 15.25 | 76.3 |
| 38 | 69.40 | 20.70 | 65.6 |
| … | … | … | … |
| 80 | 54.08 | 33.96 | 51.3 |

- **FE-true optimum: b=10 mm, h=108.5 mm, mass=8.519 kg** (deflection-binding,
  stress slack). Euler-Bernoulli optimum: h=108.29 mm, 8.501 kg — the FE needs
  h ~0.2% larger because true deflection exceeds EB (shear).

## Result

Points at `metrics.json`. Designer = in-session general-purpose subagent,
isolated in `results/agent_demo/` (verified: never read the reference / harness).

- **b = 10.0 mm** (drove width to its lower bound — the corner solution),
  **h = 108.6 mm**.
- Feasible on the independent n=8 grade: deflection **2.993 mm ≤ 3.0**, nominal
  stress **101.7 MPa ≤ 200**, within bounds.
- Mass **8.525 kg = +0.07% above the FE-true optimum.**
- 3 FE-evaluated candidates; explicitly reasoned mass ∝ b^(2/3) at the
  deflection limit, and observed FE deflection > EB (shear) → nudged h up.

## Interpretation

The agent ran a genuine design loop and effectively solved the optimisation: it
(1) derived the non-obvious corner solution (minimise width) from a
stiffness-per-mass argument rather than brute-forcing the 2-D space, (2) seeded
with the beam formula, and (3) *closed the FE loop* — it noticed the FE
deflection exceeded the Euler-Bernoulli estimate and tuned h to satisfy the FE
constraint, not the formula. The +0.07% mass gap is within FE/bisection noise of
the true optimum.

This is a stronger agentic result than trial 1: trial 1 was one-shot analysis;
this is constrained optimisation with simulation in the loop, reaching the
optimum efficiently (3 evaluations, not a blind sweep).

## Diagnostics

Unless noted, numbers reference `metrics.json`.

- intended_effect_confirmed: yes — feasible design +0.07% above optimum
  (`metrics.json:agent_designer_demo.pct_above_optimum`), found the min-width
  corner (`found_min_width_corner=true`).
- leakage_check: verified the agent did not read `reference.json` / `design.py`
  / `fea.py` — `grep` over `results/agent_demo/` clean; it authored its own deck.
- overfitting_signal: n/a — single design instance, no train/val split.
- delta_from_prior: vs trial 1 (one-shot analysis, 0.275% deflection err), this
  adds the optimisation loop; agent reached the optimum in 3 FE evals.
- unexpected_findings: agent independently produced the mass ∝ b^(2/3) argument
  for the corner solution and self-corrected for shear vs the EB seed.
- seeds_run: 1 live subagent demo; `agent_design.py --trials N` gives pass@k.
- metric_aggregation: single design; reference is an FE b-scan + h-bisection.
- next_candidates:
  - Run `agent_design.py --trials 10` for feasible-rate + mass-gap distribution
    across nondeterministic runs (subscription via `claude -p`, no API key).
  - Make the stress constraint binding (raise P / tighten bounds) so both
    constraints are active and the corner shifts — harder optimisation.
  - Tapered or stepped section (no closed form) to force FE genuinely into the
    loop rather than as a small correction to a formula.

## Follow-up

- Trial 3: differentiable inverse design on JAX-FEM (gradient-based topology /
  parameter ID) — the loop-2 substrate.
- Compare against the deck-writing vs hand-rolled-Python-FEM axis
  ([[code-from-scratch-numerical-solver-agent]]).
