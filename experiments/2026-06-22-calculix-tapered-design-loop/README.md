---
kind: experiment
slug: "calculix-tapered-design-loop"
date: "2026-06-22"
status: done
hypothesis: "On a TAPERED cantilever — which has no closed-form tip-deflection, so FE is genuinely required in the loop (not a small correction to a beam formula) — an LLM agent can run a multi-variable design search (b, h_root, h_tip) and reach the FE-true minimum-mass taper, beating the prismatic optimum."
result: "Confirmed strongly. The agent ran a 29-design FE search, drove width to its lower bound, and converged on b=10mm, h_root=148mm, h_tip=37mm = 7.261 kg — matching the FE-true optimum (7.270 kg, −0.13%) and 14.7% lighter than the prismatic optimum. It explicitly noted FE ran ~0.7% stiffer-bound than Euler-Bernoulli ('beam theory alone was unsafe') and used FE as the binding truth."
related_concepts: ["agentic-design-optimization", "agent-as-solver-operator", "physics-grounded-evaluation"]
related_literature: ["park2026self", "guo2025engdesign", "calculix"]
tags: ["calculix", "design-loop", "trial-2b", "tapered", "fe-essential", "optimization"]
---

# calculix-tapered-design-loop

Trial 2b: the **harder design loop**. Trial 2 (prismatic) was slender enough that
the beam formula nearly sufficed (FE-vs-EB gap ~0.2%). Here the section is
**tapered** (linear height h_root → h_tip), which has **no elementary
tip-deflection formula** — the agent must close the FE loop. Builds on
`../2026-06-21-calculix-cantilever-design-loop/`.

## Setup

- **Problem (`config.yaml`):** same cantilever as trial 2 (L=1.0 m, P=2000 N,
  steel) but the section tapers linearly. Design vars: b∈[10,80] mm,
  h_root∈[10,250] mm, h_tip∈[10,250] mm. Minimise mass = ρ·b·L·(h_root+h_tip)/2
  subject to FE tip deflection ≤ 3.0 mm and max nominal bending stress
  max_x 6P(L−x)/(b·h(x)²) ≤ 200 MPa.
- **Why FE is essential here:** a tapered Euler-Bernoulli deflection requires
  integrating M/(EI(x)) and still neglects shear. At the optimum the EB integral
  reads 2.956 mm vs the FE 3.0 mm — a ~1.5% gap (vs 0.2% prismatic), enough that
  a formula-only design lands infeasible. So the constraint genuinely needs
  simulation ([[physics-grounded-evaluation]]).
- **Code:** `fea.py` (tapered C3D20R deck — element z-coords scale with local
  height h(x) — + mass, max-over-length nominal stress, EB integral baseline),
  `design.py` (taper-ratio search with FE-bisection on root height; grader on a
  fixed n=8 mesh), `reference_design.py`, `agent_design.py` (`claude -p` harness,
  subscription auth, pass@k).
- **Solver:** CalculiX 2.23 via micromamba `solidmech`. See [[calculix]].

## Reference optimum (FE-true, `results/reference.json`)

Width fixed at its lower bound (mass monotone in b; b_max check = 30.0 kg ≫
optimum confirms the corner). Taper-ratio scan (root height bisected to the FE
deflection cap at each ratio):

| r = h_tip/h_root | h_root (mm) | h_tip (mm) | mass (kg) | σ_max (MPa) |
|---|---|---|---|---|
| 1.00 (prismatic) | 108.5 | 108.5 | 8.519 | 101.9 |
| 0.50 | 128.0 | 64.0 | 7.534 | 73.3 |
| **0.25** | **148.2** | **37.0** | **7.270** | 72.9 |
| 0.18 | 157.8 | 28.4 | 7.309 | 81.6 |
| 0.12 | 169.6 | 20.3 | 7.454 | 98.8 |

- **FE-true tapered optimum: b=10 mm, h_root=148 mm, h_tip=37 mm, 7.270 kg** —
  a clean interior minimum at r≈0.25, **14.7% lighter than the prismatic
  optimum** (8.519 kg). Deflection-binding; stress slack.

## Result

Points at `metrics.json`. Designer = in-session general-purpose subagent,
isolated in `results/agent_demo/` (verified: no reference/harness reads).

- **b=10 mm, h_root=148 mm, h_tip=37 mm** — matches the FE-true optimum.
- Feasible on the independent n=8 grade: deflection **3.011 mm** (within the 1%
  tolerance of the 3.0 mm cap), stress 73 MPa, in bounds.
- Mass **7.261 kg = −0.13% vs our reference** (the agent's continuous search
  slightly beat our 7-point grid) and **14.8% lighter than prismatic.**
- **29 distinct designs** FE-evaluated; explicitly observed FE > EB (~0.7%) and
  treated FE as binding.

## Interpretation

The agent ran a genuine multi-variable design search and recovered the FE-true
taper optimum. Two things make this a stronger result than trial 2: (1) the
search space is 3-D with an interior optimum (it found the right taper ratio,
not a bound), and (2) FE was genuinely *required* — the agent itself noted beam
theory was unsafe and let the simulation drive the design. The final point sits
right on the deflection boundary (3.011 mm on the independent finer grade), which
is exactly where the ~1.5% formula error would have caused a formula-only agent
to fail — concrete evidence that closing the FE loop mattered.

## Diagnostics

Unless noted, numbers reference `metrics.json`.

- intended_effect_confirmed: yes — feasible taper at the FE-true optimum
  (`metrics.json:agent_designer_demo`, pct_above_optimum=−0.13%), 14.8% lighter
  than prismatic.
- leakage_check: `grep` over `results/agent_demo/` clean; agent authored its own
  tapered deck generator.
- overfitting_signal: n/a — single design instance.
- delta_from_prior: vs trial 2 (prismatic, +0.07%), this adds a 3-var search
  with FE genuinely binding; agent used 29 FE evals and matched the optimum.
- unexpected_findings: agent independently flagged the EB-vs-FE gap ("beam
  theory unsafe") — i.e. it discovered the FE-essential property the trial was
  designed to test.
- seeds_run: 1 live subagent demo; `agent_design.py --trials N` for pass@k.
- metric_aggregation: single design; reference is a taper-ratio scan + bisection.
- next_candidates:
  - `agent_design.py --trials 10` for feasible-rate + mass-gap distribution.
  - Make stress binding (raise P) so the optimum is a stress/deflection
    trade-off; or free the taper profile (piecewise) for a richer search.

## Follow-up

- Trial 3: differentiable inverse design on JAX-FEM (loop 2, GPU).
