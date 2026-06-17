---
kind: experiment
slug: "calculix-cantilever-baseline"
date: "2026-06-16"
status: done
hypothesis: "An LLM agent, given only an engineering spec and a CalculiX runner, can autonomously author a valid .inp deck, run it, parse the result, and reach a tip-deflection answer within 5% of the Euler-Bernoulli analytical solution — and the same harness establishes the deterministic ground truth that grades it."
result: "Confirmed for deflection: the operator agent authored its own C3D20R deck, ran ccx (2 runs, 0 deck errors), checked convergence, and hit 0.275% deflection error (PASS). Stress (1.32e7 Pa, surface peak incl. clamp concentration) was 10% over nominal — a defensible physical nuance, not an agent error. Deterministic reference converges to 0.30% deflection / 4.96% stress."
related_concepts: ["agent-as-solver-operator", "physics-grounded-evaluation"]
related_literature: ["ni2023mechagents", "deotale2026allfem", "mohammadzadeh2025fembench", "calculix"]
tags: ["calculix", "operator-loop", "trial-1", "cantilever", "ground-truth"]
---

# calculix-cantilever-baseline

Trial 1 of the roadmap: the **agent-as-operator** baseline on **CalculiX**.

## Hypothesis

Given only a natural-language engineering spec (a slender steel cantilever with
an end load) and a tool that runs CalculiX on a `.inp` deck, an LLM agent can
autonomously:

1. author a valid 3D `.inp` deck (geometry, mesh, material, BCs, load),
2. run `ccx` and recover from any deck errors,
3. parse the tip deflection from the output, and
4. report an answer within **5%** of the Euler-Bernoulli analytical deflection.

This simultaneously tests the agentic loop and exercises the deterministic
grading harness ([[physics-grounded-evaluation]]).

## Setup

- **Config:** `config.yaml` (problem spec, mesh sweep, tolerances).
- **Code:**
  - `cantilever.py` — analytical Euler-Bernoulli solution, `.inp` deck
    generator, `ccx` runner, `.dat`/`.frd` parser, grader.
  - `reference_sweep.py` — deterministic mesh-convergence reference (no LLM):
    runs the templated deck at each mesh level, validates vs analytical, writes
    `results/reference.json`. Proves the solver + parser + ground truth.
  - `agent_operator.py` — the automated agentic harness: drives the **Claude
    Code CLI headless** (`claude -p`) once per trial; the agent authors its own
    deck, runs `ccx` via Bash, and returns numbers under a JSON schema, graded
    against the reference. **No `ANTHROPIC_API_KEY`** — `claude -p` uses the
    logged-in subscription (`~/.claude/.credentials.json`), billed against the
    Max token pool. Each trial runs isolated in `/tmp` (no project-framework
    autoload; cannot read the reference). pass@k via `--trials`.
- **Solver:** CalculiX 2.23 via micromamba env `solidmech`
  (`micromamba run -n solidmech ccx <job>`). See [[calculix]].
- **Data:** none — the ground truth is closed-form, so no HCE split (that
  applies once trial design uses a *distribution* of problems).

## Analytical reference (Euler-Bernoulli, end-loaded cantilever)

Rectangular section `b×h`, second moment `I = b·h³/12`, fibre distance `c = h/2`:

- tip deflection  `δ = P·L³ / (3·E·I)`
- max bending stress at the fixed end  `σ = P·L·c / I`

For the config values (L=1.0, b=0.05, h=0.10, E=210 GPa, P=1000 N):
`I = 4.167e-6 m⁴`, `δ ≈ 3.81e-4 m (0.381 mm)`, `σ ≈ 12.0 MPa`.
A 3D solid FEM will exceed EB deflection slightly (shear), so grading uses a
tolerance and a convergence trend, not exact equality.

## Result

Points at `metrics.json`.

**Deterministic reference** (`reference_sweep.py`, no LLM) — mesh sweep
n∈{1,2,4,8} elements through height:

| n | elements | tip deflection (m) | err vs EB | max vM (Pa) |
|---|---|---|---|---|
| 1 | 2   | 4.327e-4 | 13.60% | 5.47e6 |
| 2 | 10  | 3.795e-4 | 0.38%  | 8.27e6 |
| 4 | 80  | 3.817e-4 | 0.19%  | 1.05e7 |
| 8 | 640 | 3.821e-4 | 0.30%  | 1.14e7 |

Mesh-converged (successive change → 0.11%). The finest result lands ~0.30%
*above* EB — the correct 3D answer, since EB neglects shear.

**Agent operator** (`general-purpose` subagent, results in
`results/agent_run/`, graded in `results/agent_result.json`):

- Authored its own `genmesh.py` + C3D20R decks; **never read** the reference
  or `cantilever.py` (verified). Refined 20×2×4 → 40×4×8 to check convergence.
- **Tip deflection 3.82e-4 m → 0.275% error → PASS** (tol 5%).
- Max von Mises 1.32e7 Pa (extrapolated surface peak) → 10.0% over nominal →
  fails the 10% stress gate; its integration-point value 1.13e7 (5.8%) passes.
- 2 ccx runs, 0 deck errors.

## Interpretation

The agent-as-operator loop works on this problem with no human help: spec →
deck → run → parse → convergence check → answer, landing essentially on the
analytical deflection. Notably the agent independently chose to *write a mesh
generator* rather than a static deck, picked a bending-appropriate element
(C3D20R, avoiding shear locking), and explained the FEM>EB gap via shear — i.e.
it reasoned about the physics, not just the syntax.

The "stress failure" is the interesting part: the agent reported the
physically-meaningful surface peak (which includes the real stress
concentration at a fully-encastre constraint), while my tolerance was set
against the *nominal* bending stress. This is a grading-definition gap, not an
agent mistake — and it surfaces exactly the kind of subtlety
([[physics-grounded-evaluation]]) we want trial design to handle: agreeing on
*which* stress quantity is being graded, and how to treat constraint
singularities.

## Diagnostics

Unless noted, numbers reference `metrics.json`.

- intended_effect_confirmed: yes — agent reached 0.275% deflection error
  unaided (`metrics.json:agent_operator.deflection_pass=true`).
- leakage_check: verified the agent did not read `cantilever.py` /
  `reference.json` — `grep` over `results/agent_run/` clean; it authored
  `results/agent_run/genmesh.py` from scratch.
- overfitting_signal: n/a — single analytical-graded problem, no train/val split.
- delta_from_prior: first experiment; no prior. Reference vs analytical:
  0.30% deflection (`metrics.json:reference.finest_deflection_rel_err`).
- unexpected_findings: agent wrote a parametric mesh generator (not a static
  deck) and reported surface-peak stress incl. the clamp singularity, exposing
  a stress-definition ambiguity in the grader.
- seeds_run: 1 live agent trial (the `agent_operator.py` harness supports
  pass@k via --trials for nondeterminism characterization).
- metric_aggregation: single run; reference is a 4-level mesh sweep.
- next_candidates:
  - Run `agent_operator.py --trials 10` to measure pass@k and deck-error rate
    across nondeterministic runs (uses the Claude subscription via `claude -p`,
    no API key). Note: headless trials are far slower than the in-session
    subagent (~15 min vs ~90 s) — bound with `--max-turns`, isolate in `/tmp`.
  - Add a problem *distribution* (vary L/b/h/load/BCs) and introduce an HCE
    held-out test split, grading deflection against a per-problem analytical or
    converged-FEM reference.
  - Disambiguate the stress metric (nominal vs surface-peak) and add a
    von-Mises-from-.frd parser to the deterministic harness.

## Follow-up

- Trial 2 (operator design loop): size the section to a stress constraint.
- Compare deck-writing (this trial) against hand-rolled-Python-FEM
  ([[code-from-scratch-numerical-solver-agent]]) on the same problem — the
  agent already drifted toward code-from-scratch by writing a mesh generator.

## Follow-up

- Trial 2 (operator design loop): size the section to a stress constraint.
- Compare deck-writing (this trial) against hand-rolled-Python-FEM
  ([[code-from-scratch-numerical-solver-agent]]) on the same problem.
