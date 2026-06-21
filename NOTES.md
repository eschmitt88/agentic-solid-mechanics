# NOTES

Running log of work sessions. `/wrap` appends a new dated section at the
end of each session with **Did / Findings / Next** subsections. The
SessionEnd hook backstops this if you forget.

<!-- entries go below this line, newest at bottom -->

## 2026-06-16 — Project founding: scope, literature base, solver decision

### Did
- Scaffolded the project (repo + GitHub Pages, `agency: max`, RAM=91GB). Set the
  two-loop framing in README/CLAUDE: (1) agent-as-operator, (2) differentiable
  inverse design; single-physics solid mechanics first.
- Ran `/discover` (3 parallel sweeps → 15-entry triage) on agentic simulation,
  differentiable FEM, and the open-source solver landscape.
- **ADR 0001**: committed **CalculiX** (Abaqus-style `.inp` text decks) as the
  operator solver — cleanest text-in/text-out surface for an agent. JAX-FEM is
  the working assumption for loop 2 (its own ADR when that loop starts).
- Ingested all 15 ranked candidates (10 papers + 5 solver repos), declined 6
  honorable mentions with reasons, archived the triage to `_done/`.
- Built the concept graph: 6 concepts across 3 agentic design axes
  (drive / differentiate / write-from-scratch a solver) + cross-cutting
  (design-optimization, self-correction, physics-grounded-evaluation).
  All wikilinks resolve.

### Findings
- The field already ships the evaluation harnesses we'd want — **FEM-Bench**
  (objective pass@5) and **EngDesign** (simulation-in-the-loop scoring) — which
  map cleanly onto our HCE rule. **ALL-FEM** (2026, FEniCS, 39-task benchmark)
  is the closest published analog; **Foam-Agent** is the best "agent drives a
  CLI solver" architecture (RAG + dependency-aware scheduling, MCP tools).
- Known agent failure modes are specific: geometric nonlinearity,
  buckling/eigenvalue problems, writing discriminative tests → a ready-made
  difficulty ladder for trials.
- Install caveats logged: FEniCSx needs conda (this box is uv-only → reason we
  picked CalculiX); JAX-FEM needs recent jaxlib (CUDA 12.6+) for the RTX 5080's
  Blackwell sm_120 before loop-2 trials.

### Next
- Trial roadmap (in README, not yet authorized to build):
  1. CalculiX cantilever operator baseline (agent writes deck → runs → parses →
     validates vs. analytical beam + mesh convergence).
  2. Operator design loop (min mass s.t. stress constraint, iterative).
  3. Differentiable inverse design on JAX-FEM (topology / param ID, GPU).
- Borrow FEM-Bench's pass@5 + EngDesign's sim-in-the-loop scoring for HCE grading.

### Did (cont.) — Trial 1 built and run
- Installed CalculiX 2.23 + Gmsh via micromamba (`solidmech` env, no sudo, on SN850X).
- `experiments/2026-06-16-calculix-cantilever-baseline/`: full harness —
  `cantilever.py` (analytical + C3D20R deck gen + ccx runner + .dat parser +
  grader), `reference_sweep.py` (deterministic ground truth), `agent_operator.py`
  (automated LLM operator loop w/ run_ccx + submit_answer tools, pass@k).
- Ran it: deterministic reference mesh-converged (0.30% deflection err); a live
  subagent operated ccx **unaided** and hit **0.275% deflection error (PASS)**,
  authoring its own mesh generator + deck, 2 runs, 0 errors.

### Findings (cont.)
- Agent-as-operator loop works end to end on the cantilever; the agent reasoned
  about physics (chose C3D20R to avoid shear locking, explained FEM>EB shear gap).
- Stress grading exposed a real subtlety: agent reported surface-peak vM (incl.
  clamp stress concentration, 10% over nominal) vs the grader's nominal bending
  stress — a metric-definition gap to fix, not an agent error.
- Two harness bugs found+fixed: PyYAML needs signed exponents (`2.1e+11`);
  mesh convergence = successive-solution-change→0, not error-vs-EB→0.

### Next (updated)
- `agent_operator.py --trials 10` for pass@k + deck-error rate (needs ANTHROPIC_API_KEY).
- Trial 2 (operator design loop: size section to a stress constraint).
- Add a problem *distribution* + HCE held-out test split; disambiguate the
  stress metric and add a .frd von-Mises parser.

## 2026-06-21 — Trial 2: operator design loop

### Did
- Built `experiments/2026-06-21-calculix-cantilever-design-loop/`: size a
  cantilever (b,h) to minimise mass s.t. tip deflection ≤ 3mm (FE-graded) +
  nominal bending stress ≤ 200MPa (closed-form — applies trial-1's lesson that
  the FE clamp-stress peak is a mesh-dependent singularity, ill-posed as a constraint).
- `fea.py` (reuses trial-1 C3D20R gen), `design.py` (FE-true reference via b-scan +
  h-bisection; grader on fixed n=8 mesh so reference passes its own grader),
  `reference_design.py`, `agent_design.py` (claude -p harness, pass@k, no API key).
- FE-true optimum: b=10mm, h=108.5mm, mass=8.519kg (deflection-binding, stress slack).

### Findings
- Live subagent designer: **+0.07% above the FE-true optimum**, feasible, 3 FE evals.
  Found the non-obvious corner solution (drive width b→min; reasoned mass∝b^(2/3) at
  the deflection limit) AND closed the FE loop (noticed FE deflection > Euler-Bernoulli
  seed due to shear, nudged h up). Genuine constrained optimisation, not a blind sweep.
- This beam is slender (L/h≈9) so the EB-vs-FE gap is only ~0.2% — feasibility is
  easy; the discriminator is optimality. To make FE *essential*, next variants need a
  stubbier/tapered section or both constraints active.

### Next
- `agent_design.py --trials 10` → feasible-rate + mass-gap distribution (pass@k).
- Harder variants: both constraints active (raise P), or tapered/stepped section (no
  closed form) so FE is genuinely in the loop.
- Trial 3: differentiable inverse design on JAX-FEM (loop 2, GPU).
