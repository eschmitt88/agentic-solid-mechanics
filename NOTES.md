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
- Decision pending from user: build trial 1, or hold.
- Borrow FEM-Bench's pass@5 + EngDesign's sim-in-the-loop scoring for HCE grading.
