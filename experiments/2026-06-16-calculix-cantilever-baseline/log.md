# Log — calculix-cantilever-baseline

Chronological record of what was run and observed.

## 2026-06-16

- Installed CalculiX 2.23 + Gmsh via micromamba env `solidmech` (no sudo; root prefix on SN850X).
- Built harness: cantilever.py (analytical + C3D20R deck gen + ccx runner + .dat parser + grader), reference_sweep.py, agent_operator.py.
- Fixed PyYAML float gotcha (210.0e9 parsed as str → 2.1e+11).
- Fixed convergence metric: mesh convergence = successive solution change → 0, not error-vs-EB → 0 (FEM converges ~0.3% above EB due to shear).
- Reference sweep: mesh-converged, finest deflection err 0.30%, stress err 4.96%.
- Live agentic demo: general-purpose subagent operated ccx unaided → wrote its own genmesh.py + C3D20R deck, 2 runs, 0 errors, deflection 3.82e-4 m (0.275% err, PASS). Reported surface-peak stress 1.32e7 (10% over nominal → stress gate fail, defensible).
- Artifacts (.inp/.dat/.frd) gitignored as regenerable; kept JSON results + agent's genmesh.py.

## 2026-06-17

- Answered "do we need ANTHROPIC_API_KEY?": no. Rewrote agent_operator.py to drive
  `claude -p` headless, which auths off the subscription (~/.claude/.credentials.json),
  billed to the Max token pool. Confirmed: ANTHROPIC_API_KEY unset throughout.
- First sonnet trial (cwd inside project, no turn bound): solved correctly
  (0.13% err in its beam.dat) but ran 15 min → timed out before self-reporting.
  Root cause: in-project cwd auto-loads the whole framework each turn.
- Fix: isolate trials in /tmp + --max-turns + salvage. Re-validated with
  claude-opus-4-8: pass@1 1/1, deflection 0.306% err, 12 turns, ~$0.79-equiv,
  4 ccx runs. Stress 10.9% over nominal (clamp surface peak) — reproducible.
