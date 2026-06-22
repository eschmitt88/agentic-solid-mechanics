# Log — calculix-cantilever-design-loop

## 2026-06-21

- Trial 2: operator design loop — size cantilever (b,h) to minimise mass s.t.
  deflection ≤ 3mm (FE-graded) + nominal stress ≤ 200MPa (closed-form, per
  trial-1 singularity lesson).
- Built fea.py (reuses trial-1 C3D20R gen), design.py (FE-true reference via
  b-scan + h-bisection, grader on fixed n=8 mesh), reference_design.py,
  agent_design.py (claude -p harness, pass@k).
- Set mesh_n_grade=8 so reference optimum and grader use the same mesh
  (reference design passes its own grader).
- FE-true optimum: b=10mm, h=108.52mm, mass=8.519kg (deflection-binding, stress
  slack). EB optimum 8.501kg; FE needs h +0.2% (shear).
- Live subagent designer: b=10mm (found min-width corner), h=108.6mm, feasible
  (defl 2.993mm), mass 8.525kg = +0.07% above optimum, 3 FE evals. Reasoned
  mass∝b^(2/3), self-corrected EB→FE for shear. Clean (no reference reads).
- Headless agent_design.py (opus, claude -p): feasible 1/1, +0.07% above optimum, b=10mm h=108.6mm, 11 turns. Scriptable design harness validated on subscription (no API key).
- pass@10 (headless claude -p, opus): feasible 10/10, all found min-width corner, +0.0% to +0.3% above optimum (mean +0.10%), median 11.5 turns. Design loop is reliable.
