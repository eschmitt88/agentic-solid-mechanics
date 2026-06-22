# Log — calculix-tapered-design-loop

## 2026-06-22

- Trial 2b: TAPERED cantilever design loop (no closed-form deflection → FE essential).
- fea.py (tapered C3D20R: z-coords scale with local h(x); mass, max-over-length nominal
  stress, EB integral baseline), design.py (taper-ratio scan + FE-bisection on root
  height; grader on n=8), reference_design.py, agent_design.py (claude -p, pass@k).
- FE-true optimum: b=10mm, h_root=148mm, h_tip=37mm, 7.270kg — interior min at taper
  ratio ~0.25, 14.7% lighter than prismatic (8.519kg). EB integral at opt = 2.956mm vs
  FE 3.0mm (~1.5% gap → FE genuinely required; formula-only design would be infeasible).
- Live subagent: 29-design FE search, b=10mm h_root=148mm h_tip=37mm, mass 7.261kg
  (matches optimum, -0.13% vs our grid), feasible (defl 3.011mm within 1% tol, stress
  73MPa). Explicitly noted FE ~0.7% stiffer-bound than EB ("beam theory unsafe").
