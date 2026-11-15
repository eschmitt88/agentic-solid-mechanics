import json
INTERP = """**Rung 1.** The solver reproduces the benchmark, and the exercise shows why "reproducing the benchmark" is a
soft target. Three things moved the answer by more than the spread I was graded against:

1. *Boundary conditions nobody prints.* The shear test only matches the published stiffness if the side edges are
   held vertically. None of the papers opened says so; I found it by comparing my elastic stiffness (46 kN/mm)
   with the digitized Miehe curve (63.5 kN/mm) and trying the constraint (63.5 kN/mm).
2. *Mesh.* Peak load falls roughly linearly with h/l. The published shear numbers are coarse-mesh results (Zhou: h ≈ l/2);
   refining to l/8 lowers the shear peak by 7 %. I pass the ±10 % tolerance on the fine mesh, but only because the tolerance is wide.
3. *The staggered scheme.* One pass per step lags the crack. Tension is insensitive once the step is 1e-6; shear is
   not converged at Miehe's step (−2.4 % load for a 4× smaller step, and the location of the maximum jumps). Iterating
   the shear problem to convergence with coarse steps stagnated outright (one step > 3000 passes).

So the honest statement is: within ±10 % of the literature on every pre-registered number, exactly on top of the
literature at the literature's own resolution, and about 5–10 % below it when the numerics are converged.

**Rung 2.** The blind prediction stands on a small, checkable chain: one fracture energy explains four calibration
geometries to within ±9 %; the same model in 3D gives the challenge peak. The weakest links are (a) the model is
7–9 % wrong in opposite directions for the far-offset notch (too strong) and the rotated notch (too weak), and the
challenge geometry combines both; (b) the predicted crack path above mid-height is biased towards vertical (mesh for HB, model form for HA);
(c) the displacement axis is mostly an empirical rod-contact law, not mechanics I computed. The peak-load interval
reflects (a); the crack-path caveat reflects (b).

What surprised me: how much of the "ductile" look of the gypsum curves is the fixture (the beam itself is 2–3×
stiffer than the measured record), and that a larger length scale *improved* the calibration fit for the wrong
reason (it moved the crack away from the notch) — only the crack-path photos exposed that."""

DIAG = """- intended_effect_confirmed: partial — rung 1 yes (12/12 pre-registered checks, `results/rung1_grade.json`); rung 2
  delivered as a frozen prediction for the fallback challenge, not for SFC3, and is ungraded (`results/dmc_prediction.json`).
- leakage_check: rung 1 references were fixed in `PREREGISTRATION.md` before the first run and only sources opened
  this session are used; rung 2 — no outcome source opened, prior memory recorded in `log.md` (21:00). One SFC3
  outcome fragment was seen by accident (`log.md`, 20:35), irrelevant to the challenge actually predicted.
- overfitting_signal: calibration residuals HC −2 %, HB +3 %, HA +8 %, H45 −9 % with one free fracture parameter
  (`results/dmc/calibration_fit.json`) — under-fit rather than over-fit; no held-out calibration geometry was kept back.
- delta_from_prior: first fracture trial; reuses the JAX-autodiff FEM approach of trials 3–5 (`pf.py:115-166`).
- unexpected_findings: published benchmark values are not mesh-converged (shear −7 % at h=l/8,
  `results/pf_run_metrics.json:s_l0075_h8`); shear single-pass scheme not step-converged at the standard step
  (`results/pf_run_metrics.json:s_l015_h4_du2.5e-6`); structured-mesh bias of the 3D crack path
  (`results/dmc/crack_path_validation.json`).
- next_candidates:
  - Get the SFC3 package through a Globus login and run the originally intended ductile-failure prediction in CalculiX.
  - Re-run the challenge geometry on an unstructured or much finer mesh (h ≤ l/8) to remove the crack-path bias before grading the path.
  - Replace the single-pass scheme by a monolithic or over-relaxed staggered solve and repeat the shear benchmark to get a step-converged reference curve."""

FOLLOW = """- Grade `results/dmc_prediction.json` against Morris et al. 2024 (orchestrator; this agent has not seen the outcome).
- If graded poorly on the crack path, check first whether the finer-mesh path (`results/dmc/crack_path_validation.json`) would have been closer.
- Kalthoff–Winkler / asymmetric bending crack-path benchmarks were not attempted; the calibration-beam crack paths (measured, not blind) served as the crack-path test instead."""
json.dump(dict(INTERP_BODY=INTERP, DIAG_BODY=DIAG, FOLLOW_BODY=FOLLOW), open("report_src/readme_static.json","w"))
