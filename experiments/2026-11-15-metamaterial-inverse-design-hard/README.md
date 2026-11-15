---
kind: experiment
slug: "metamaterial-inverse-design-hard"
date: "2026-11-15"
status: done
hypothesis: "Trial 9's differentiable inverse-design loop, pushed on its four weak points, can (1) reach harder pre-registered targets or report honestly that it cannot, (2) raise the share of fabrication-perturbed copies inside tolerance by designing for the error model, (3) converge without hand intervention from the starts that stalled yesterday once fold points are treated as explicit unknowns and an automatic outer loop is added, and (4) keep its designs inside tolerance when the verifier drops plane strain or the rigid frame."
result: "None of the three new targets passed in CalculiX: H1 foam plateau NRMSE 0.081 vs 0.03, H2 two-cell staircase 0.217 vs 0.04, H4 (expected infeasible) FAIL-unverifiable with an energy floor of 0.397 vs 0.05. Robust design gave no gain (CalculiX pass fraction 58 % nominal vs 58 % robust; design solver 47 % vs 42 %) because force scales as thickness^2.96. Fold-point sensitivities are exact (≤ 9e-6 vs FD) but the stall is a branch-flip discontinuity at a pitchfork, not a moving snap; success from the 11 stalled starts: trial 9's method 0, pre-registered remedy 1 (bar was 8), post-hoc eigenvalue-margin variant 5 (1 of 6 on fresh starts, 0 of 6 more with doubled budget). The bistable design passes in CalculiX (≤ 0.27 %) only on the finest ladder mesh. Relaxing idealisations moves curves by 5–11 % of range (3D, forces ×0.90) and 11–20 % (elastic frame, forces ×0.83)."
related_concepts: ["differentiable-inverse-design", "physics-grounded-evaluation"]
related_literature: []
tags: ["jax", "inverse-design", "metamaterial", "snap-through", "bistability", "fold-point-sensitivity", "bifurcation", "robust-design", "calculix", "negative-result", "trial-12"]
parent: "metamaterial-inverse-design"
---

# metamaterial-inverse-design-hard

Follow-on to trial 9 (`../2026-11-14-metamaterial-inverse-design/`), attacking the four weaknesses of that pass.
Review page: `docs/qa/trial-12-inverse-design-hard/index.html`. Chronology with every stall, bug and correction:
`log.md`. Pass rules: `PREREGISTRATION.md` (sha256 prefix in `log.md`, not edited after 16:37 UTC).

## Hypothesis

See frontmatter. Written down before any optimisation: the targets, tolerances, verification mesh ladder,
fabrication-error model, the 11 snap-test starts, four arms and the bar "≥ 8 of 11 where the baseline gets ≤ 3".

## Setup

- Config: `config.yaml`. Cell, material and validity limits as in trial 9; design families S9 (9 parameters), S13
  (13) and D26 (two-cell series stack, 26).
- Code: `fem.py` (solver; modes as argument, stack, repaired continuation), `design.py` (targets, grading, nominal and
  robust least squares), `stack.py` (reduced two-cell stack), `snap.py` (fold tools, arms, outer loop, margin),
  `robust.py`, `certificate.py`, `ccx.py` / `verify.py` / `verify_relaxed.py` (CalculiX ladder, 3D, elastic frame),
  `gradcheck_fold.py`, `branch_flip.py`, `robust_decompose.py`, `snap_analyse.py`, `plots.py`, `build_page.py`,
  `page_text.py`. Launchers `run_opt.sh`, `run_snap*.sh`. `scratch/` holds the debugging scripts cited in the log.
- External source (opened this session): L. J. Gibson, MIT OCW 3.054 Lecture 7 "Foams" notes, URL in
  `PREREGISTRATION.md` — E* = E_s ρ², σ*_el = 0.05 E_s ρ², ε_D = 1 − 1.4 ρ.
- No dataset or held-out split; the grade is the independent CalculiX re-solve.

## Result

All numbers: `metrics.json`. Curves: `results/verify_*.json`, `results/relaxed_*.json`, `results/snap/`.

### Targets (graded on the finest ladder mesh that completes)

| target | optimiser loss | CalculiX | tolerance | second criterion | verdict |
|---|---|---|---|---|---|
| H1 foam plateau (Gibson–Ashby), S13 | 0.0798 | **0.0809** (t_min/6) | 0.03 | plateau off 0.031 kPa, limit 0.021 | MISS |
| H2 two-step staircase, D26 | 0.2170 | **0.2172** (t_min/6) | 0.04 | steps off 0.30 / 0.28 kPa, limits 0.028 / 0.056 | MISS |
| H4 expected infeasible, S13 | 0.4672 | no run completes (all stop at 9.25 mm) | 0.05 | energy floor for any elastic cell 0.397 | FAIL (unverifiable) |
| B bistable (trial 9 target), S9, from arm A+O | ≤ 0.07 % | **−0.13 %, +0.01 %, −0.27 %** (t_min/6; t_min/3 and /4 aborted at 4.66 mm) | 5 % each | — | PASS |
| B, margin-arm design (post-hoc) | ≤ 0.78 % | −0.64 %, +0.81 %, −0.62 % (all three meshes complete) | 5 % each | — | PASS |

H1, H2: every start exhausted its 80 evaluations without a converged optimum (`results/opt/*.json`), so these are
"not reached in budget". Ladder spread ≤ 0.12 %, solver gap 0.35–0.8 % of range.

### Robust design (target T2 of trial 9; 64 fresh copies, first 24 in CalculiX)

| design | own error (CalculiX) | pass fraction, design solver (64) | pass fraction, CalculiX (24) |
|---|---|---|---|
| nominal optimum | 0.0101 (trial 9) | 47 % | 58 % |
| robust optimum (SAA, pre-registered start) | 0.0122 | 42 % | 58 % |

Gain −5 / 0 points against the +15 required: no robustness gain. Other starts (not pre-registered): 44 %, 44 %, 2 %,
0 %. One-at-a-time: centre-line error alone 100 % pass, 3 % thickness alone 62 %, thickness modes alone 81 %;
d log σ / d log t = 2.96 (`results/robust_decompose_T2.json`).

### Past the snap

- Diagnosis corrected: trial 9's stalled design sits on a perfect pitchfork; the loading path flips branch under a
  1e-4 parameter change and E_back jumps by 0.54 % (`results/branch_flip.json`). Trial 9's continuation chose
  between the branches by round-off. Continuation repaired (tangent predictor, bisect on indefinite tangent).
- Fold-point sensitivities (`gradcheck_fold.py`): dd_s/dp ≤ 8.6e-6, energy-drop derivative ≤ 2.8e-6, whole pipeline
  ≤ 9.1e-6 relative to central FD in 10 probes; 7 probes straddle the branch flip (no derivative exists).
- Success = all three features within 1 % by trial 9's own extractor, ≤ 60 evaluations:

| arm | 11 pre-registered starts | 6 fresh starts |
|---|---|---|
| B0-legacy (trial 9 exactly) | 0 | 0 |
| B0 (trial 9 gradient, repaired continuation) | 0 | — |
| B0+O (+ outer loop) | 0 | — |
| A (fold-aware gradient) | 0 | — |
| **A+O (pre-registered remedy)** | **1** (4 within 5 %) | — |
| A+O2 (post-hoc: quicker stall exit) | 1 | 1 |
| AM+O2 (post-hoc: eigenvalue margin ρ ≥ 0.15) | 5 (6 within 5 %) | 1 (3 within 5 %) |
| AM+O2, 120 evaluations, on its 6 misses | 0 (3 within 5 %) | — |

  Pre-registered bar (≥ 8 of 11): **not met**.

### Relaxed idealisation (CalculiX, t_min/2.5 in plane; reference = plane strain on the same mesh)

| design | 3D 10 mm deep | elastic frame | NRMSE to target: plane strain → 3D → elastic frame |
|---|---|---|---|
| T2 nominal | shift 4.6 %, forces ×0.90 | shift 11.5 %, ×0.84 | 0.010 → 0.050 → 0.118 |
| T2 robust | 4.7 %, ×0.90 | 10.7 %, ×0.83 | 0.012 → 0.051 → 0.111 |
| H1 | 10.8 %, ×0.91 | 19.8 %, ×0.83 | 0.081 → 0.150 → 0.226 |
| bistable (A+O) | F_pk −15 %, E_back −18 % vs target | F_pk −8 %, E_back −11 % | outside 5 % in both |
| bistable (margin) | F_pk −14 %, E_back −22 % | F_pk −11 %, E_back −13 % | outside 5 % in both |

## Interpretation

The harder targets discriminate: nothing new passed, and the miss is reported identically by optimiser and verifier.
H4 shows the reporting path for an impossible target (multi-start stall + energy certificate + unverifiable in
CalculiX). H1/H2 are budget-limited misses, not impossibility results.

Robustness cannot be bought by reshaping this cell: the response is bending-dominated (σ ∝ t³), thickness scatter
dominates, and the robust optimum coincides with the nominal one.

The gradient's "blindness at a snap" was mis-diagnosed yesterday. A moving fold is differentiable and the limit-point
formula handles it exactly; what stalls the optimiser is a discontinuity of the path itself at a bifurcation, which
the optimiser is attracted to because removing the imperfection raises the peak force. Restarts across the jump and
a margin that keeps the design off the bifurcation help (0 → 1 → 5 of 11) but neither reaches the bar, the margin
variant does not generalise to fresh starts (1 of 6), and more budget does not rescue its misses. The margin has a
second, verifiable benefit: its design runs on every CalculiX mesh, while both bifurcation-sitting designs abort on
the coarse ones.

The shared idealisations were worth more than every tolerance: an elastic one-material frame moves the curves by
11–20 % of range. Trial 9's passing plateau design is a miss with an elastic frame and marginal in 3D.

## Diagnostics

Numbers reference `metrics.json` unless noted; "val" = the independent CalculiX re-solve (no data split).

- intended_effect_confirmed: partial — misses reported as misses (`metrics.json:targets.H1.verdict`, `.H2.verdict`, `.H4.verdict`); robust gain not confirmed (`metrics.json:robust_design.claim_robust_design_helps` = false); snap bar not met (`metrics.json:snap_test.A+O.preregistered.success` = 1 of 11); relaxed checks show designs leave tolerance (`metrics.json:relaxed_idealisation.T2_nominal.ef.passed` = false).
- leakage_check: targets, tolerances, starts and arms fixed in `PREREGISTRATION.md` before optimisation; post-hoc arms scored separately on 6 fresh starts fixed in `snap.py:snap_starts` before they ran — AM+O2's 5 of 11 on the seen starts is contaminated, its 1 of 6 on fresh starts is the clean number (`metrics.json:snap_test.AM+O2`).
- overfitting_signal: train=optimiser NRMSE 0.0798 / 0.2170, val=CalculiX 0.0809 / 0.2172, gap ≤ 0.0011 — no gap; robust design fitted its 8 optimisation copies (rms 0.0395 → 0.0384) and lost 5 points on 64 fresh copies (`metrics.json:robust_design`).
- delta_from_prior: vs metamaterial-inverse-design, bistability success from the same starts 0 of 11 → 1 of 11 (pre-registered) / 5 of 11 (post-hoc margin), attributed to restarts across the branch flip and the eigenvalue margin (`metrics.json:snap_test`); T2 pass fraction under fabrication error unchanged (`metrics.json:robust_design.gain_calculix` = 0).
- unexpected_findings: trial 9's stalled and final bistable designs both sit on a pitchfork where the path flips branch under a 1e-4 change (`results/branch_flip.json`, `log.md` 16:45 entry); an elastic frame shifts curves more than 3D depth does (`metrics.json:relaxed_idealisation`).
- next_candidates:
  - Re-optimise T2 and the bistable cell with the elastic frame and 3D correction inside the design solver (compliant boundary as extra dofs), then re-run the relaxed verification.
  - Replace the soft eigenvalue margin by a hard constraint with a feasibility-restoring line search and test on 20 fresh starts with the bar fixed in advance.
  - Attack fabrication robustness with a stretch-dominated or pre-tensioned member (force ∝ t rather than t³) and the same 64-copy protocol.
  - Give H1 and H2 a 300-evaluation budget with a staged (plateau-first) objective to separate budget from capability.

## Follow-up

- Nothing has been printed; self-contact and rate-dependent rubber remain untested in both solvers.
- The elastic-frame layout is one choice (5 mm posts on the lattice symmetry plane); an edge cell would be softer.
- 24 CalculiX copies per design give ±10 points on a pass fraction.
