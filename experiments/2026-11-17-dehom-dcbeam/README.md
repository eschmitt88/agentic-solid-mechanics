---
kind: experiment
slug: "dehom-dcbeam"
date: "2026-11-17"
status: done     # running | done | abandoned
hypothesis: "Second test problem for trial 14 (2026-11-17-dehom-ea-followups): the same evolutionary arms on Xu et al. 2026 Sec. 4.1 double-clamped beam, objective umax (and compliance)."
result: "The paper's double-clamped beam (vf, umax), our pipeline: FULL improves held-out HV on every seed (100 gen: 8.2–11.6 %; 300 gen: 8.2–10.4 %); nominal gain at 300 generations 10.2–10.6 % vs the paper's 15.1 % (normalisation differs); against all 480 initial designs 3.5–4.4 %, in range 2.3–3.0 %, umax at matched volume 1–2 % lower. VAE vs linear PCA crossover: no difference (d = +1.5 pp [−2.6, 5.6]). The beam is barely realisation-sensitive (median optimism ×1.05). Graded in trial 14 (H1, H2, H8)."
related_concepts: []
related_literature: []
tags: [dehomogenisation, evolutionary, reproduction, xu2026]
# members: only set when kind: ensemble — list parent experiment slugs.
# parent:  only set when this experiment was produced via /expand.
---

# dehom-dcbeam

## Hypothesis

Not a separate pre-registration: this folder is trial 14's second problem.
Hypotheses and arms are trial 14's (`../2026-11-17-dehom-ea-followups/DESIGN.md`);
this experiment only provides the problem, its gen-0 population, SIMP
baselines and gen-0 evaluation caches.

## Setup

- Problem: Xu et al. 2026 Sec. 4.1 / Fig. 8a double-clamped beam, 150 x 40 mm.
  Definition and every choice the paper leaves open: `hf.py` module docstring
  (`hf.DCBEAM`, `hf.PROBLEM`). Stiffness only; no buckling (fixed supports).
- Config: `config.yaml`
- Code: self-contained copy of trial 14. Adapted: `hf.py` (problem config,
  `support="fixed"`, block load), `lf.py` (exact fixed-dof elimination in the
  banded solve), `dehom.py` (grids and signed distances from `hf.PROBLEM`),
  `simp.py` (free-dof solve, no buckling), `run_arm.py` (`--obj umax`).
  Copied unchanged and still girder-specific: `verify_ccx.py`, `grade_ccx.py`,
  `make_figures.py`, `analyze.py`; `rereal.py`, `run_morelf.py` follow
  `hf.PROBLEM` but only know `stiff|buck`.
- `results/homog_coeffs.json`: byte copy of trial 14's (same cell).
- Gen 0: `python -u lf.py gen0` (480 LF designs, `LF_WORKERS`), `python simp.py`
  (`SIMP_WORKERS`), `python -u run_arm.py --obj stiff|umax --gen0-only`.
- Checks: `scratch/checks.py` (solid LF vs HF, Timoshenko), `scratch/look.py`
  (`scratch/look.png`), `scratch/chaos.py` (continuity).

## Result

Fill in after the run. Point at `metrics.json` (validation split — this
is the search signal and the file every other skill reads). A separate
`final_metrics.json` holds held-out test-split numbers and is written
only by the `dvc repro final_eval` pass at chain end. See
`~/.claude/rules/evaluation.md`.

## Interpretation

What did you actually learn? What surprised you?

## Diagnostics

Fill in after the run. One line per field; leave `n/a` rather than
blank. `next_candidates` must list ≥2 concrete one-sentence proposals.
Every concrete claim below needs a **citation anchor** — a code
reference like `train.py:42-58`, a metrics file path like
`metrics.json:val_acc`, or a wikilink into `literature/`. Unanchored
assertions are flagged by `/lint` (Kosmos, arXiv 2511.02824).

Unless otherwise noted, metric numbers here reference `metrics.json`
(validation split). Cite `final_metrics.json` only if this experiment
is itself the final-scoring pass.

- intended_effect_confirmed: <yes | no | partial> — <one-line evidence with anchor>
- leakage_check: <method used> — <finding>
- overfitting_signal: train=<x> val=<y> gap=<z> — <interpretation> (from metrics.json)
- delta_from_prior: vs <related_prior_slug>, <metric_delta> attributed to <cause> (metrics.json)
- unexpected_findings: <one or two sentences, or "none">
- next_candidates:
  - <one-sentence proposal 1>
  - <one-sentence proposal 2>

## Follow-up

- ...


## Result (folded in 2026-11-18 15:10 UTC; grading in trial 14)

| run | gen | nominal | held-out vs gen-0 front | vs all 480 | in range | matched-vf umax |
|---|---|---|---|---|---|---|
| FULL s0–s2 | 100 | 9.4–10.3 % | 8.2–11.6 % | 3.4–5.1 % | 2.0–3.6 % | 1–2 % lower |
| PCA s0–s2 | 100 | 8.9–9.2 % | 11.4–12.1 % | 5.1–6.0 % | 2.8–4.0 % | 1.6–2.3 % lower |
| FULL s10–s12 | 300 | 10.2–10.6 % | 8.2–10.4 % | 3.5–4.4 % | 2.3–3.0 % | 1.1–1.9 % lower |

Paper (Xu et al. 2026, Sec. 4.1.2): +15.1 % HV after 300 iterations; their HV normalisation/reference is not
stated, so the comparison is indicative. Source: `../2026-11-17-dehom-ea-followups/results/decomposition.json`.
