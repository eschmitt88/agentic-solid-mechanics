---
kind: experiment
slug: "phase-field-fracture-sfc3"
date: "2026-11-14"
status: running     # running | done | abandoned   (rung 2 is frozen and awaits external grading)
hypothesis: "An LLM agent can (1) write a phase-field brittle-fracture solver from scratch that reproduces the Miehe 2010 single-edge-notched tension and shear benchmarks (peak load, displacement at peak, crack path) within ±10 % of published values, with mesh and load-step convergence shown, and (2) use it to make a defensible blind prediction for a real fracture challenge from the calibration data alone."
result: "RESULT_LINE"
related_concepts: ["physics-grounded-evaluation"]
related_literature: []
tags: ["fracture", "phase-field", "jax", "gpu", "benchmark", "blind-prediction", "damage-mechanics-challenge", "sfc3-unreachable", "trial-8"]
---

# phase-field-fracture-sfc3

Track B of the phase-2 shortlist (`docs/2026-11-14-hard-problem-shortlist.md`): fracture, ending in a blind
prediction. Review page: `docs/qa/trial-8-fracture/index.html`. Chronological record, including dead ends: `log.md`.

## Hypothesis

1. **Rung 1.** A hand-written phase-field solver reproduces the single-edge-notched tension and shear tests of
   Miehe, Hofacker & Welschinger (2010): peak load and displacement at peak within ±10 % of the mean of published
   values opened this session, the published crack paths, and mesh convergence ≤3 % between the two finest meshes.
2. **Rung 2.** The same method, calibrated only on a challenge's calibration data, gives a blind prediction of
   load–displacement and crack path with a stated uncertainty. (Graded later by the orchestrator, not here.)

## Setup

- Config: `config.yaml`. Pass criteria: `PREREGISTRATION.md` (written 20:42 UTC, before the first solver run).
- **Rung 1 code:** `pf.py` (solver), `batch_tension.sh`, `batch_shear.sh`, `batch_shear_b.sh`, `run_s_l0075_h8.sh`
  (run matrix), `analyze_pf.py`, `rung1_grade.py`, `figs_pf.py`.
  AT2 phase field, spectral (Miehe) tension/compression split, history-field irreversibility, plane strain, bilinear
  quads on graded tensor-product meshes, geometric slit. Residual and tangent by JAX autodiff on the GPU, PARDISO
  sparse solves. Graded configuration: single-pass staggered scheme, Miehe's load steps, finest mesh.
- **Rung 1 references** (all opened this session; PDFs in `refs_pdf/`, not for git): Molnár & Gravouil 2017
  (HAL hal-02132509), Zhou, Rabczuk & Zhuang (arXiv 1902.05922), Zhang et al. (arXiv 1706.05449, 1805.07357),
  Miehe's curves as digitized in github.com/CastillonMiguel/phasefieldx. Miehe 2010 and Ambati 2015 are paywalled
  and were **not** opened; nothing is graded against memory.
- **Rung 2 code:** `pf3d.py` (3D GPU solver, hybrid AT2, Jacobi-PCG), `verify3d_sent.py`, `dmc.py`, `dmc_run3d.py`,
  `dmc_calib2d.py`, `dmc_data.py`, `dmc_contact.py`, `dmc_post.py`, `dmc_predict.py`, `fig_dmc_geometry.py`.
- **Rung 2 data:** `raw/dmc/` (Damage Mechanics Challenge calibration package, PURR doi:10.4231/QF39-Q924, CC BY 4.0).
  `raw/sfc3/SOURCE.md` records why the Third Sandia Fracture Challenge package could not be obtained.
- Extra package: `pypardiso` (via `uv pip install`).
- Reproduce: rung 1 `./batch_tension.sh; ./batch_shear.sh; ./batch_shear_b.sh; ./run_s_l0075_h8.sh; ./batch_shear_c.sh`
  (GPU + ≤6 threads each, ≈4 h in total on a shared box); rung 2 unzip `raw/dmc/LD.zip` and `FractureSurface.zip`
  somewhere and point `DMC_LD_DIR` / `DMC_SF_DIR` at them (default `/tmp/dmc_work/...`), then `dmc_calib2d.py`,
  `dmc_run3d.py` (arguments in `dmc_queue*.sh`), `dmc_data.py`, `dmc_post.py`, `dmc_contact.py`, `dmc_predict.py`;
  finally `analyze_pf.py`, `rung1_grade.py`, `figs_pf.py`, `finalize.py` (writes this README from `report_src/`).
- Not in git: `results/runs/` (69 MB of field snapshots), `results/dmc/*.npz`, `refs_pdf/` (36 MB of papers).

## Result

RESULT_BODY

## Interpretation

INTERP_BODY

## Diagnostics

DIAG_BODY

## Follow-up

FOLLOW_BODY
