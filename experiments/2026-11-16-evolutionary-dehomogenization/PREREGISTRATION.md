# Pre-registration — trial 13 (evolutionary de-homogenisation, new geometry)

Written 2026-11-16 ~15:20 UTC. At this point the pipeline exists and has been unit-tested
(`hf.py`, `homog.py`, `lf.py`, `dehom.py`, `ea.py`, `run_arm.py`, `run_morelf.py`), the gen-0 LF
designs exist and the gen-0 stiffness evaluation is running. **No evolutionary generation of any
arm has been run.** Nothing below is edited after the first arm starts; misses are reported as
misses. Verdicts: PASS / MISS / FAIL (could not be evaluated).

## Paper claims being tested

Xu, Guo, Kawabe & Yaji (Eng. Struct. 2026, arXiv 2510.08830): starting from homogenisation-based
LF designs, a gradient-free EA (NSGA-II selection, VAE latent crossover, image-deformation mutation)
over PCA-reduced control fields, scored on de-homogenised HF geometry, improves the
(volume, performance) Pareto front: HV +15.1 % (stiffness), +19.4 % (buckling with stiffness
constraint) after 300 generations; mutation raised the stress case's HV gain from +55 % to +107 %;
MFTD buckling designs beat SIMP at similar volume.

## Fixed set-up

New problem (`hf.GIRDER`): 100 × 50 mm simply supported girder, spring bearings under 5 mm pads,
uniform top load on a 2.5 mm deck, Ø12 mm duct with a 2.5 mm liner at quarter span. Coarse LF grid
40 × 20 (hc = 2.5 mm), phasor period 7.5 mm, HF pixel grid 400 × 200 (h = 0.25 mm), K_σ stress
filter r_f = 1.0 mm, bearing springs k = 1/mm. Gen 0 = 480 LF designs (V0 = linspace(0.25, 0.50, 160)
× Lmin ∈ {0.10, 0.15, 0.20}, robust projection, width filter), evaluated after PCA reconstruction
(per-channel PCA, 99 % variance, cap 128 → 506 dims, RMS reconstruction error 11 %). All deviations from the paper are listed in `log.md`,
`INTERFACES.md` and module docstrings.

Objectives (minimised): **S** (stiffness) = (vf, compliance). **B** (buckling) = (vf, −BLF) subject
to compliance ≤ gen-0 compliance envelope at the same vf (constrained domination).
HV: both objectives normalised by the gen-0 feasible non-dominated set (min → 0, max → 1), reference
point r = (1.1, 1.1) (primary) and (2, 2) (secondary), computed on the archive of all evaluated
designs. "HV gain" = HV_final / HV_gen0 − 1 (same normalisation, same r).

Budget per generation: 110 HF evaluations for every EA arm (100 crossover + 10 mutants; NOMUT: 110
crossover). Generations: S arms 100, B arms 60. Elites 100, VAE 506→256→128→8, β = 1, 1000 epochs,
BLX-0.25 latent blend + mean-σ noise, mutation 10 centres, r_m = 10 hc, p0 = 2, |u|max = 1.5 hc.

## Arms

| arm | objective | seeds | what changes |
|---|---|---|---|
| FULL | S, B | 0, 1, 2 | the paper's method |
| NOMUT | S | 0, 1, 2 | no mutation (paper's Case 2) |
| NOVAE | S | 0, 1, 2 | same BLX crossover in whitened PCA space instead of VAE latent |
| NOSEL | S, B | 0, 1, 2 (S); 0 (B) | FULL's operators, uniform parents, random survivors (selection-only null) |
| MORELF | S, B | 1 set | 11 000 (S) / 6 600 (B) extra LF designs with randomised LF settings, no EA (`run_morelf.py`); matched evaluation count |

If wall time forces cuts, the order of sacrifice is: NOVAE seeds 1–2, NOMUT seeds 1–2, NOSEL S
seeds 1–2. FULL S/B seeds 0–2 and the B controls are run first.

## Hypotheses and pass rules

- **H1 replication.** FULL HV gain (r = 1.1) ≥ 10 %, mean over 3 seeds, and positive on every seed.
  Separately for S and B. (Paper: 15.1 %, 19.4 % at 300 generations; we run 100 / 60.)
- **H2a search beats more-LF (two-sided).** FULL final HV vs MORELF HV at the same evaluation count.
  Verdict "EA > more LF" if FULL > MORELF on all 3 seeds; "more LF ≥ EA" if MORELF ≥ FULL on all
  seeds; otherwise "no difference". MORELF primary = raw LF designs, secondary = PCA-projected.
- **H2b selection matters.** FULL > NOSEL on all seeds (S); FULL > NOSEL seed 0 (B).
- **H3 mutation.** FULL − NOMUT mean HV gain ≥ 2 percentage points (S). Expected: null.
- **H4 generative model.** FULL − NOVAE mean HV gain ≥ 2 percentage points (S).
- **H5 buckling vs SIMP.** At the SIMP baseline's vf ∈ {0.30, 0.35, 0.40, 0.45}, the best FULL-B
  (seed 0) archive design within ±0.01 vf has BLF > SIMP's at ≥ 3 of 4 volumes. Reported alongside
  (not graded): its compliance relative to SIMP (expected 10–30 % worse).
- **H6 winner's curse.** The final FULL front (each seed, S and B) and the gen-0 front are
  re-evaluated with 5 random phasor phase seeds per design; HV is recomputed from the
  phase-averaged objectives. PASS if the phase-averaged HV gain retains ≥ 50 % of the nominal
  gain (mean over seeds). Motivation: one design's compliance varies ±15 % across phase seeds.
- **H7 evaluator calibration (CalculiX, body-fitted).** On ≥ 24 designs (gen-0 and final fronts,
  S and B, spanning vf): Spearman ρ(pixel, CalculiX) ≥ 0.9 for compliance and for BLF; CV of the
  pixel/CalculiX ratio ≤ 10 % for compliance and ≤ 20 % for BLF; and for each objective the FULL
  front dominates the gen-0 front under CalculiX at ≥ 3 of 4 matched volumes iff it does under
  the pixel evaluator. CalculiX settings (corner-rounding radius, mesh size) are those recommended
  by the verifier builder before any front is graded.

## Diagnostics logged every generation (not graded)

HV (both r), population diversity (mean pairwise distance in whitened PCA space), fraction of
children connected, fraction of mutants among survivors, share of a mutation surviving PCA
re-projection, VAE loss and mean σ, and for H6/H7 the HF/LF compliance ratio of survivors.

## Evaluator integrity (HCE clause 4)

The in-loop evaluator is `hf.py` + `dehom.py` + `ea.hv2d`/`ea.Normaliser`; their sha256 are
recorded in `log.md` before the first arm. CalculiX (`verify_ccx.py`) is never called in the loop.
Known-bad check already passed: an empty design is rejected (connected = False, C = ∞).
