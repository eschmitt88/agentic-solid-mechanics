---
title: "What does evolutionary de-homogenisation actually gain? A pre-registered replication with held-out realisations"
status: outline (2026-11-18); numbers final except buckling seeds 3–4
target: Structural and Multidisciplinary Optimization (research paper); arXiv preprint first
replicates: Xu, Guo, Kawabe & Yaji, Eng. Struct. 371 (2027) 124076, doi 10.1016/j.engstruct.2026.124076 (arXiv 2510.08830)
evidence: experiments/2026-11-16-evolutionary-dehomogenization (trial 13), experiments/2026-11-17-dehom-ea-followups (trial 14), experiments/2026-11-17-dehom-dcbeam (trial 14b)
---

# Outline

## Claim (one sentence)

The hypervolume gain that evolutionary de-homogenisation reports over its initial population comes
mostly from three effects that are not design improvement — fitting the one phasor realisation the
search sees, comparing against a baseline selected on that realisation, and extending the front to
volumes the initial population never sampled (which running the low-fidelity optimiser at lower targets
reproduces) — and once these are controlled the gain at matched volume is a few percent and the
paper's generative-model operator is indistinguishable from linear recombination.

## 1. Introduction
- Evolutionary de-homogenisation (Xu et al. 2026; MFTD lineage: Yaji 2020/2022, Kii 2024, Kawabe 2025):
  LF homogenisation TO → phasor de-homogenisation (Woldseth 2024) → gradient-free EA scored on HF geometry.
- Hypervolume gain over the initial population is the field's headline metric. What does it measure?
- Contributions: (i) pre-registered replication on the paper's own beam and a new girder, compliance and
  buckling; (ii) an evaluation protocol (held-out realisations, full-population baseline, in-range
  decomposition, lottery and extension controls, continuity test); (iii) the decomposition; (iv) operator
  ablations under that protocol.

## 2. Method reproduced (with Table 1: deviations from the paper)
- Problems: paper's double-clamped beam (vf, max displacement); girder 100×50 with spring bearings, deck
  load, lined duct (compliance; linear buckling with stiffness constraint).
- LF: cross-cell homogenisation (spline fit — the paper's polynomial cannot fit the jump at full width),
  MMA, robust projection; 480 designs over V0 × Lmin as in the paper.
- De-homogenisation: phasor port of deHomTop808 (branch closure not ported — limitation).
- HF: pixel Q4, solid-only system; buckling with stress-averaged geometric stiffness (pixel buckling is
  otherwise not mesh-objective); spring bearings (fixed supports give a corner mode with BLF → 0).
  CalculiX body-fitted verification: ρ = 0.991 / 0.992 (30 designs).
- EA: PCA (506 dims), NSGA-II, VAE latent crossover (paper) or linear BLX in PCA space, image mutation.

## 3. Evaluation protocol
- Realisations: the phasor's global phase (and alignment order) is non-physical; held-out = 10 seeds
  never used in search; translation-only variant reported. Geometric means over realisations.
- Baseline: held-out front of all 480 initial designs (not their nominal front).
- Decomposition: nominal → held-out → fair baseline → in-range HV (clipped at the baseline's minimum vf);
  matched-volume attainment ratios.
- Controls: LOTTERY (jittered gen-0 front, no selection, matched budget); extension (LF at V0 0.10–0.25).
- Continuity test: ε-perturbation of design coordinates (catches representation bugs; Supplement S2).
- Pre-registration and addenda; paired-seed statistics (t-intervals, exact sign-flip).

## 4. Results
1. Replication (H1): the EA improves held-out HV on every seed/problem (girder S +15–22 %, B +11–19 %,
   beam +8–12 % against the nominal-front baseline).
2. Decomposition (Fig. 3): girder S nominal 29–37 % → in-range 0.5–0.6 %; B 26–31 % → 0.3–1.3 %;
   beam 9–10 % → 3.0–3.2 %. Matched-volume compliance in range: 2–9 % lower (girder S), BLF ≤ 2 %,
   beam umax 1–2 %. Extension: LF at lower V0 matches the EA's low-volume front (median ratio ~1.1 S,
   ~1.0 beam). LOTTERY recovers ~1/3 of the held-out HV gain (5.0 vs ~15 %).
3. Winner's curse (Fig. 5): median per-design optimism of the default realisation: gen-0 front ×1.08,
   evolved ×1.28–1.41 (S); ×1.40 vs ×1.68–1.92 (B); tails to ×5–37. Same under translation-only.
4. Operators (Fig. 4): VAE vs linear crossover: nominal says PCA +5–10 pp; held-out d ≈ 0 in every block
   (S gen 100, n = 5: matched-vf compliance d = −0.6 % [−1.3, +0.2]); latent 8/32/64 and raw-θ encoding
   indistinguishable; image mutation inert (≤ 2 % of survivors; trial 13); robust selection (3 draws)
   no gain at matched evaluations.
5. SIMP reference: multi-start SIMP is stiffer at every volume and higher BLF at 3/4 volumes (girder).

## 5. Threats to validity
Not the authors' pipeline (pixel HF in loop, our encoding, no branch closure, 100 generations for B and
beam); one evaluator artefact class (one-pixel necks) that the search finds; seeds 3–5 per block;
stress objective untested; phase as the realisation variable (translation-only control).

## 6. Recommendations for the field
Report held-out-realisation metrics; use the full initial population as the baseline; report in-range HV
or matched-volume attainment; sample the initial population over the volume range the front will cover;
run a lottery control; ε-continuity test for any new encoding; filter one-pixel necks in pixel evaluators.

## Figures and tables
- Fig 1 pipeline + where the three confounds enter. Fig 2 nominal vs held-out fronts (one run per problem,
  gen-0 vf range shaded). **Fig 3 decomposition (draft: figs/fig3_decomposition.png).** Fig 4 paired d with
  CIs, nominal beside held-out. Fig 5 per-design realisation spread + gallery of fragile designs.
  Fig 6 SIMP vs lattice at matched volume.
- Table 1 deviations. Table 2 hypotheses, pre-registered rule, verdict.
- Supplement: pre-registration + addenda; the decoder bug (θ/θ+π tie) and 2×2 diagnosis; evaluator
  verification (Euler column, Timoshenko, CalculiX); VAE sweep; SIMP multi-start; per-seed tables; code.

## Before submission (decisions for the human author)
- Contact the authors with the draft (critical replication, code not public): ask about phase handling,
  vf range of their initial and final fronts, latent size, number of runs. Email from the human author.
- AI-use declaration per the journal; short methods paragraph on the agentic workflow; human author
  responsible.
- Optional strengthening: beam FULL at 300 generations (direct comparison with their 15.1 %); branch
  closure; more seeds.
