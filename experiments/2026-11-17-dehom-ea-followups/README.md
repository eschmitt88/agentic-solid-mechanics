---
kind: experiment
slug: "dehom-ea-followups"
date: "2026-11-17"
status: done
hypothesis: "Follow-ups to trial 13 under a clean evaluator and held-out-realisation grading (PREREGISTRATION.md H1–H9 + Addenda A, B): does the evolutionary de-homogenisation gain (Xu et al. 2026) survive held-out phasor realisations, a fair baseline and front-extension controls; does the paper's VAE crossover beat linear PCA crossover; do latent size, encoding, robust selection or a lottery null change the picture; and how do the lattices compare with SIMP — on the girder (compliance, buckling) and the paper's own double-clamped beam."
result: "The EA improves held-out hypervolume on every seed and problem (H1 pass), but most of the reported-style gain is not design improvement: nominal gains of 27–37 % (girder) fall to 0.5–1.0 % in-range once held-out realisations, a full-population baseline and the volume range are controlled; at matched volume the EA lowers compliance 2–9 %, raises BLF ≤ 2 %, beam displacement 1–2 %, and its low-volume extension is matched by running the LF optimiser at lower targets. The paper's VAE and linear PCA crossover are indistinguishable held-out on every block (H2), though PCA leads nominally by 5–9 pp; latent size and raw-θ encoding don't matter (H3); robust selection gives nothing (H5); SIMP beats the best lattice at every graded volume in compliance and BLF (H7). Before any run, trial 13's evaluator noise was traced to our decoder (θ/θ+π tie), fixed; evaluator 10× faster."
related_concepts: ["physics-grounded-evaluation", "agentic-design-optimization"]
related_literature: ["xu2025evolutionary", "woldseth2024dehomtop808"]
tags: ["topology-optimization", "dehomogenization", "replication", "held-out", "winner's-curse", "hypervolume", "trial-14"]
---

# dehom-ea-followups (trial 14)

Pre-registration `PREREGISTRATION.md` (frozen before any arm; Addendum A budget, Addendum B post-hoc and
declared). Chronology and every decision: `log.md`. Interim brief: `RESULTS_INTERIM.md`. Paper skeleton:
`docs/paper/outline.md` + `docs/paper/figs/`. Companion: `experiments/2026-11-17-dehom-dcbeam/` (the paper's beam).

## Setup changes vs trial 13
- Decoder fix (θ vs θ + π tie → realisation chosen by rounding; ~9 % per-design evaluator noise; found by a
  2×2: deHomTop808's algorithm is stable on raw fields, our encoding was not).
- Evaluator: factorise once (pypardiso refactorised every eigsh iteration) + solid-only system (no ersatz
  void): 10× faster; CalculiX spot-check ρ = 0.991 / 0.992 on 30 trial-14 front designs.
- Primary metric: held-out HV over 10 phasor seeds never used in search.

## Result (results/heldout_summary.json, decomposition.json, paired_stats.json, paper_tables.json)

| | verdict | evidence |
|---|---|---|
| H1 EA gain > 0 held-out | PASS | girder S +16.7–19.6 % (gen 300), B +11.3–18.8 %, beam +8.2–12.1 % |
| H2 VAE vs PCA | no difference (4/4 blocks) | S gen 300 d = −0.3 pp [−7.4, 6.9]; S gen 100 (n = 5) +0.8 [−2.2, 3.7]; B (n = 5) +1.3 [−4.7, 7.3]; beam +1.5 [−2.6, 5.6] |
| H3 latent / encoding | "artefact" (rule) | VAE32 18.1, VAE64 13.5, RAWTH 17.8 vs PCA 15.3 % (gen 100, s0–s1) |
| H4 search > lottery | PASS (primary) | LOTTERY 5.0 % vs ~15–16 %; in range LOTTERY 0.3 % vs search 0.4–0.7 % (Addendum B) |
| H5 robust selection | no difference | ROB3 12.8 / 15.5 % vs PCA 22.4 / 15.4 % at 33 000 evals |
| H6 novelty, H9 LLM pilot | cut (Addendum B) | — |
| H7 lattice vs SIMP | SIMP wins | compliance 4/4, BLF 4/4 at vf 0.30–0.45 |
| H8 beam | H1 PASS, H2 no difference | see H1/H2 |

Addendum B decomposition (girder S / B / beam, mean of arms): nominal 29–37 / 27–29 / 9–10 % → held-out
18 / 14–16 / 10–12 % → fair baseline 17 / 5–6 / 4–5 % → in range 0.5–0.6 / 0.6–1.0 / 3.0–3.2 %.
Matched-volume attainment in range: S 0.91–0.98, B 0.98–1.00, beam 0.98–0.99. Extension vs LF at
V0 0.10–0.25: median ratio ~1.1 (S), ~1.1 (B), 0.95–0.99 (beam). Winner's curse (median optimism of the
reported front): S gen-0 ×1.08, FULL ×1.28, PCA ×1.41; B ×1.40 / 1.68 / 1.92; beam ≈ ×1.05–1.10; the
same under translation-only realisations.

## Interpretation
- The search is real (beats a lottery and a no-selection null) but small at matched volume; the
  headline-style HV gain is mostly (i) fitting the single phasor realisation the search sees, (ii) a
  baseline selected on that realisation, (iii) extension below the initial population's volume range.
- Operator conclusions drawn from nominal HV can be wrong: trial 13's "VAE hurts" was PCA mining
  realisation luck more effectively (higher optimism, 16 % of its front disconnects under re-draws).
- The beam is barely realisation-sensitive (optimism ~×1.05), the girder strongly — the size of the
  winner's curse is problem-dependent.

## Diagnostics
- intended_effect_confirmed: partial — H1 yes; the paper's operator claims not supported (`results/paired_stats.json`)
- leakage_check: held-out seeds 5000–5009 never used in search (`heldout.py:HELDOUT_SEEDS`, `run_arm.py` uses phase_seed=None or random ROB3 seeds); evaluator hashes logged at freeze (`log.md` 18:30)
- overfitting_signal: held-out/nominal ratio 0.38–0.71 on the girder (`results/heldout_summary.json`); per-design optimism p90 up to ×37 (`results/paper_tables.json:fig5_optimism`)
- delta_from_prior: vs trial 13, "VAE hurts −10 pp" reversed to no difference once graded held-out (`results/paired_stats.json`); evaluator noise from the decoder bug removed (`scratch/diag2x2.json`)
- unexpected_findings: the initial population's own nominal front is winner's-cursed (buckling ×1.40); one-pixel-neck artefacts recur in evolved buckling fronts (`results/ccx_spot.json`)
- next_candidates:
  - Beam FULL at 300 generations (queued, `../2026-11-17-dehom-dcbeam/campaign300.sh`) for the one number comparable to the paper's 15.1 %.
  - Port branch closure (deHomTop808) and measure how much of the in-range gain it absorbs.
  - Robust selection with more draws (K = 10) or elite-only resampling, matched evaluations.
