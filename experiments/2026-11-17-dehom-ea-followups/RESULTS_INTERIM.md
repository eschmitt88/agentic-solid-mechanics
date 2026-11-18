# Trial 14 — interim results (2026-11-18 03:35 UTC), for the publication-framing consult

Pre-registration: PREREGISTRATION.md. Held-out gain = HV gain on realisation-averaged objectives over
10 held-out phasor seeds (primary). Nominal = archive HV gain on the default realisation used in search.

## Headline blocks (complete)

| block | FULL (paper, VAE) held-out | PCA-BLX held-out | d = PCA − FULL (pp) | nominal FULL / PCA |
|---|---|---|---|---|
| girder stiffness, gen 300 | 19.5 / 16.7 / 19.6 | 22.4 / 15.4 / 17.1 | +2.9 / −1.3 / −2.5 | 30.7, 28.0, 27.6 / 40.7, 36.1, 33.5 |
| girder stiffness, gen 100 (5 seeds) | 16.5 / 12.5 / 16.8 / 15.4 / 15.6 | 17.8 / 12.7 / 15.3 / 14.8 / 20.3 | +1.3 / +0.2 / −1.5 / −0.6 / +4.7 | ~24–29 / ~31–39 |
| girder buckling, gen 100 | 13.3 / 11.3 / 13.8 | 14.5 / 18.8 / 18.2 | +1.2 / +7.5 / +4.4 | 25.3, 24.0, 28.2 / 30.1, 35.4, 27.9 |
| paper's double-clamped beam (umax), gen 100 | 11.6 / 8.2 / 10.6 | 11.5 / 11.4 / 12.1 | −0.1 / +3.2 / +1.5 | 9.9, 9.4, 10.3 / 8.9, 9.2, 9.0 |

- H1 PASS everywhere (FULL > 0 on every seed, every block).
- H2 "no difference" in all four blocks (rule: |d| ≥ 5 pp same sign on every seed). Buckling leans PCA.
- H4 PASS: LOTTERY (gen-0 front + N(0, 0.1²) jitter, no selection, 11 000 evals) held-out +5.0 %
  vs FULL mean 15.4 %, PCA mean 16.2 % at gen 100.
- Winner's-curse index (held-out / nominal): girder 0.38–0.71 (PCA lower, 0.38–0.65; FULL 0.47–0.71):
  about half of the nominal gain is fitting the default phasor realisation. Beam: 0.87–1.34 — on the
  beam the gen-0 front is hurt more by re-realisation than the evolved fronts (unexplained yet).
- Trial 13's nominal-metric conclusion "the VAE hurts (−10 pp)" does not survive held-out grading;
  PCA's nominal lead comes with a lower winner's-curse index.

## Still running (results by ~Wed 14:00 UTC; box constrained 07:30–11:15)
VAE32 / VAE64 / RAWTH × 2 seeds (H3), ROB3 × 2 (H5), problem-grain pilot RAND vs LLM (H9), PCANOV × 2
(H6, first to cut). SIMP multi-start done (best C 6.79 / 5.69 / 4.92 / 4.37 / 3.95 / 3.61 at vf
0.25–0.50; robust to rmin/penal within 5 %). H7 (lattice vs SIMP) not yet graded.

## The arc so far (trials 13 → 14)
1. Trial 13 reproduced the method on a new problem with pre-registered controls; nominal metrics said
   EA works, VAE hurts, mutation inert, SIMP wins, winner's curse present.
2. Trial 14 found our decoder bug (θ/θ+π tie → ~9 % per-design evaluator noise), diagnosed it with a
   2×2 (808 algorithm stable on raw fields; our encoding chaotic), fixed it; found pypardiso refactorising
   every eigsh iteration and a needless ersatz void (10× evaluator speed-up).
3. With a clean evaluator and held-out grading: the EA's gain is real and beats noise-mining, but ~half of
   any nominal gain is realisation-fitting, and the operator ablation that looked decisive nominally is a
   null held-out.
