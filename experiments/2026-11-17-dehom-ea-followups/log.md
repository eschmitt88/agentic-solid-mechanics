# log — dehom-ea-followups (trial 14)

## 2026-11-17 16:10–17:10 UTC — setup and two findings before any run
- Code copied from trial 13 (self-contained); gen-0 caches copied (byte-identical, md5 checked).
- Speed: pypardiso `solve()` without `factorize()` refactorised on every eigsh iteration. hf.py now
  factorises once and calls phase 33: buckling eval 6–13 s → ~1.4 s, identical numbers (1e-13).
- Discontinuity (`scratch/chaos.py`): ε = 1e-12 perturbation of a design's PCA coordinates changes
  pixels in 98 % of designs and compliance by median 8.7 % (43 % of designs > 10 %); same at ε up to
  1e-3. Causes traced: (i) our bug — non-design θ decoded from zero channels (fixed in run_arm.py:
  fixed LF principal-stress angle) — did not remove it; (ii) the orientation clean-up (808
  filterVectorField port) ties when fixed neighbours cancel; (iii) phase alignment amplifies even with
  identical clean-up output. The design → geometry map is effectively stochastic (~9 % noise).
  Trial 13's conclusions stand (same noise in all arms) but explain H6's winner's curse.
- DESIGN.md drafted (A/B/C + publication questions); Fable consulted.

## 17:40 UTC — Fable review adopted; discontinuity diagnosed: OUR decode, not 808
- Fable (technical consultant) recommended: gate on a stage-bisection + 2×2 diagnostic; add LOTTERY
  null, held-out realisation-averaged metrics, 5 paired seeds on the headline, VAE latent sweep and a
  raw-θ (paper) encoding arm, SIMP multi-start, the paper's double-clamped beam; cut ADV and in-loop
  CalculiX; C as a pilot; target a pre-registered critical replication for SMO.
- 2×2 (`scratch/diag2x2.py`, ε = 1e-12): raw LF fields perturbed directly → 0 % pixel change in both
  808-faithful mode (new `faithful=True`: per-layer clean-up, no swap, plain first sweep) and our
  mode; PCA + decode route → 100 % changed in both. The algorithm is stable; our encoding was not.
- Mechanism: `ea.decode` chose θ among 4 quarter-turn candidates by distance to a direction defined
  mod π, so candidates k and k+2 tied exactly and rounding picked θ or θ + π (n → −n); the phasor's
  sign convention then gave a different realisation. Fix: 2 candidates + canonicalise θ to
  (−π/2, π/2]. After the fix: ε = 1e-12 → 0 % change; 1e-6 → 5 % of designs any pixel, no compliance
  change; 1e-3 → median |ΔC|/C 0.3 % (17 % of designs > 10 %: genuine topology changes); 1e-1 → 11 %.
- Consequence for trial 13: its evaluator had ~9 % per-design noise from this bug (all arms equally);
  relative conclusions plausible but the EA partly mined noise (consistent with H6). Trial 14 reruns
  the headline arms with the fixed decoder; trial 13 results are not reused for claims.
