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

## 17:47 UTC — pre-registration frozen; evaluator hashes
- Solid-only evaluator: 2.3x faster; gen-0 caches rebuilt (S 22 evals/s, B 4.8 evals/s box-wide).
- Box is memory-bandwidth bound (Ryzen 9950X, dual-channel): throughput flat from 4 to 24 workers (scratch/scaling.py).
- sha256 hf.py: c74a5d2c24e535b1
- sha256 dehom.py: d6ffd1ca10d4c24e
- sha256 ea.py: 3d67b954cce9a25b
- sha256 run_arm.py: fd657705e92bff92
- sha256 heldout.py: a87d9d1a6aca0bf1
- sha256 run_pg.py: cbe24dbc17d705e2
- sha256 pg_ops.py: d9a0ceb290c15590
- sha256 lf_passive.py: 19fe973f60b05dab

## 19:10 UTC — first held-out numbers (interim, not grades)
- Stiffness: FULL s0/s1 gen 100: held-out 16.5 / 12.5 % vs nominal 28.7 / 23.9 %; PCA s0 gen 100 17.8 vs
  38.8 %, gen 300 22.4 vs 40.7 %. About half of every nominal gain is realisation-specific (the
  search fits the default phasor realisation). PCA's nominal lead over FULL (~10 pp) shrinks to ~1 pp
  held-out at gen 100 — the trial-13 "VAE hurts" result may be mostly PCA mining realisation luck.
- heldout_daemon.sh grades finished arms (both trials) automatically.

## 03:50 UTC (11-18) — headline blocks graded; beam "ratio > 1" explained
- H1 PASS (all blocks), H2 "no difference" (all four blocks; B leans PCA +1.2/+7.5/+4.4), H4 PASS
  (LOTTERY +5.0 % vs ~15–16 %). Table: RESULTS_INTERIM.md, results/heldout_summary.json.
- Per-design optimism of nominal over realisation-averaged objectives (median): girder S gen-0 front
  ×1.08, FULL ×1.28, PCA ×1.41 (16 % of PCA front designs disconnect under some held-out seed);
  girder B ×1.40 / 1.68 / 1.92; beam umax ×1.05 / 1.10 / 1.05. The beam's lattices barely depend on
  the phasor phase, so nominal ≈ held-out there (ratio 0.87–1.34 is HV noise around 1); the girder is
  realisation-sensitive and search amplifies it, PCA most (explains PCA's nominal lead).
- Fable consulted on the publication framing.

## 05:40 UTC — Addendum B decomposition (results/decomposition.json)
- Fair baseline (all 480 gen-0 designs held-out): girder S held-out gain 11–21 % (vs 12–22 % against the
  gen-0 front — small effect for S), girder B 4.8–8.1 % (vs 11–19 %: the gen-0 front baseline was
  strongly winner's-cursed for buckling), beam 3.4–6.0 % (vs 8–12 %).
- In-range (HV clipped at the baseline's min vf): S 0.2–0.8 %, B 0.0–1.8 %, beam 2.0–4.0 %. The rest is
  front extension below gen 0's sampled volumes (gen 0 starts at V0 = 0.25; runs reach vf 0.10–0.24).
- Matched-volume attainment, in range (geometric mean ratio, < 1 = run better): S 0.91–0.98
  (compliance 2–9 % lower), B 0.98–1.00, beam 0.98–0.99. LOTTERY S 0.977.
- Extension control (240 LF designs at V0 0.10–0.25, held-out): the EA's low-vf front vs LF-at-low-V0,
  matched vf: ratio 0.65–1.64, median ~1.1 (S) and 1.0–1.3 (B) — running the LF optimiser at lower
  targets gets as far as the EA in most runs. Beam extension control running.
- Operators (RAWTH, VAE32, VAE64, PCA, FULL) remain indistinguishable on every decomposed metric.

## 05:50 UTC — paired statistics (results/paired_stats.json); translation-only mode
- d = PCA − FULL, 95 % t-CI, exact sign-flip p: no block/metric differs (all p ≥ 0.19). Tightest:
  S gen 100 (n = 5) in-range matched-vf compliance ratio d = −0.006 [−0.013, +0.002] — equivalence within
  ~1.3 %. Held-out HV vs all-480: S gen 300 −0.2 pp [−7.4, +6.9]; B (n = 3) +1.5 pp [−2.4, +5.4];
  beam +1.2 pp [−1.1, +3.5]. B s3–s4 running → n = 5.
- dehom.py: DEHOM_PHASE_MODE=global (translation-only realisations: fixed raster tie order, random
  per-layer global phase); default path unchanged. One gen-0 design: compliance 4.63 (default) →
  4.75 / 5.64 / 6.47 under three translations alone. translation.py running.
