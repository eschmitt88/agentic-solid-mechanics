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
