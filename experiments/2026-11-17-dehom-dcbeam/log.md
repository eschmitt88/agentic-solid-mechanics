# log — dehom-dcbeam (second problem for trial 14)

## 2026-11-17 17:05–18:25 UTC — setup, checks, gen 0 (subagent for the trial-14 orchestrator)
- Code copied from trial 14 (all *.py) + results/homog_coeffs.json (md5 79dd029a…, identical).
  Problem from one config `hf.PROBLEM = hf.DCBEAM` (choices in hf.py docstring). Adapted:
  hf.py (box masks, `support="fixed"`, load on the load block's top edge; buckling refuses
  fixed supports), lf.py (exact fixed-dof elimination in the banded solve), dehom.py (grids,
  box signed distances), simp.py (free-dof solve, no buckling), run_arm.py (`--obj umax`).
  verify_ccx / grade_ccx / make_figures / analyze copied unchanged (girder-specific, marked).
- Checks (scratch/checks.json): solid domain C: LF coarse 6.5415, Q4 coarse 6.5415, HF fine
  6.6030 → LF −0.93 % vs HF. Timoshenko clamped-clamped, central point load, L = 150, d = 40:
  bending 3.296 + shear 2.925 = 6.221; FE mid-span axis 5.782 (0.93×), umax (top, under load)
  6.740 (1.08×, includes load-patch indentation).
- Gen-0 LF sweep 480 designs: 744 s on 4 workers (6.2 s/design), vol error ≤ 1.5e-4.
  HF/LF compliance ratio median 1.39 (5–95 %: 1.11–1.94). 100 % connected.
- SIMP (6 vf, 600 × 160): 935 s; all stop at 300 it (change 0.07–0.2, not < 0.01). Binary C
  23.17 / 19.64 / 17.32 / 15.49 / 13.25 / 11.92; gen-0 best at the same vf (±0.01) is 10–18 %
  more compliant (26.86 / 21.90 / 19.06 / 18.22 / 15.63 / 14.09).
- Gen-0 caches: eval14_stiff 354 s, eval14_umax 344 s (4 workers, 2.9 s/eval; identical
  records). PCA nc [64, 44, 63, 128, 128] (427 dims), recon err 0.105.
- Continuity (scratch/chaos.py, 40 designs): ε = 1e-12 → 0 % pixels, 0 % ΔC; 1e-6 → 2.5 %
  designs any pixel, max |ΔC|/C 1e-9; 1e-3 → median 1.1 %, 7.5 % of designs > 10 %.
- umax caveat: umax/C median 1.03 but up to 2.3: the max sits on dangling solid stubs (a
  bar attached at one end, rotating rigidly), not near the load (scratch/umax_outlier.png).
  umax rewards removing such stubs; compliance does not see them.

## 18:09 UTC — orchestrator: integrated, campaign queued
- heldout.py copied from trial 14 and extended to umax (geometric mean over 10 held-out seeds).
- Arms: FULL and PCA x s0-s2 x 100 generations on (vf, umax), PREREGISTRATION in trial 14 (H8).
- sha256 hf.py: 0aa476c251da4d43
- sha256 dehom.py: 1af70aa45bab8f78
- sha256 ea.py: 3d67b954cce9a25b
- sha256 run_arm.py: 8f49deddb2d1b23c
- sha256 heldout.py: b5ecfe03c67bbdce
- sha256 lf.py: 53203a2fe8d8cd85
