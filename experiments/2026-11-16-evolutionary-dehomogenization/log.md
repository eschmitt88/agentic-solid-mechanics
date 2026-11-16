# log — evolutionary-dehomogenization (trial 13)

## 2026-11-16 04:10 UTC — setup
- Source: Xu, Guo, Kawabe, Yaji, Eng. Struct. 371 (2027) 124076; preprint arXiv 2510.08830 →
  `raw/papers/xu2025evolutionary.pdf`. deHomTop808 paper → `raw/papers/woldseth2024dehomtop808.pdf`;
  code snapshot `raw/repos/deHomTop808/` (commit 87894eb, MIT). Woldseth 2024 CMAME phasor paper
  text only (26 MB PDF kept out of git): `scratch/papers/woldseth_phasor.txt`.
- Design draft `DESIGN.md` reviewed by Fable (technical consultant); review adopted except where
  the evaluator studies below override it (resolution, support model, K_σ regularisation).

## 04:20–04:50 UTC — HF pixel evaluator studies (`hf.py`, `scratch/*.py`)
- Buckling code check: Euler clamped column 4 × 100, BLF / Euler = 1.030 (h=1), 1.007 (0.5),
  1.001 (0.25). Correct.
- Finding 1 — support singularity: with fixed nodes under 5 mm pads, the solid girder's BLF falls
  2.005 → 1.397 → 1.245 for h = 0.5 → 0.25 → 0.2 (ratio √2 per halving), mode at the roller pad's
  inner corner. Stiff pads (E=100) and taller pads only shift it. Distributed springs (Winkler)
  bound the traction but the free-edge traction step stays log-singular: still drifts
  (k=1: 2.74 → 2.14 → 1.99 → 1.73). Kept springs (k_line = 1/mm, bearing-pad-like) because they
  remove the √h corner singularity; residual drift handled by finding 2's fix.
- Finding 2 — pixel buckling is not mesh-objective: a 45° cross lattice (w = 0.3, λ = 6) gives
  BLF 0.420 → 0.418 → 0.371 → 0.288 for h = 0.5 → 0.125 while compliance converges (±3 %).
  Mode = single re-entrant pixel corner (duct sliver at h=0.25, bar/pad corner at h=0.125).
  Fix: K_σ from stress averaged over a disc of physical radius r_f over solid pixels only
  (`kg_filter_r`). r_f = 1.0 mm: lattice w=0.3 BLF 0.498/0.486/0.498/0.483, w=0.15
  0.085/0.077/0.083 for h = 0.5…0.125 — converged within ~3–5 %. r_f = 0.5 not enough.
  Deviation from the paper (plain COMSOL linear buckling); the matching body-fitted
  regularisation (corner rounding) is being calibrated in `verify_ccx.py`.
- Cost at 1 thread: static 0.4 s (h=0.25) / 3 s (0.125); buckling (k=1, tol 1e-3) 6–10 s /
  30–50 s. Fable's 0.10–0.125 mm grid (1e4 evals/arm) is unaffordable for buckling → h = 0.25,
  coarse hc = 2.5 mm, λ = 7.5 mm (3 hc, inside the CMAME window), thinnest bar 3 px. The pixel
  bias this leaves is measured by CalculiX calibration (H5 as calibration/ranking test).
- Known-bad check: empty design → connected=False, compliance=inf (rejected). Solid design:
  C = 2.531, BLF 2.75 (h = 0.25, final GIRDER).
- deHomTop808 cross-check under Octave was planned (installed `octave` micromamba env) but
  running the downloaded MATLAB was blocked by the session's permission classifier; the port is
  instead validated against analytic uniform-field tests (see dehom builder report).
