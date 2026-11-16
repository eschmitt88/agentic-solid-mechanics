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

## 05:20 UTC — dehom.py (builder report, orchestrator checks)
- Port of Xu Eqs. 8–11 from the deHomTop808 source (read, not executed). 0.11 s/call, 1 core.
  Uniform fields: spacing 7.500 mm, direction exact, thickness within ~1 %, volume vs cross formula
  within 1 % except one quantisation case. Branch closure/boundary layer not ported (not in Xu).
  Deviations listed in the module docstring (flood-ordered first alignment sweep, w1/w2 swap at
  layer jumps, direct cubic interpolation to the fine grid).
- Orchestrator fix: default global phase 0 put axis-aligned w = 0.2 bar edges on pixel centres
  (−15 % thickness); set a deterministic 0.3 rad offset → (0.2, 0.2) at 0°/90° now 0.360 = formula.
  Residual pixel quantisation: a 4.5-px bar renders as 4 px (w = 0.15 cross: 0.249 vs 0.2775).
  Inherent to h = 0.25 mm; covered by the CalculiX calibration (it meshes the continuous `level`).
- Builder finding to carry into the pre-registration: re-realising one design with 4 phase seeds
  moves its compliance 10.35–14.19 (±15 %). Winner's-curse control needed.

## 05:40 UTC — LF half (builder report) + orchestrator decisions
- homog.py: periodic 200² cross cell; C^H jumps at w = 1 (a slit cuts the other bar), so the paper's
  Eq. 4 polynomial cannot fit (degree 5: 60–1745 % error). Used a tensor-product cubic spline on
  [0, 0.95]² + linear ramp to C_iso over [0.95, 1]: off-grid error ≤ 0.65 %, exact solid at w = 1,
  PD after projection. Deviation from the paper (Eq. 4).
- lf.py: paper Eq. 3 as printed is the stress transform; the engineering-shear strain transform of
  deHomTop808 is used (rotated single bar matches analytic to 2.4e-6). Own Svanberg MMA. Solid LF
  compliance 2.426 vs HF 2.531 (−4.2 %, coarse mesh + snapped duct).
- Decision: gen 0 uses deHomTop808's robust projection (three-field left ρ̄ 49–68 % grey, so Lmin
  was not a minimum width) and a width filter rmin_mu = 1.5 (checkerboard widths). Regenerated:
  `results/gen0/lf_designs.npz`, 242 s on 12 processes. Older sets kept as *_threefield / *_robust.
- simp.py: 88-line-style SIMP on the fine grid, 6 volumes; HF numbers in results/simp/simp_eval.json.
- Gen-0 look (scratch/ea/gen0_look.png): tied A-frames; HF/LF compliance ratio ≈ 1.1–1.25 at
  V0 = 0.5 but 3.7–11 at V0 = 0.25 (thin regions fragment without branch closure).

## 15:00 UTC — container restart
- Restart killed the gen-0 evaluation and the CalculiX builder (resumed from its transcript).
  ~9.5 h lost. Gen-0 stiffness evaluation restarted.
