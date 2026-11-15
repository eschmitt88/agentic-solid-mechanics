# Pre-registration — written before any model was fitted to any measured data

Written 2026-11-15 16:39 UTC. At this point: the dataset paper and test matrix had been read, the
nominal outlines of all geometries plotted, the uniaxial dog-bone files inspected for size/peak force,
and **one calibration file (O1-3)** opened to learn the file layout and plot its force–extension curve.
`fem.py` existed and had passed its unit checks (uniaxial closed form, finite-difference check of
sensitivities) on made-up parameters. No held-out file had been opened (they were downloaded and
md5-checked only), and no parameter had been fitted to anything.
Nothing below may be edited after the first fit; misses are reported as misses.

## Data

Jones et al., "Digital image correlation and infrared thermography data for seven unique geometries of
304L stainless steel", Scientific Data 11:1101 (2024), doi:10.1038/s41597-024-03949-y; files from
Figshare+ doi:10.25452/figshare.plus.25483534 (CC0). 1.5 mm 304L sheet, stereo-DIC (VIC-3D), load cell.

Scope restriction (decided from the test matrix, before looking at fields): only the **slow** tests
(5·10⁻³ mm/s, about 1.5 h each, measured temperature rise ≈ 0 in O1-3) are used, so a rate-independent,
isothermal model is defensible. The fast tests (self-heating, rate effects) and the D-specimen (fast
only) are out of scope. All slow full-field tests but one are in the transverse orientation.

## Split (fixed)

| role | specimens | used for |
|---|---|---|
| calibration (full-field) | **O1-3, O14-3** (bespoke plates 1 and 14, transverse, slow) | FEMU fit |
| development check | **O5-4** (bespoke 5, transverse, slow) | model-form selection, forward-model debugging, metric dry-run |
| calibration (uniaxial baseline) | dog bones **BT3, CT3, CT4** (transverse, slow) | baseline fit only; not used by the full-field fit |
| **held-out, primary** | **O15-3** (bespoke 15: slot hole + one notch), **XT10** (X-specimen, transverse) | final scoring only |
| held-out, orientation transfer | **XR5** (X-specimen, rolling direction) | final scoring only, reported separately |
| not downloaded | O18-3 (15 GB), all fast tests | — |

Held-out files are opened exactly once, by `score_heldout.py`, after the models are frozen
(parameters written to `results/frozen_models.json`). That script was developed and dry-run on O5-4.
Rolling/diagonal dog bones are not used for fitting.

## Forward model (fixed choices)

- Plane stress, finite strain (logarithmic-strain plasticity), T6 triangles, nominal thickness 1.50 mm,
  nominal outline from the `*NominalGeometry.mat` files.
- Elastic constants fixed: E = 193 GPa, ν = 0.29 (typical handbook 304L values, **from memory, not
  graded**; the tests are at 10–40 % strain where elasticity barely matters — sensitivity reported).
- Boundary conditions (standard FEMU practice, same rule for every specimen including held-out ones):
  the model is cut at y_lo = min(Y0)+3 mm and y_hi = max(Y0)−3 mm of the DIC point cloud; on each cut
  line the measured DIC displacements (U, V) of points within ±0.75 mm are fitted with a cubic in x at
  every station and imposed on the mesh nodes of that line. So the held-out *boundary kinematics* are
  an input; the held-out **force** and **interior strain field** are the predictions that are scored.
- Stations: x_F = LVDT extension at maximum force. Station j (j = 0…40) is the frame whose extension
  is nearest to j/40 · 0.9 · x_F. The model has no damage/fracture, so nothing beyond 0.9 x_F and no
  peak force is predicted or graded.
- Force-drop rule (the paper says slow tests contain spurious force drops from the anti-rotate bars):
  a station is excluded from force scoring if its force is more than 1 % of F_max below the rolling
  maximum of force over ±5 frames.
- DIC–FE levelling: strains are not taken from VIC-3D. Green–Lagrange strains are computed from
  displacements at the DIC points with one operator (local affine least-squares fit over neighbours
  within 1.0 mm) applied identically to the measured (U, V) and to the FE displacements interpolated to
  the same points. Points that are NaN / unmatched (sigma = −1) at any scored station, or outside the
  mesh, are dropped. (Comparison with VIC-3D's own strains is reported but not graded.)

## Model forms (compared on calibration + development data only)

Isotropic hardening σ_y(α) with von Mises or plane-stress Hill-48 yield:
- **M1**: von Mises + Voce, σ_y = s0 + Q(1 − e^{−bα})  (3 parameters)
- **M2**: von Mises + Voce + linear, σ_y = s0 + Q(1 − e^{−bα}) + Hα  (4)
- **M3**: von Mises + double Voce + linear  (6)
- **M4**: Hill-48 (a, h, c; yield along the loading direction = σ_y) + the hardening law of the better of M2/M3  (7 or 9)

FEMU objective (calibration specimens O1-3 + O14-3, equal weight per specimen):
J = mean_j[(F_sim − F)/ (0.02·mean F)]² (stations 5…40, drops excluded)
  + mean over stations {13, 27, 40}, points, components [(E_sim − E)/(0.10·rms(E_meas at that station))]².
Levenberg–Marquardt with exact sensitivities; several starts.

**Selection rule:** score every form on the development specimen O5-4 with
S = force NRMSE / 0.05 + mean strain-field error / 0.15 (metrics below); the **primary model** is the
form with the fewest parameters whose S is within 10 % of the best S. All forms are scored on the
held-out set and reported; the verdict is read on the primary model.

**Uniaxial baseline:** the primary model's hardening law, von Mises, fitted by least squares to the
mean true-stress–plastic-strain curve of BT3/CT3/CT4 up to maximum force (A0 = 12.7 × 1.5 mm²), scored
with the identical script. (Hill coefficients cannot be identified from one uniaxial direction, so
the baseline is isotropic by construction.)

## Metrics and pass tolerances (per held-out specimen, primary model)

1. **Force curve**: NRMSE_F = rms_j(F_sim − F_meas) / mean_j(F_meas), stations 5…40 without excluded
   drops. **Pass ≤ 5 %.**
2. **Force at the last station** (0.9 x_F): |F_sim − F_meas| / F_meas. **Pass ≤ 5 %.**
3. **Strain field** at stations 13, 27, 40 (≈ 30, 60, 90 % of 0.9 x_F): relative L2 error
   e_E = sqrt(Σ (ΔExx² + ΔEyy² + 2ΔExy²)) / sqrt(Σ (Exx² + Eyy² + 2Exy²)) over all kept DIC points.
   **Pass ≤ 15 % at each of the three stations.**
4. **Strain hot spot** at station 40: mean of the top 1 % of the major principal Green–Lagrange strain,
   FE vs DIC (levelled). **Pass within 15 %.**

Because the boundary displacements are imposed, metric 3 is partly satisfied by kinematics alone. To
show how much, the same script scores a **null model** (the material forced to stay elastic) on every
specimen; this is reported next to each strain score, not graded.

Orientation transfer (XR5): same four metrics and tolerances, reported as a separate verdict.

Overall verdict: "generalises" only if both primary held-out specimens pass metrics 1–4 with the
primary model. Whether full-field data helped: compare primary vs uniaxial baseline on metrics 1–4.

## Verification gates

- **Warm-up (synthetic truth)**: data generated on the O1 outline with a fine mesh (h ≈ 0.6 mm near
  features) from a known M4-type model, noise added (displacement σ = 3.6 µm, force σ = 2 N, the
  paper's Table 6 / load-cell figures), identified on a different, coarser mesh from a start ≥ 30 % off.
  **Pass:** recovered σ_y(α) within 2 % of truth for all α ∈ [0, α_95] (α_95 = 95th percentile of
  equivalent plastic strain in the truth solution) and Hill coefficients a, h within 0.05, c within 0.3.
  Gradient check: direct sensitivities vs central finite differences, relative error ≤ 1e-5.
- **Convergence** of the headline held-out predictions: mesh (calibration mesh vs one with half the
  feature size) and load step (40 vs 80 increments); report the change in metrics 1–4. A pass that
  flips under refinement is reported as such.
- **CalculiX cross-check**: the identified von Mises model (the primary model if it is von Mises,
  otherwise the best von Mises form) re-run in CalculiX 2.23 (CPS6, NLGEOM, tabulated hardening, same
  mesh and boundary displacements) for one primary held-out specimen. **Pass:** force curves agree
  within 2 % NRMSE and levelled strain fields within 3 % relative L2 at station 40.
- **Identifiability**: for each form, the normalised Gauss–Newton Hessian (JᵀJ in log-parameters) —
  eigenvalues, parameter correlation, and linearised 95 % intervals — plus multi-start scatter.

## Amendment 1 — 16:42 UTC, before any fit and before any held-out file was opened

Found while pre-processing the two calibration files (no model had been run on data):
- The force-drop rule as written ("rolling maximum over ±5 frames") also flags every station on the
  steeply rising early part of the curve, because the *following* frames are higher. Corrected to the
  rolling maximum over the **preceding** 5 frames (the station frame included). On O1-3 this flags
  nothing; on O14-3 it flags stations 38 and 40, which sit in a band of load-cell chatter (force jumping
  24–27.6 kN between consecutive frames).
- Consequently metric 2 ("force at the last station") is evaluated at the **last non-excluded
  station** when station 40 itself is excluded.
Nothing else is changed: split, metrics, tolerances and gates stand as written above.
