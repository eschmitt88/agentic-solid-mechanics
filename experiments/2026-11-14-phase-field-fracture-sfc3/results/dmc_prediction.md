# Blind prediction — Damage Mechanics Challenge geometry

**Status: FROZEN 2026-11-15 (UTC), not graded.** Made from the organisers' information packet (15 Jan 2022) and
calibration files only (`raw/dmc/`). This replaces the SFC3 prediction the brief asked for: the SFC3 data package
and problem statement could not be obtained (`raw/sfc3/SOURCE.md`), so the named fallback challenge was used.
Machine-readable version: `dmc_prediction.json`. Curve: `dmc_prediction_Q1.txt`. Crack surface: `dmc_prediction_crack_surface.csv`.

## What was asked (information packet, slides 13–18)
- Q1: force–displacement curve (vertical displacement at the central rod) from first loading through post-peak.
- Q2: crack position as a function of load. Q3: displacement fields of the front and back faces.
- Bonus: coordinates of the fracture surfaces. Discussion: model variability.
Coordinates: x along the beam from the loading rod, y through the width (front −6.35, back +6.35 mm), z up from the top face (bottom z = −25.4 mm).
Challenge notch: 1 mm wide, x = −12.7 mm on the front face to −6.35 mm on the back face (26.6° from the cross-section), depth 5.08 mm (front) tapering to 2.54 mm (back).

## Prediction

| Quantity | Value | Uncertainty |
|---|---|---|
| Peak load | **298 N** | 1σ ±24 N; 80 % interval 266–329 N (mean of the test series), 259–337 N (one specimen) |
| Displacement at peak (as the LVDT would record it, toe included) | 0.41 mm | 1σ ±0.09 mm |
| Loading stiffness between 30 % and 80 % of peak | 800 N/mm | calibration beams scatter ±20 % |
| Beam deflection alone at peak (no contact compliance) | 0.16 mm | – |
| Post-peak | abrupt (unstable): the structural solution falls to 11 % of peak within one increment of 0.0026 mm; in the measured record this is a vertical drop to about 2 % of peak followed by a low tail | – |

Crack path (Q2 / bonus), x-position of the crack [mm] on three planes through the width:

| height | front (y=-6.35) | mid-width (y=0) | back (y=+6.35) |
|---|---|---|---|
| z=-20.0 mm | -11.6 | -9.6 | -7.1 |
| z=-17.5 mm | -9.6 | -8.1 | -7.1 |
| z=-15.0 mm | -8.6 | -7.6 | -7.1 |
| z=-12.5 mm | -7.6 | -7.1 | -6.6 |
| z=-10.0 mm | -6.6 | -6.6 | -6.6 |
| z=-7.5 mm | -6.1 | -6.1 | -6.1 |
| z=-5.0 mm | -6.1 | -6.1 | -6.1 |
| z=-2.5 mm | -6.1 | -6.1 | -6.1 |
| z=-1.0 mm | -6.1 | -6.1 | -6.1 |

- Initiation: at the notch root; first on the deep (front, y=-6.35) side of the tapered notch.
- Shape: single crack; twists from the 26.6-degree skewed notch plane towards a plane normal to the beam axis and drifts towards the loading rod (x=0).
- Top exit: the raw model puts the crack at x = -6.1 mm on all three planes when it reaches the top face. On the calibration beams the same model and mesh under-predict the sideways drift towards the loading rod in the upper half of the beam by 2.3 mm on average (HB 2.8 mm, HA 1.8 mm). Mesh refinement removes most of that for HB but not for HA, so part of it is the model, not the mesh. **Predicted top exit: x ≈ -3.8 mm (range -6.4 to -2.8 mm), the same across the width to within about 1 mm.** Below z ≈ −12 mm the tabulated raw values are the prediction.

Not predicted: Q3 displacement fields, fracture-surface roughness, crack position as a function of load during the
unstable drop (the model breaks the ligament within one displacement increment).

## Model choices
- Method: AT2 phase-field fracture, hybrid formulation (isotropic degradation, spectral tensile driving force, history-field irreversibility), staggered to convergence. Solver: own JAX code pf3d.py, hex8 tensor grid, GPU Jacobi-PCG; 95,020 elements, h = 0.5 mm in the crack region, 26 elements through the width.
- Elasticity: isotropic, E = 1400 MPa, ν = 0.2. E is a compromise between the compression-test slopes
  (≥ ~1 GPa if the cylinders are 25 mm × 51 mm, which the packet does not state) and the ultrasonic moduli (several GPa). It only affects the
  split of the displacement between beam and contact, not the peak load (which depends on E·Gc).
- Fracture: length scale l = 1.0 mm, Gc = 0.0514 N/mm, one value fitted to the mean peak loads of the four calibration geometries.
  l ≥ 2 mm was rejected because the far-offset beam (HA) then breaks at mid-span instead of at the notch, contrary to the photographs.
- Loading: line displacement under the central rod, line supports 30.48 mm either side; no damage allowed within 2 mm of the rods.
- Displacement record: delta = -0.128 + delta_FE + 0.0438 * P^0.379 (mm, N), fitted to the 16 calibration beams. The LVDT record of the calibration beams is 2–3× softer than the beam itself;
  this empirical term stands for rod indentation and fixture compliance.
- Not modelled: anisotropy of the printed layers (ultrasonic velocities differ by ~15 % between directions), rate effects, friction at the rods,
  dynamics of the unstable crack run.

## Calibration fit (not blind)
| Geometry | measured mean peak [N] | model [N] | error |
|---|---|---|---|
| HC | 216 | 212 | -2.0 % |
| HB | 243 | 250 | +3.1 % |
| HA | 379 | 411 | +8.4 % |
| H45 | 270 | 246 | -8.7 % |

## Sensitivity of the predicted peak load (each variant re-calibrated on the same four geometries)
| Variant | Predicted peak |
|---|---|
| base l=1.0 h=0.5 | 295 N |
| finer mesh l=1.0 h=0.35 | 303 N |
| smaller l=0.6 h=0.3 | 295 N |

The point prediction is the mean of these variants.

## Uncertainty estimate (my own)
Relative 1σ contributions to the peak load: model form 7.4 % (scatter of the four calibration geometries about the
one-parameter fit), numerical / length-scale choice 2.0 %, uncertainty of the calibration means 3.1 %,
specimen-to-specimen scatter 6.1 % (applies to a single specimen). Combined: 8.2 % for the mean of the
test series, 10.2 % for one specimen.
Known direction of bias: the model over-predicts the two offset-notch beams (+3 %, +8 %) and under-predicts the rotated-notch beam (−9 %).
The challenge notch is both offset and rotated, so the two errors should partly cancel, but if one dominates the truth could sit near either
end of the interval. I would not be surprised by anything between 265 and 335 N; I would be surprised by a value outside 240–360 N.
The crack path is less certain than the peak load: the lower half (up to z ≈ −12 mm) matched the calibration scans to about 0.5 mm, but the upper half is biased towards vertical (partly the structured mesh, partly the model); the top-exit estimate carries a calibration-based correction of 2.3 mm and an uncertainty of roughly ±1.5 mm. The through-width untwisting of the crack (from the skewed notch plane into a cross-beam plane within about 10 mm of height) is the robust part: it is seen in the H45 calibration scans and reproduced by the model.

## Blindness
No outcome information for the challenge geometry was read, searched for, or opened. Before predicting I recorded what I remembered
(`log.md`, 21:00): that a participant paper using a phase-field model exists and reports good agreement; no numbers.
