# VHB 4910 cyclic uniaxial data — provenance

Immutable source data for the viscohyperelastic calibration experiment. Do not edit.

## What this is

Uniaxial **cyclic loading–unloading** stress–stretch histories of **VHB 4910**
(a very-high-bond acrylic elastomer), at three stretch rates and four maximum
stretches — strongly rate-dependent with clear hysteresis.

Each file: whitespace-delimited columns
`time[s], dt[s], λ1, λ2, λ3, stress` (incompressible uniaxial: λ2 = λ3 = λ1^-1/2).
182 rows each. Stress is the value as digitized from the source figures (see
"units" below).

| file group | λ_max | rates (file suffix) |
|---|---|---|
| `Fig3_1_rate{1,2,3}` | ~2.25 | rate1=0.01, rate2=0.03, rate3=0.05 s⁻¹ |
| `Fig3_2_rate{1,2,3}` | ~4.0  | " |
| `Fig4_1_rate{1,2,3}` | ~6.25 | " |
| `Fig4_2_rate{1,3}`   | ~8.9  | (rate2 not present in source) |

## Origin & license

- Redistributed from the **iCANN** archive: *Theory and implementation of
  inelastic Constitutive Artificial Neural Networks: Source code and data*,
  Zenodo record **10066805**, file `iCANN_Data.zip`, folder `02_Example02`.
  License **CC-BY-4.0**. <https://zenodo.org/records/10066805>
- The experimental data itself is digitized from:
  **M. Hossain, D. K. Vu, P. Steinmann (2012)**, *Experimental study and
  numerical modelling of VHB 4910 polymer*, Computational Materials Science 59,
  65–74. (Figures 3 & 4 — cyclic loading at multiple stretch rates.)

Cite both the iCANN archive and Hossain et al. (2012) when using this data.

## Units note

The stress column is reproduced from the digitized source as-is. Hossain et al.
(2012) report **nominal (engineering) stress** P = force / undeformed area vs.
stretch; the calibration uses the nominal-stress convention and verifies the fit
accordingly. The absolute magnitude is large (peak ~20–28 in the file units) — we
calibrate in the data's own units; the inverse problem and held-out validation
are unit-agnostic.
