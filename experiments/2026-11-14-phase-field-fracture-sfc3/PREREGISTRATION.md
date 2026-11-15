# Rung 1 pass criteria — fixed 2026-11-14 20:42 UTC, before the first solver run

(Copied verbatim into README.md "Setup"; kept as a separate file so the timestamp is visible.)

## Reference values (all read this session; by-eye figure read-offs are ±3 % on load, ±0.1e-3 mm on displacement)

| Case | Source (opened this session) | peak load [kN] | u at peak [mm] |
|---|---|---|---|
| Tension, l=0.015 | Molnár & Gravouil 2017, Fig. 7 (HAL hal-02132509) | 0.70 | 5.9e-3 |
| Tension, l=0.015 | Miehe 2010 curve as digitized in phasefieldx repo CSV (github.com/CastillonMiguel/phasefieldx, reference_solutions/miehe_solution.csv) | 0.701 | 5.55e-3 |
| Tension, l=0.015 | Zhou, Rabczuk, Zhuang, arXiv 1902.05922, Fig. 7 | 0.80 | 5.5e-3 |
| Tension, l=0.0075 | Molnár & Gravouil 2017, Fig. 7 | 0.73 | 5.9e-3 |
| Tension, l=0.0075 | Zhou et al., arXiv 1902.05922, Fig. 7 | 0.74 | 5.2e-3 |
| Tension, l=0.0075 | Zhang et al., arXiv 1706.05449, Fig. 8 | 0.68 | 5.0e-3 |
| Tension, l=0.0075 | Zhang et al., arXiv 1805.07357, Fig. 7 | 0.73 | 5.0e-3 |
| Shear, l=0.0075 | Zhou et al., arXiv 1902.05922, Fig. 13 | 0.60 | 0.0110 |
| Shear, l=0.0075 | Zhang et al., arXiv 1706.05449, Fig. 16b | 0.595 | 0.0108 |
| Shear, l=0.015 (secondary) | Zhou et al., arXiv 1902.05922, Fig. 13 | 0.54 | 0.0100 |
| Shear, l unstated (secondary) | Miehe curve digitized in phasefieldx CSV (miehe_solution_shear.csv) | 0.531 | 0.0097 |
| Shear crack angle, l=0.0075 | Molnár & Gravouil 2017, text p.16: "initiating with a β = 61° angle from the direction of the deformation" | 61° | – |
| Shear crack end point | by eye from Molnár Fig. 6c (x≈0.83) and Zhou Fig. 12c/f (x≈0.88): crack meets the bottom edge at x ≈ 0.85 | – | – |

Not opened (paywalled), so NOT used for grading: Miehe et al. 2010 CMAME/IJNME originals, Ambati et al. 2015.
Excluded up front: Zhang 1805.07357 shear (0.65–0.67, uses a modified crack-boundary treatment), Jodlbauer–Wick
(different length scale 0.004), Kristensen (l=0.024, Amor split), Lampron (AT1) — shown as scatter context only.

## Headline configuration (the thing that is graded)
AT2 phase field, spectral (Miehe) tension/compression split, history-field irreversibility, plane strain,
single-pass staggered scheme with Miehe-style load steps (tension: Δu=1e-5 mm to u=5e-3, then 1e-6;
shear: Δu=1e-5), finest mesh run of the convergence series. Reference target = arithmetic mean of the
matched-l sources above.

## Pass criteria (not to be moved)
| # | Quantity | Reference (mean) | Tolerance |
|---|---|---|---|
| T1 | Tension l=0.015 peak load | 0.734 kN | ±10 % |
| T2 | Tension l=0.015 displacement at peak | 5.65e-3 mm | ±10 % |
| T3 | Tension l=0.0075 peak load | 0.720 kN | ±10 % |
| T4 | Tension l=0.0075 displacement at peak | 5.28e-3 mm | ±10 % |
| T5 | Tension crack path | straight along y=0.5 to the right edge | centreline deviation ≤ l |
| S1 | Shear l=0.0075 peak load | 0.598 kN | ±10 % |
| S2 | Shear l=0.0075 displacement at peak | 0.0109 mm | ±10 % |
| S3 | Shear initial crack angle below the notch line (fit of centreline 0.02–0.10 mm from tip) | 61° | ±10° |
| S4 | Shear crack path | single branch, curves down, meets bottom edge at x=0.85 | ±0.07 mm, no second crack |
| C1 | Mesh convergence of every headline peak load | change between two finest meshes | ≤ 3 % else "not converged" flag |

Secondary (reported, not part of pass/fail): shear l=0.015 vs 0.54 / 0.531 kN; sensitivity to load step and
to staggered iteration (single pass vs iterated to convergence); ablations without energy split and without
irreversibility; surface-energy check Γ·Gc vs Gc × crack length.
