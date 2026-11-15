# raw/dmc — Damage Mechanics Challenge calibration package (blind-prediction fallback for SFC3)

- **Dataset:** Jiang, L., Yoon, H., Pyrak-Nolte, L. J., Bobet, A., Morris, J. (2021/2022). *Calibration Data Set for
  Damage Mechanics Challenge on Brittle-Ductile Material.* Purdue University Research Repository.
  doi:10.4231/QF39-Q924. Version 1.0, published 19 Jan 2022. Licence CC BY 4.0.
- **Landing page:** https://purr.purdue.edu/publications/3905/1
- **Downloaded:** 2026-11-14 ~21:00 UTC from https://purr.purdue.edu/publications/3905/serve/1?render=archive
  (single 27.7 MB zip, no login needed).
- Files are unmodified. The outer download was unpacked once; the inner zips are stored as delivered.

| File | Size | What it is | Git? |
|---|---|---|---|
| `large/purr3905_download_bundle.zip` | 27.7 MB | the download exactly as served | **no** (>10 MB) |
| `large/DamageChallenge_SlideDeck_01.pdf` | 20.8 MB | Challenge information packet (68 slides, updated 15 Jan 2022, SAND2021-14877 O): challenge statement, geometry, questions, test set-up, calibration plots | **no** (>10 MB) |
| `README.txt` | 1 kB | PURR archival readme | yes |
| `LD.zip` | 3.0 MB | load–displacement CSVs: 3-point bending HC/HA/HB/H45 (groups 3–6), UCS (H, T, V ×5), Brazilian (HBT ×3). Column 1 load [N], column 2 displacement [mm] | yes |
| `FractureSurface.zip` | 1.2 MB | laser-profilometry fracture surfaces of the 16 calibration beams (250×120 grid, 0.1 mm step, asperity in mm) | yes |
| `MaterialProperties.zip` | 3.9 MB | ultrasonic waveforms on 2-inch cubes, XRD | yes |

**Contains no challenge-geometry test results** (the packet says "HCH: challenge data (not shown)"). The supplementary
500 GB DIC / X-ray CT set (doi:10.4231/2E8M-W085, Globus) was not downloaded.

**Not opened, by design (blind protocol):** Morris et al. 2024 (challenge outcome paper), any participant paper,
and any PURR record holding the challenge-geometry results.
