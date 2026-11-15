# raw/sandia-304l-dic — Sandia 304L sheet: DIC + IR for seven geometries (track E data)

- **Dataset:** Jones, E. M. C., Reu, P. L., Kramer, S. L. B., Jones, A. R., Carroll, J. D., Karlson, K. N.,
  Seidl, D. T., Turner, D. Z. *Data set accompanying "Full-Field Digital Image Correlation (DIC) and Infrared
  Thermography (IR) Data for Seven Unique Geometries of 304L Stainless Steel Sheet Metal".* Figshare+,
  doi:10.25452/figshare.plus.25483534 (v1, published 2024-08-29). **Licence: CC0.**
- **Descriptor paper:** Scientific Data 11:1101 (2024), doi:10.1038/s41597-024-03949-y (open access,
  CC BY-NC-ND 4.0). Saved as `paper/article.pdf`, `paper/article.html`, `paper/article.txt` (pdftotext).
- **How it was found:** Crossref record of the paper → data DOI → Figshare API
  `https://api.figshare.com/v2/articles/25483534` (saved verbatim as `figshare_article_25483534.json`:
  200 files, direct `https://ndownloader.figshare.com/files/<id>` URLs, md5 sums). **No login needed.**
- **Downloaded:** 2026-11-15 16:30–16:44 UTC with curl. Every file with an md5 in the API record was
  verified against it. Files are unmodified.
- The full record is several hundred GB (raw DIC image stacks, IR sequences, calibration images, processed
  `*-Data.mat` per specimen). Only the processed, registered `*-Data.mat` (MATLAB v7.3 / HDF5) of the six
  slow tests used were fetched, plus the small metadata.

| Path | Size | What | Git? |
|---|---|---|---|
| `figshare_article_25483534.json` | 0.2 MB | Figshare API record (file list, URLs, md5, licence) | yes |
| `paper/article.{pdf,html,txt}` | 2.9 MB | data descriptor paper | yes |
| `metadata/TestMatrix.pdf`, `SpecimenDrawings.pdf`, `304L cert rolled alloys.pdf` | 0.6 MB | test matrix, drawings, mill certificate (scanned image) | yes |
| `geometry/*-NominalGeometry.mat` (8) | < 0.1 MB | nominal outlines (`bOut`, `bHole`) of O1, O5, O14, O15, O18, D, X, dog bone | yes |
| `dogbone/*-Data.mat` (27) | 2.1 MB | tensile dog bones: time, virtual-extensometer strain (22 mm), force; 3 rates × 3 orientations | yes |
| `large/calibration/O1-3-Data.mat` | 3.74 GB | bespoke 1, transverse, slow — md5 5829e8bf5a35d0e8a6ed8f41056dbcb2 | **no** |
| `large/calibration/O14-3-Data.mat` | 3.56 GB | bespoke 14, transverse, slow — md5 41c0b906312a3e6fd5d88372d3445438 | **no** |
| `large/calibration/O5-4-Data.mat` | 5.82 GB | bespoke 5, transverse, slow — md5 38a79df38d9b67b9bfd87116936904c9 | **no** |
| `large/heldout/O15-3-Data.mat` | 8.32 GB | bespoke 15, transverse, slow — md5 02362f68c8aa9deaff730ed0b25c8a83 (the API record carries no md5 for this file; size 8 315 419 934 bytes equals the record) | **no** |
| `large/heldout/XT10-Data.mat` | 2.31 GB | X-specimen, transverse, slow — md5 6245561662ca6131471028d2344901cc | **no** |
| `large/heldout/XR5-Data.mat` | 2.31 GB | X-specimen, rolling, slow — md5 ae46bce5e675f3151d5fcbb2b0756d5c | **no** |

`large/` (26 GB) is excluded from git by `.gitignore` in this folder. To re-fetch: file ids
45310135 (O1-3), 45310096 (O14-3), 45310705 (O5-4), 45310729 (O15-3), 45316885 (XT10), 45316882 (XR5).

**Not downloaded:** O18-3 (15.3 GB, the fifth slow bespoke test), every fast test (they self-heat; out of the
modelling scope), all raw image / IR / calibration zips, Movies.zip.

**Variables in `*-Data.mat`** (paper Table 4): `time` s, `extension` mm (grip LVDTs), `force` kN, `force6DOF`,
`X0 Y0 Z0` mm (undeformed DIC points, registered to the nominal outline), `U V W` mm, `Exx Eyy Exy`
(Lagrange, VIC-3D, subset 23 px, step 3 px, strain window 5), `sigma` px (−1 = unmatched), `IR_temperature` °C.
Arrays are stored (time, point). Noise floors (paper Table 6): in-plane displacement 3.6 µm, strain 3.3e-4.
