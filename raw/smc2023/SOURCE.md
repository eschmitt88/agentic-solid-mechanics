# raw/smc2023 — Sandia Mechanics Challenge 2023 information packet (blind-prediction target of trial 10)

- **Document:** Kramer, S. L. B., Ivanoff, T. A., Corona, E. (2023). *Sandia Mechanics Challenge 2023 Information
  Packet: Structure with a Threaded Fastener Joint.* Sandia National Laboratories report SAND2023-06848, printed
  25 July 2023, 102 pages. OSTI ID 2430623.
- **Downloaded:** 2026-11-15 16:30 UTC from https://www.osti.gov/servlets/purl/2430623 (no login needed).
  Record page: https://www.osti.gov/biblio/2430623
- **Licence / terms:** work of a US Department of Energy laboratory, released through OSTI. The packet states
  (§1.3): "All data released in this packet may be used freely with proper attribution to this information packet
  and DOI provided for data in the Materials Data Facility entry." Data DOI: 10.18126/gmi9-62tm.
- The file is unmodified (moved once into `large/` at ingest; sha256 below).

| File | Size | What it is | Git? |
|---|---|---|---|
| `large/SAND2023_smc2023_information_packet_osti2430623.pdf` | 36.9 MB | the information packet as served; sha256 `9c2b8744b8630545fe60519acaf56a8a1d24398a41af048daa08d991d6217d02` | **no** (>10 MB) |

**Not obtained:** the SMC data package on Materials Data Facility (`smc2023_v1.1`,
https://acdc.alcf.anl.gov/mdf/detail/smc2023_v1.1): calibration data files (SI and US units), measured
acceleration–time profiles of the 19 tests, measured part dimensions, the October 2023 supplemental package
(single-joint validation tests, DIC, fractography). The landing page loads, but the only data link is a Globus
file-manager URL (`origin_path=/mdf_open/smc2023_v1.1/`), which needs a Globus login; direct HTTPS guesses returned
nothing (checked 2026-11-15).

**Used from the packet in trial 10:** drawings (Fig. A-3: rod Ø1.675 × 6.25 in with Ø3.0 × 0.25 in flange, six Ø0.221
holes on a Ø2.52 circle), Table 4-1 (19 tests: ΔV, peak acceleration, impact duration, weights), Fig. 3-1 (AA6061
tension), Fig. 3-5/3-6 (fastener tension), Fig. 3-7 (fastener double shear), §2.1 (assembly, torque 60–63 in-lbf),
§5 (quantities of interest, points A/B/C, coordinate system). All calibration numbers are read-offs from the
figures.

**Not opened, by design (blind protocol):** the outcome. In particular Mersch, Romero & Corona, post-blind report,
OSTI 3013870 (https://www.osti.gov/biblio/3013870) — its bytes were fetched once into a scratch file during URL
probing and overwritten unread; see `experiments/2026-11-15-contact-ladder/log.md`, entry 16:29. No participant
paper or results presentation was opened either.
