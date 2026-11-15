# raw/sfc3 — Third Sandia Fracture Challenge (SFC3): access record

**Nothing could be downloaded.** This folder records the attempt (2026-11-14) so the next session does not repeat it.

| What | URL | Result (2026-11-14) |
|---|---|---|
| MDF landing page | https://acdc.alcf.anl.gov/mdf/detail/pub_29_kramer_third_v1.2/ | HTTP 200, landing page only. DOI 10.18126/M26D20. |
| Data (Globus collection 82f1b5c6-6e9b-11e5-ba47-22000b92c6ec, path `/published/publication_625/`) | https://data.materialsdatafacility.org/published/publication_625/ | Redirects to auth.globus.org login. Needs an interactive Globus identity; none available on this box. |
| Sandia RAIICE catalog entry | https://www.sandia.gov/raiice/?p=2213 | Links back to the MDF record only. |
| Summary paper, author manuscript (Kramer et al., Int. J. Fract. 2019, doi 10.1007/s10704-019-00361-1) | https://dspace.mit.edu/bitstream/1721.1/131747/1/10704_2019_361_ReferencePDF.pdf | HTTP 429 from the MIT DSpace web firewall on 4 tries over 20 min ("log in, then open the item again"). Not circumvented. |
| Springer version | https://link.springer.com/content/pdf/10.1007/s10704-019-00361-1.pdf | Paywall HTML. |
| OSTI | title search "Sandia Fracture Challenge" | Only post-deadline results talks / team papers (1468320, 1497555, 1640719, 1640209, 1507401, 1559548, 1570251, 2002166). Not opened (blind protocol). |

To unblock: a person with a (free) Globus account can download `/published/publication_625/` from the MDF collection
and drop it here. The blind fallback named in the task brief (Damage Mechanics Challenge) was used instead — see `raw/dmc/`.
