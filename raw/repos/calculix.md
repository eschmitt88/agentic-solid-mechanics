---
title: "CalculiX — Free 3D Structural Finite Element Program (CCX solver + CGX pre/post)"
source_urls:
  - http://www.calculix.de/
  - http://www.dhondt.de/
  - https://github.com/Dhondtguido/CalculiX
fetched: 2026-06-16
---

# CalculiX — source capture

## What it is (from www.calculix.de)

CalculiX is "A Free Software Three-Dimensional Structural Finite Element
Program." It comprises two main components:

1. **CCX (CalculiX CrunchiX)** — the solver. Performs linear and nonlinear
   calculations: "Static, dynamic and thermal solutions."
2. **CGX (CalculiX GraphiX)** — an interactive 3D pre/post-processor "using
   the openGL API" for model building and result visualization.

## Input / output formats

- **Input**: Abaqus input format — ".inp" text deck. "The solver makes use of
  the abaqus input format." Keyword-driven ASCII deck (e.g. `*NODE`,
  `*ELEMENT`, `*MATERIAL`, `*STEP`, `*STATIC`, `*BOUNDARY`, `*CLOAD`,
  `*NODE FILE`, `*EL FILE`).
- **Output**:
  - `.frd` — CalculiX results file (nodal/element field results; the format
    CGX reads for post-processing).
  - `.dat` — ASCII output of requested results (e.g. via `*NODE PRINT`,
    `*EL PRINT`), human/parser-friendly tabular text.
  - Also exports to nastran, ansys, code-aster, dolfyn, duns, ISAAC, and
    OpenFOAM formats.

## CLI invocation

Standard CalculiX usage is `ccx jobname`, where the solver reads
`jobname.inp` and writes `jobname.frd` and `jobname.dat`. (The exact flag
syntax such as `-i` is documented in the CCX user manual; invocation shape
`ccx <jobname>` is the established convention. Specific flags = unverified
from fetched pages.)

## Solid-mechanics capabilities (from www.calculix.de)

- Static and dynamic structural analysis, including nonlinear effects.
- Eigenfrequency calculation, cyclic symmetry (jet-engine example).
- Thermal solutions.
- Contact, plasticity / material nonlinearity, and large-deformation
  (geometric nonlinearity, NLGEOM) are standard CalculiX features documented
  in the CCX keyword manual. (Detailed enumeration = drawn from the manual;
  per-feature specifics not re-quoted here = unverified from fetched text.)

## Version & license

- **Current version: 2.23** (per www.calculix.de and www.dhondt.de).
- GitHub mirror (Dhondtguido/CalculiX) README states "Version 2.22 of
  CalculiX is available!" — mirror slightly behind the official site.
- **License: GNU General Public License v2 or later (GPL-2.0+).** "This
  program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free
  Software Foundation; either version 2 of the License, or (at your option)
  any later version."

## Documentation / keyword manual (from www.dhondt.de)

CCX reference manuals are downloadable from www.dhondt.de:
- HTML: `ccx_2.23.htm.tar.bz2`
- PDF: `ccx_2.23.pdf`
- PostScript: `ccx_2.23.ps.tar.bz2`

This is the authoritative CCX keyword manual (every `*KEYWORD`, element type,
material model, and analysis step documented).

## Install paths

- Debian/Ubuntu: `apt install calculix-ccx` (package name `calculix-ccx`;
  provides the `ccx` binary). `calculix-cgx` for the pre/post-processor.
- conda-forge: `conda install -c conda-forge calculix` (provides ccx).
- Source build from the GitHub mirror: Fortran (74.8%) + C (24.8%); `src/`
  holds solver + preprocessor, `doc/`, `test/`, `pictures/`. GPL-2.0.

## Python helper wrappers (ecosystem)

- **ccx2paraview** (`pip install ccx2paraview`) — converts CalculiX `.frd`
  output to ParaView VTK/VTU for visualization/parsing.
- **PyCCX** — Python interface to drive CCX (build model, run, read results).
- **pycalculix** — Python library to build 2D models, mesh, run CCX, and read
  results (helper for slices/axisymmetric).

(Wrapper capabilities from general ecosystem knowledge; PyPI page fetch for
ccx2paraview failed to load — wrapper feature details = unverified from a
live fetch this session.)

## Repository facts (GitHub Dhondtguido/CalculiX)

- 174 stars, 58 forks, 411 commits on master (at fetch time).
- Discourse community: calculix.discourse.group.
