# QA review surface

A **static, interactive** human-QA site for the agentic FEA trials, served from
GitHub Pages at `docs/qa/` (linked from the project landing page → *QA review*).
No running service — unlike the agentic-CAD viewer, FEA experiment results are
frozen once a run completes, so the QA surface is baked and pushed.

Open: <https://eschmitt88.github.io/agentic-solid-mechanics/qa/>

## Two QA axes per experiment

1. **Physics correctness** — interactive 3D deformed/stress scenes (rotate &
   zoom in the browser, no plugin), mesh-convergence and design-space plots,
   topology density fields.
2. **Agentic legitimacy** — did the agent genuinely do the work: a **leakage
   scan** (the agent's own files contain no reference/grader reads), an
   **independent recompute** of the headline metric, FE-eval / turn counts, and
   pass@k. This axis is the point of an *agentic* project and is what the
   geometry-only CAD viewer doesn't cover.

## Regenerate

Runs headless (VTK EGL offscreen — no X server / no Xvfb). Requires the
`solidmech` micromamba env with `ccx`, `ccx2paraview`, `pyvista`, `meshio`,
`matplotlib`, `trame`:

```sh
cd ~/projects/research/agentic-solid-mechanics
~/.local/bin/micromamba run -n solidmech python _meta/qa/build_qa.py
git add docs/qa && git commit -m "qa: rebuild" && git push
```

## Interactive 3-D scenes with a live density-threshold slider

The 3-D topology scenes (trial 4) are NOT pyvista `export_html` snapshots —
those bake the threshold server-side, so the hidden material isn't in the file
and can't be revealed. Instead `qa_lib.write_threshold_scene_3d()` writes a
self-contained Three.js page: every cell (density ≥ floor) is embedded once,
sorted by density, and drawn as a cube via an `InstancedMesh`. Because instances
are density-sorted, "show density ≥ t" is just "render the first K instances"
(`mesh.count = K`, K from a binary search), so the slider re-thresholds instantly
client-side — no server, no re-export. Three.js loads from a CDN (`unpkg`); a
static PNG (still rendered by pyvista) is the fallback behind a try/catch.

The pure-JS logic (data embedding, K(t), viridis) and the Three.js API calls are
node-checkable headlessly; only the final WebGL paint needs a browser.

## How it works

- `qa_lib.py` — `.frd → .vtu` (ccx2paraview), warp-by-displacement + colour-by-σ
  render to **PNG (static fallback)** + **self-contained interactive HTML**
  (`export_html`), matplotlib plots, topology heatmaps, the leakage scan, and the
  page templating.
- `build_qa.py` — one builder per experiment + the dashboard index. The committed
  decks only `*NODE PRINT` to `.dat`, so for the design trials the showcased
  design is **re-solved** with `*NODE FILE`/`*EL FILE` (fields patched into the
  deck) to colour the scene by stress. Trial 1 reuses the agent's field-bearing
  `.frd` directly.

## Page structure (write for a newcomer)

Every trial page must be self-contained for a reader with no project context:

1. **Objective** — one plain-language paragraph: what is being tested and why.
2. **The problem & how it is held** — a boundary-condition diagram
   (`qa_lib.bc_beam` / `bc_domain_2d` / `bc_domain_3d` / `bc_mbb`) plus a table
   of structure / material / supports / load / goal.
3. **What is measured** — the metric and the pass criterion, in plain terms.
4. Results (interactive scene + plots), then the **"Did the agent really do
   this?"** checks panel.
5. A **`terms_block([...])`** glossary footer — every acronym used on the page
   must have an entry in `qa_lib.GLOSSARY` and be listed here. Spell acronyms out
   on first use in the prose too (FEA, SIMP, OC, MBB, …). No insider phrases
   ("loop-2 substrate", "forward-model gold", etc.).

`qa_lib.intro_block(objective, setup_rows, bc_png, bc_caption, measured)` renders
items 1–3; use it at the top of every builder.

## Adding an experiment

Add a `build_trialN(slug)` to `build_qa.py` returning
`{slug, title, axis, headline}`, append it in `main()`. Open with the
`intro_block` + a BC diagram, close with a `terms_block`. Reuse `qa_lib`
helpers; degrade gracefully (a failed scene still emits the page + checks panel).

## Gotchas (learned the hard way)

- NumPy 2.x removed `ndarray.ptp()` — use `np.ptp(arr)`.
- pyvista colours need full 6-digit hex (`#888888`), not `#888` or 8-digit rgba.
- The reference density is saved image-oriented (`reshape(nelx,nely).T`); the
  agent saved the **raw flat element vector** (column-major `elx*nely+ely`).
  Match orientation with `reshape(nelx,nely).T` before imshow — a naive
  `reshape(nely,nelx)` scrambles it into a plausible-but-wrong picture.
