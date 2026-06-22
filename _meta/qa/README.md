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

## Adding an experiment

Add a `build_trialN(slug)` to `build_qa.py` returning
`{slug, title, axis, headline}`, append it in `main()`. Reuse `qa_lib`
helpers; degrade gracefully (a failed scene still emits the page + legitimacy
panel).

## Gotchas (learned the hard way)

- NumPy 2.x removed `ndarray.ptp()` — use `np.ptp(arr)`.
- pyvista colours need full 6-digit hex (`#888888`), not `#888` or 8-digit rgba.
- The reference density is saved image-oriented (`reshape(nelx,nely).T`); the
  agent saved the **raw flat element vector** (column-major `elx*nely+ely`).
  Match orientation with `reshape(nelx,nely).T` before imshow — a naive
  `reshape(nely,nelx)` scrambles it into a plausible-but-wrong picture.
