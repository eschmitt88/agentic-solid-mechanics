"""qa_lib.py — shared helpers for the static interactive QA site.

Run inside the `solidmech` micromamba env (needs ccx, ccx2paraview, pyvista,
meshio, matplotlib). Renders are headless via VTK's EGL offscreen window — no
X server, no Xvfb. See _meta/qa/README.md.

Two QA axes per experiment:
  * physics    — deformed + von-Mises 3D scenes, convergence / design-space plots
  * legitimacy — did the agent genuinely do the work (leakage, recompute, turns)
"""
from __future__ import annotations

import os

# Force offscreen EGL BEFORE pyvista imports. No DISPLAY -> vtkEGLRenderWindow.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.pop("DISPLAY", None)

import glob
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# pyvista / matplotlib imported lazily inside functions that need them so that
# pure-data helpers (legitimacy, templating) work even if GL is unavailable.

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# solver / format conversion
# --------------------------------------------------------------------------- #
def patch_deck_fields(inp_text: str) -> str:
    """Insert *NODE FILE (U) + *EL FILE (S) before *END STEP so the .frd carries
    a displacement + stress field (the committed decks only *NODE PRINT to .dat)."""
    fields = "*NODE FILE\nU\n*EL FILE\nS\n"
    if "*NODE FILE" in inp_text.upper():
        return inp_text
    # insert before the (last) *END STEP, case-insensitive
    m = list(re.finditer(r"(?im)^\*END STEP\s*$", inp_text))
    if not m:
        return inp_text.rstrip() + "\n" + fields + "*END STEP\n"
    last = m[-1]
    return inp_text[: last.start()] + fields + inp_text[last.start() :]


def solve_deck(inp_text: str, workdir: Path, job: str = "viz") -> Path:
    """Write a deck (fields patched in), run ccx, return the .frd path."""
    workdir.mkdir(parents=True, exist_ok=True)
    deck = workdir / f"{job}.inp"
    deck.write_text(patch_deck_fields(inp_text))
    r = subprocess.run(
        ["ccx", job], cwd=workdir, capture_output=True, text=True, timeout=600
    )
    frd = workdir / f"{job}.frd"
    if not frd.exists():
        raise RuntimeError(f"ccx produced no .frd for {job}:\n{r.stdout[-2000:]}\n{r.stderr[-1000:]}")
    return frd


def frd_to_vtu(frd: Path) -> Path:
    """Convert a CalculiX .frd to .vtu via ccx2paraview; return the .vtu path."""
    frd = Path(frd)
    subprocess.run(
        ["ccx2paraview", str(frd), "vtu"], capture_output=True, text=True, timeout=300
    )
    cands = sorted(frd.parent.glob(f"{frd.stem}*.vtu"))
    if not cands:
        raise RuntimeError(f"ccx2paraview produced no .vtu for {frd}")
    return cands[-1]


# --------------------------------------------------------------------------- #
# 3D rendering (deformed shape coloured by von Mises) — PNG + interactive HTML
# --------------------------------------------------------------------------- #
def render_deformed(
    vtu: Path,
    png_out: Path,
    html_out: Path,
    *,
    warp_frac: float = 0.15,
    title: str = "von Mises stress (Pa)",
    window=(1100, 460),
) -> dict:
    """Warp the mesh by displacement U (auto-scaled so peak deflection reads as
    ~warp_frac of the model diagonal) and colour by S_Mises. Writes a static PNG
    and a self-contained interactive scene (rotate/zoom). Returns field stats."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    g = pv.read(str(vtu))
    if "U" not in g.point_data:
        raise RuntimeError(f"{vtu} has no U field; arrays={g.array_names}")
    umag = float(np.linalg.norm(np.asarray(g["U"]), axis=1).max()) or 1.0
    diag = float(np.linalg.norm(np.ptp(np.asarray(g.bounds).reshape(3, 2), axis=1)))
    factor = (warp_frac * diag) / umag
    warped = g.warp_by_vector("U", factor=factor)

    scalars = "S_Mises" if "S_Mises" in g.point_data else None

    def _draw(pl):
        pl.add_mesh(g, color="#888888", opacity=0.10)  # undeformed ghost
        pl.add_mesh(
            warped,
            scalars=scalars,
            cmap="turbo",
            show_edges=True,
            edge_color="#222222",
            line_width=0.4,
            scalar_bar_args={"title": title, "color": "black", "n_labels": 5},
        )
        pl.set_background("white")
        pl.view_isometric()
        pl.camera.zoom(1.25)

    png_out.parent.mkdir(parents=True, exist_ok=True)
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    _draw(pl)
    pl.screenshot(str(png_out))
    pl.close()

    pl2 = pv.Plotter(off_screen=True, window_size=list(window))
    _draw(pl2)
    pl2.export_html(str(html_out))
    pl2.close()

    return {
        "warp_factor": round(factor, 2),
        "max_disp_mm": round(umag * 1e3, 4),
        "vm_min_pa": float(np.asarray(g["S_Mises"]).min()) if scalars else None,
        "vm_max_pa": float(np.asarray(g["S_Mises"]).max()) if scalars else None,
        "n_nodes": int(g.n_points),
        "n_cells": int(g.n_cells),
    }


# --------------------------------------------------------------------------- #
# matplotlib plots
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_convergence(sweep: list[dict], analytical_defl_m: float, png_out: Path):
    """Mesh-convergence: tip deflection vs element count, with the 3D-true band."""
    plt = _mpl()
    n = [s["n_elements"] for s in sweep if s.get("ran")]
    d = [s["tip_deflection_m"] * 1e3 for s in sweep if s.get("ran")]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(n, d, "o-", color="#1f6feb", lw=2, label="FE tip deflection")
    ax.axhline(analytical_defl_m * 1e3, ls="--", color="#888", label="Euler–Bernoulli")
    ax.set_xscale("log")
    ax.set_xlabel("elements (log)")
    ax.set_ylabel("tip deflection (mm)")
    ax.set_title("Mesh convergence — FE settles above EB (shear)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_out, dpi=130)
    plt.close(fig)


def plot_design_space_b(scan: list[dict], fe_opt: dict, agent: dict, png_out: Path):
    """Mass vs width b at the deflection limit; mark FE optimum + agent's design."""
    plt = _mpl()
    b = [s["b"] * 1e3 for s in scan]
    m = [s["mass_kg"] for s in scan]
    order = np.argsort(b)
    b = np.array(b)[order]
    m = np.array(m)[order]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(b, m, "-", color="#1f6feb", lw=2, label="min-mass section vs b (defl-limited)")
    ax.plot(fe_opt["b"] * 1e3, fe_opt["mass_kg"], "*", ms=16, color="#2da44e",
            label=f"FE optimum {fe_opt['mass_kg']:.3f} kg")
    if agent:
        ax.plot(agent["b"] * 1e3, agent["mass_kg"], "o", ms=9, color="#cf222e",
                label=f"agent {agent['mass_kg']:.3f} kg")
    ax.set_xlabel("width b (mm)")
    ax.set_ylabel("mass (kg)")
    ax.set_title("Design space — mass monotone in b → min-width corner")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_out, dpi=130)
    plt.close(fig)


def plot_taper_scan(scan: list[dict], opt: dict, png_out: Path):
    """Mass vs taper ratio (h_tip/h_root); interior optimum the formula misses."""
    plt = _mpl()
    tr = [s["taper_ratio"] for s in scan]
    m = [s["mass_kg"] for s in scan]
    order = np.argsort(tr)
    tr = np.array(tr)[order]
    m = np.array(m)[order]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(tr, m, "o-", color="#1f6feb", lw=2, label="FE min-mass vs taper")
    ax.plot(opt["taper_ratio"], opt["mass_kg"], "*", ms=16, color="#2da44e",
            label=f"optimum {opt['mass_kg']:.3f} kg @ ratio {opt['taper_ratio']:.2f}")
    ax.set_xlabel("taper ratio  h_tip / h_root")
    ax.set_ylabel("mass (kg)")
    ax.set_title("Tapered design space — interior optimum (no closed form)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_out, dpi=130)
    plt.close(fig)


def topology_heatmap(ref: np.ndarray, agent: np.ndarray | None, png_out: Path,
                     comps: tuple[float, float] | None = None):
    """Density field(s): reference vs agent side by side (black = solid)."""
    plt = _mpl()
    fields = [("reference", ref)] + ([("agent", agent)] if agent is not None else [])
    fig, axes = plt.subplots(1, len(fields), figsize=(5.6 * len(fields), 2.6))
    if len(fields) == 1:
        axes = [axes]
    for ax, (name, fld) in zip(axes, fields):
        ax.imshow(fld, cmap="gray_r", origin="lower", vmin=0, vmax=1, aspect="equal")
        c = ""
        if comps and name == "reference":
            c = f"  (C={comps[0]:.1f})"
        if comps and name == "agent":
            c = f"  (C={comps[1]:.1f})"
        ax.set_title(f"{name} density{c}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Topology optimisation — minimum compliance, vol ≤ 0.40", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# legitimacy: did the agent genuinely do the work?
# --------------------------------------------------------------------------- #
FORBIDDEN_DEFAULT = (
    "reference.json", "reference_density", "design.py", "fea.py",
    "cantilever.py", "reference_design.py", "topopt_reference.py",
)


def leakage_scan(agent_dir: Path, forbidden=FORBIDDEN_DEFAULT) -> dict:
    """Grep the agent's own authored files (.py/.md/.txt/.json/.sh) for references
    to the reference solution / grader internals. Evidence the search was honest.
    Returns {clean: bool, hits: [...]}. Absence of hits != proof, but a hit is a
    red flag worth surfacing."""
    agent_dir = Path(agent_dir)
    hits = []
    if agent_dir.exists():
        for f in agent_dir.rglob("*"):
            if not f.is_file() or f.suffix not in {".py", ".md", ".txt", ".json", ".sh", ".log"}:
                continue
            try:
                txt = f.read_text(errors="ignore")
            except Exception:
                continue
            for tok in forbidden:
                if tok in txt:
                    hits.append({"file": str(f.relative_to(agent_dir)), "token": tok})
    return {"clean": not hits, "hits": hits[:20], "scanned_dir": str(agent_dir)}


# --------------------------------------------------------------------------- #
# HTML templating — self-contained static pages
# --------------------------------------------------------------------------- #
CSS = """
:root{--fg:#1c2128;--mut:#636c76;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--bg:#fff;--card:#f6f8fa}
*{box-sizing:border-box}body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}
.wrap{max-width:1040px;margin:0 auto;padding:28px 22px 80px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:25px;margin:.2em 0 .1em}h2{font-size:18px;margin:1.7em 0 .5em;border-bottom:1px solid var(--bd);padding-bottom:.25em}
.sub{color:var(--mut);margin:0 0 1.4em}
.crumb{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0}
iframe.scene{width:100%;height:480px;border:1px solid var(--bd);border-radius:8px;background:#fff}
img.fig{max-width:100%;border:1px solid var(--bd);border-radius:8px;display:block}
.hint{color:var(--mut);font-size:12.5px;margin:.4em 0 0}
table.leg{border-collapse:collapse;width:100%;font-size:14px}
table.leg td{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top}
table.leg td:first-child{color:var(--mut);width:230px;white-space:nowrap}
.chip{display:inline-block;font-size:12px;font-weight:600;padding:1px 9px;border-radius:20px}
.chip.ok{background:#dafbe1;color:var(--ok)}.chip.bad{background:#ffebe9;color:var(--bad)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
code{background:#eaeef2;padding:1px 5px;border-radius:5px;font-size:12.5px}
.metric{font-variant-numeric:tabular-nums}
ul.exp{list-style:none;padding:0}ul.exp li{border:1px solid var(--bd);border-radius:10px;padding:14px 16px;margin:10px 0;background:var(--card)}
ul.exp .ttl{font-weight:600;font-size:16px}ul.exp .axis{color:var(--mut);font-size:13px}
details summary{cursor:pointer;color:var(--accent);font-size:13.5px}
pre{background:#0d1117;color:#c9d1d9;padding:12px;border-radius:8px;overflow:auto;font-size:12px;max-height:340px}
"""


def chip(ok: bool, label_ok="PASS", label_bad="FAIL") -> str:
    return f'<span class="chip {"ok" if ok else "bad"}">{label_ok if ok else label_bad}</span>'


def page(title: str, subtitle: str, body: str, crumb: str = "") -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><div class="wrap">
{crumb}
<h1>{html.escape(title)}</h1><p class="sub">{subtitle}</p>
{body}
</div></body></html>"""


def scene_block(html_name: str, png_name: str, caption: str, stats: dict | None) -> str:
    s = ""
    if stats:
        s = (f'<p class="hint">warp ×{stats["warp_factor"]} (peak deflection '
             f'{stats["max_disp_mm"]} mm) · σ<sub>vM</sub> '
             f'{stats["vm_min_pa"]/1e6:.2f}–{stats["vm_max_pa"]/1e6:.2f} MPa · '
             f'{stats["n_nodes"]} nodes / {stats["n_cells"]} elems</p>')
    return f"""<div class="card">
<iframe class="scene" src="{html_name}" loading="lazy" title="{html.escape(caption)}"></iframe>
<p class="hint">{caption} — drag to rotate, scroll to zoom. Static fallback:
<a href="{png_name}">PNG</a>.</p>{s}</div>"""
