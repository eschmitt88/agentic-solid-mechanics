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


def render_topology3d(rho3d: np.ndarray, png_out: Path, html_out: Path | None = None,
                      *, thresh: float = 0.5, window=(1000, 520)) -> dict:
    """Render a 3D density field (nelx,nely,nelz) as thresholded voxels coloured by
    density: static PNG (always) + optional pyvista interactive HTML. The
    user-facing interactive scene with a live threshold slider is written
    separately by write_threshold_scene_3d()."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    nx, ny, nz = rho3d.shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    grid.cell_data["rho"] = rho3d.flatten(order="F")  # VTK: x fastest
    solid = grid.threshold(thresh, scalars="rho")

    def _draw(pl):
        pl.add_mesh(solid, scalars="rho", cmap="viridis", show_edges=True,
                    edge_color="#333333", clim=[thresh, 1.0],
                    scalar_bar_args={"title": "density", "color": "black"})
        pl.add_mesh(grid.outline(), color="#888888")
        pl.set_background("white")
        pl.view_isometric()
        pl.camera.zoom(1.2)

    png_out.parent.mkdir(parents=True, exist_ok=True)
    pl = pv.Plotter(off_screen=True, window_size=list(window)); _draw(pl)
    pl.screenshot(str(png_out)); pl.close()
    if html_out is not None:
        pl2 = pv.Plotter(off_screen=True, window_size=list(window)); _draw(pl2)
        pl2.export_html(str(html_out)); pl2.close()
    return {"solid_cells": int(solid.n_cells), "total_cells": int(grid.n_cells),
            "vol_frac_shown": round(solid.n_cells / grid.n_cells, 3)}


def write_threshold_scene_3d(rho3d: np.ndarray, html_out: Path, *,
                             floor: float = 0.1, clim=(0.3, 1.0)) -> dict:
    """Self-contained interactive 3D scene with a LIVE density-threshold slider.

    Every cell with density >= floor is embedded once (sorted by density) and
    drawn as a cube via a Three.js InstancedMesh. Because instances are
    density-sorted, 'show density >= t' is just 'render the first K instances',
    so the slider re-thresholds instantly client-side — no server, no re-export.
    Rotate/zoom via OrbitControls. Static, hostable on GitHub Pages."""
    nx, ny, nz = rho3d.shape
    flat = rho3d.flatten(order="F")  # x fastest, matches VTK cell order
    cells = []
    for e, r in enumerate(flat):
        if r >= floor:
            ix = e % nx
            iy = (e // nx) % ny
            iz = e // (nx * ny)
            cells.append((round(ix + 0.5, 1), round(iy + 0.5, 1), round(iz + 0.5, 1), float(r)))
    cells.sort(key=lambda c: -c[3])  # density descending
    pos = [v for c in cells for v in c[:3]]
    rho = [round(c[3], 3) for c in cells]

    data = json.dumps({"dims": [nx, ny, nz], "pos": pos, "rho": rho,
                       "floor": floor, "clim": list(clim)})
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(_THRESHOLD_TEMPLATE.replace("__DATA__", data))
    return {"cells": len(cells), "total": int(nx * ny * nz)}


_THRESHOLD_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 html,body{margin:0;height:100%;font:13px -apple-system,Segoe UI,Roboto,sans-serif;background:#fff}
 #wrap{position:relative;width:100%;height:480px}
 #c{width:100%;height:480px;display:block}
 #ui{position:absolute;left:12px;top:12px;background:rgba(255,255,255,.92);border:1px solid #d0d7de;
     border-radius:8px;padding:9px 12px;user-select:none}
 #ui input{width:170px;vertical-align:middle}
 #n{color:#636c76;font-size:12px;margin-top:4px}
 #err{position:absolute;left:12px;bottom:12px;color:#cf222e;font-size:12px}
</style></head><body>
<div id="wrap"><canvas id="c"></canvas>
 <div id="ui">density &ge; <b id="v">0.50</b>
  <br><input id="t" type="range" min="0.1" max="1" step="0.01" value="0.5">
  <div id="n"></div></div>
 <div id="err"></div></div>
<script type="importmap">{"imports":{
 "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const dims=D.dims, POS=D.pos, RHO=D.rho, N=RHO.length, clim=D.clim;
const slider=document.getElementById('t'); slider.min=D.floor;
function viridis(t){const s=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
 t=Math.max(0,Math.min(1,t));const x=t*4,i=Math.min(3,Math.floor(x)),f=x-i,a=s[i],b=s[i+1];
 return [(a[0]+(b[0]-a[0])*f)/255,(a[1]+(b[1]-a[1])*f)/255,(a[2]+(b[2]-a[2])*f)/255];}
try{
 const canvas=document.getElementById('c');
 const W=()=>canvas.clientWidth||800, H=()=>480;
 const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
 renderer.setPixelRatio(window.devicePixelRatio); renderer.setSize(W(),H(),false);
 renderer.setClearColor(0xffffff,1);
 const scene=new THREE.Scene();
 const cam=new THREE.PerspectiveCamera(45,W()/H(),0.1,5000);
 const geo=new THREE.BoxGeometry(0.92,0.92,0.92);
 const mesh=new THREE.InstancedMesh(geo,new THREE.MeshLambertMaterial(),N);
 const o=new THREE.Object3D(), col=new THREE.Color();
 const cx=dims[0]/2, cy=dims[1]/2, cz=dims[2]/2;
 for(let i=0;i<N;i++){o.position.set(POS[3*i]-cx,POS[3*i+1]-cy,POS[3*i+2]-cz);o.updateMatrix();
  mesh.setMatrixAt(i,o.matrix);
  const c=viridis((RHO[i]-clim[0])/(clim[1]-clim[0]));col.setRGB(c[0],c[1],c[2]);mesh.setColorAt(i,col);}
 mesh.instanceMatrix.needsUpdate=true; mesh.instanceColor.needsUpdate=true; scene.add(mesh);
 scene.add(new THREE.AmbientLight(0xffffff,0.75));
 const dl=new THREE.DirectionalLight(0xffffff,0.55); dl.position.set(1,1.2,1.6); scene.add(dl);
 const box=new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(dims[0],dims[1],dims[2])),
  new THREE.LineBasicMaterial({color:0xbbbbbb})); scene.add(box);
 const R=Math.max(dims[0],dims[1],dims[2]);
 cam.position.set(dims[0]*0.9,dims[1]*1.1+R*0.5,R*1.7);
 const ctr=new OrbitControls(cam,renderer.domElement); ctr.target.set(0,0,0); ctr.update();
 function K(t){let lo=0,hi=N;while(lo<hi){const m=(lo+hi)>>1;if(RHO[m]>=t)lo=m+1;else hi=m;}return lo;}
 function apply(){const t=+slider.value;document.getElementById('v').textContent=t.toFixed(2);
  const k=K(t);mesh.count=k;document.getElementById('n').textContent=k+' of '+N+' cells shown';}
 slider.addEventListener('input',apply); apply();
 (function loop(){requestAnimationFrame(loop);ctr.update();renderer.render(scene,cam);})();
 window.addEventListener('resize',()=>{renderer.setSize(W(),H(),false);cam.aspect=W()/H();cam.updateProjectionMatrix();});
}catch(e){document.getElementById('err').textContent='3D view failed to load ('+e.message+'). See the PNG fallback.';}
</script></body></html>"""


def plot_xy(rows: list[dict], xkey: str, ykey: str, png_out: Path, *,
            xlabel: str, ylabel: str, title: str, refline: float | None = None,
            reflabel: str = "", logx: bool = False):
    """Generic line plot for convergence-style tables."""
    plt = _mpl()
    x = [r[xkey] for r in rows]
    y = [r[ykey] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(x, y, "o-", color="#1f6feb", lw=2)
    if refline is not None:
        ax.axhline(refline, ls="--", color="#888", label=reflabel)
        ax.legend(fontsize=8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png_out, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# boundary-condition schematics (plain, labelled diagrams for newcomers)
# --------------------------------------------------------------------------- #
def _clamp_wall(ax, x, y0, y1, width=0.06, side="left"):
    """Draw a hatched fixed-support wall (the clamp)."""
    import matplotlib.patches as mp
    dx = -width if side == "left" else width
    ax.add_patch(mp.Rectangle((min(x, x + dx), y0), abs(dx), y1 - y0,
                              hatch="////", facecolor="#dde3ea", edgecolor="#333", lw=1.2))
    ax.plot([x, x], [y0, y1], color="#333", lw=2)


def _load_arrow(ax, x, y, label, dy=0.32, color="#cf222e"):
    ax.annotate("", xy=(x, y - dy / 2), xytext=(x, y + dy / 2),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=color))
    ax.text(x, y + dy / 2 + 0.04, label, ha="center", va="bottom",
            color=color, fontsize=9, fontweight="bold")


def bc_beam(png_out: Path, *, h_root=0.5, h_tip=0.5, title="", load_label="load P (downward)"):
    """Cantilever side-view: clamped left wall, beam, downward tip load."""
    import matplotlib.patches as mp
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    L = 1.0
    yc = 0.0
    poly = [(0, yc - h_root / 2), (L, yc - h_tip / 2), (L, yc + h_tip / 2), (0, yc + h_root / 2)]
    ax.add_patch(mp.Polygon(poly, closed=True, facecolor="#cfe2ff", edgecolor="#1f6feb", lw=1.5))
    _clamp_wall(ax, 0, yc - h_root / 2 - 0.06, yc + h_root / 2 + 0.06)
    _load_arrow(ax, L, yc + h_tip / 2, load_label)
    ax.annotate("", xy=(0, -0.62), xytext=(L, -0.62), arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(L / 2, -0.7, "length L", ha="center", va="top", fontsize=9, color="#555")
    ax.text(-0.07, yc, "fixed end\n(clamped:\nall motion = 0)", ha="right", va="center",
            fontsize=8.5, color="#333")
    ax.text(L + 0.02, yc - h_tip / 2 - 0.05, "free end", ha="left", va="top", fontsize=8.5, color="#333")
    ax.set_xlim(-0.5, 1.35); ax.set_ylim(-0.95, 0.7); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(png_out, dpi=130); plt.close(fig)


def bc_domain_2d(png_out: Path, *, nx=48, ny=16, title="2-D design domain"):
    """Rectangular design domain: clamped left edge, load at mid-right edge."""
    import matplotlib.patches as mp
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    w, h = float(nx), float(ny)
    ax.add_patch(mp.Rectangle((0, 0), w, h, facecolor="#f0f3f7", edgecolor="#1f6feb", lw=1.5, ls="--"))
    _clamp_wall(ax, 0, -1, h + 1, width=2.0)
    _load_arrow(ax, w, h / 2, "load (downward)", dy=h * 0.45)
    ax.text(w / 2, h / 2, "material is placed\nhere by optimization", ha="center", va="center",
            fontsize=8.5, color="#777")
    ax.text(-2.4, h / 2, "left edge\nclamped", ha="right", va="center", fontsize=8.5, color="#333")
    ax.text(w / 2, -2.0, f"grid: {nx} × {ny} cells", ha="center", va="top", fontsize=8.5, color="#555")
    ax.set_xlim(-9, w + 7); ax.set_ylim(-4, h + 3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(png_out, dpi=130); plt.close(fig)


def bc_mbb(png_out: Path, *, nx=60, ny=20):
    """MBB beam: symmetry rollers on the left edge, vertical roller bottom-right,
    downward load at the top-left corner."""
    import matplotlib.patches as mp
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    w, h = float(nx), float(ny)
    ax.add_patch(mp.Rectangle((0, 0), w, h, facecolor="#f0f3f7", edgecolor="#1f6feb", lw=1.5, ls="--"))
    # left-edge symmetry rollers (x fixed): small circles
    for yy in np.linspace(2, h - 2, 5):
        ax.add_patch(mp.Circle((-1.2, yy), 0.9, facecolor="#dde3ea", edgecolor="#333", lw=1))
    ax.text(-3.0, h / 2, "symmetry\n(left edge:\nx fixed)", ha="right", va="center", fontsize=8.5, color="#333")
    # bottom-right vertical roller (y fixed)
    ax.add_patch(mp.Polygon([(w, 0), (w - 2, -2.6), (w + 2, -2.6)], closed=True,
                            facecolor="#dde3ea", edgecolor="#333", lw=1))
    ax.add_patch(mp.Circle((w - 1, -3.4), 0.8, facecolor="#dde3ea", edgecolor="#333", lw=1))
    ax.add_patch(mp.Circle((w + 1, -3.4), 0.8, facecolor="#dde3ea", edgecolor="#333", lw=1))
    ax.text(w + 3, -2.0, "roller\n(y fixed)", ha="left", va="center", fontsize=8.5, color="#333")
    # downward load at top-left, label placed to the left so it clears the title
    ax.annotate("", xy=(0, h - 1), xytext=(0, h + 6), arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#cf222e"))
    ax.text(-1.5, h + 5, "load\n(downward)", ha="right", va="center", color="#cf222e",
            fontsize=9, fontweight="bold")
    ax.set_xlim(-13, w + 9); ax.set_ylim(-6, h + 10); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("MBB beam — standard textbook benchmark setup", fontsize=10, pad=14)
    fig.tight_layout(); fig.savefig(png_out, dpi=130); plt.close(fig)


def bc_domain_3d(png_out: Path, *, nx=24, ny=8, nz=8):
    """3-D block: one end face clamped, downward load at centre of the far face."""
    plt = _mpl()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(6.6, 3.4))
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = float(nx), float(ny), float(nz)
    # box edges
    pts = np.array([[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0],
                    [0, 0, Z], [X, 0, Z], [X, Y, Z], [0, Y, Z]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot(*zip(pts[a], pts[b]), color="#1f6feb", lw=1.2)
    # clamped face x=0 (shaded)
    face = [pts[0], pts[3], pts[7], pts[4]]
    ax.add_collection3d(Poly3DCollection([face], facecolor="#dde3ea", edgecolor="#333", alpha=0.8))
    ax.text(0, Y / 2, Z + 2, "this face clamped\n(all motion = 0)", color="#333", fontsize=8)
    # load arrow at far-face centre, downward (-y)
    ax.quiver(X, Y / 2, Z / 2, 0, -Y * 0.7, 0, color="#cf222e", lw=2.5, arrow_length_ratio=0.3)
    ax.text(X, Y * 0.1, Z / 2, "load\n(downward)", color="#cf222e", fontsize=8, fontweight="bold")
    ax.set_xlabel("x (length)"); ax.set_ylabel("y (height)"); ax.set_zlabel("z (width)")
    ax.set_title(f"3-D design block: {nx} × {ny} × {nz} cells", fontsize=10)
    ax.set_box_aspect((X, Y, Z)); ax.view_init(elev=22, azim=-60)
    fig.tight_layout(); fig.savefig(png_out, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# plain-language intro + glossary
# --------------------------------------------------------------------------- #
GLOSSARY = {
    "FEA": "Finite Element Analysis — a numerical method that splits a structure into many small "
           "“elements” and solves for how it deforms and where it is stressed under load.",
    "agent": "Here, an AI agent: the large language model does the engineering work itself — writing "
             "the solver input, running it, reading the results, and deciding the next step — instead of "
             "a human driving the tool.",
    "boundary conditions": "How the structure is held and loaded: which parts are fixed in place and "
                           "where forces are applied.",
    "cantilever": "A beam fixed rigidly at one end and free at the other (like a diving board).",
    "Euler–Bernoulli": "Classic beam theory giving a closed-form formula for deflection; it ignores "
                       "shear deformation, so it is slightly off for short, stubby beams.",
    "Timoshenko": "Beam theory that adds the shear-deformation term Euler–Bernoulli omits; accurate "
                  "for stubby beams too.",
    "von Mises stress": "A single scalar combining the stress components, compared against a material’s "
                        "yield strength to judge failure.",
    "compliance": "A measure of flexibility (force × displacement at the load). Lower compliance = "
                  "stiffer structure, so minimizing compliance maximizes stiffness.",
    "topology optimization": "Deciding the optimal layout of material within a region — which parts "
                             "should be solid and which empty — for a given amount of material.",
    "SIMP": "Solid Isotropic Material with Penalization — the standard topology-optimization scheme "
            "that treats each cell’s density as a design variable and pushes intermediate “grey” "
            "values toward fully solid or fully empty.",
    "OC": "Optimality Criteria — a classic, fast update rule for topology optimization.",
    "differentiable FEM": "A finite-element solver written so you can compute exact gradients of the "
                          "result with respect to the design (via automatic differentiation) — which is "
                          "what lets a gradient-based optimizer improve the design.",
    "CalculiX": "A free, open-source FEA solver that reads Abaqus-style plain-text input files (“decks”).",
    "C3D20R": "A 20-node quadratic hexahedral (“brick”) element with reduced integration — chosen "
              "because simpler elements suffer “shear locking” (artificial over-stiffness in bending).",
    "MBB beam": "Messerschmitt-Bölkow-Blohm beam — the canonical textbook benchmark problem in "
                "topology optimization.",
    "ndof": "Number of degrees of freedom — the count of unknown displacements the solver computes "
            "(roughly, 2 or 3 per mesh node in 2-D / 3-D).",
    "pass@k": "Run the agent k independent times; report how often it succeeds — a reliability measure.",
    "GPU": "Graphics Processing Unit — massively parallel hardware that accelerates the linear algebra "
           "in the solver.",
    "shear locking": "A numerical artifact where simple elements act too stiff in bending; it shrinks "
                     "as the mesh is refined.",
    "conjugate gradient": "An iterative method that solves a large system of equations using only "
                          "matrix-times-vector products — no need to store or factorise the full matrix.",
    "matrix-free": "Solving without ever building the big matrix: its action on a vector is computed on "
                   "the fly, so memory grows with the number of unknowns rather than its square.",
}


def terms_block(keys: list[str]) -> str:
    rows = "".join(f"<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(GLOSSARY[k])}</td></tr>"
                   for k in keys if k in GLOSSARY)
    if not rows:
        return ""
    return ("<details><summary>Terms used on this page</summary>"
            "<div class='card'><table class='leg'>" + rows + "</table></div></details>")


def intro_block(objective: str, setup_rows: list[tuple[str, str]], bc_png: str,
                bc_caption: str, measured: str) -> str:
    rows = "".join(f"<tr><td>{html.escape(l)}</td><td>{v}</td></tr>" for l, v in setup_rows)
    return (
        "<h2>Objective</h2><div class='card'><p>" + objective + "</p></div>"
        "<h2>The problem &amp; how it is held</h2><div class='card'>"
        f"<img class='fig' src='{bc_png}' alt='boundary conditions' style='max-width:560px'>"
        f"<p class='hint'>{bc_caption}</p>"
        "<table class='leg' style='margin-top:10px'>" + rows + "</table></div>"
        "<h2>What is measured</h2><div class='card'><p>" + measured + "</p></div>")


def topology_heatmap(ref: np.ndarray, agent: np.ndarray | None, png_out: Path,
                     comps: tuple[float, float] | None = None,
                     labels: tuple[str, str] = ("reference", "agent"),
                     suptitle: str = "Topology optimisation — minimum compliance, vol ≤ 0.40",
                     origin: str = "lower"):
    """Density field(s): reference vs agent side by side (black = solid)."""
    plt = _mpl()
    fields = [(labels[0], ref)] + ([(labels[1], agent)] if agent is not None else [])
    fig, axes = plt.subplots(1, len(fields), figsize=(5.6 * len(fields), 2.6))
    if len(fields) == 1:
        axes = [axes]
    for i, (ax, (name, fld)) in enumerate(zip(axes, fields)):
        ax.imshow(fld, cmap="gray_r", origin=origin, vmin=0, vmax=1, aspect="equal")
        c = f"  (C={comps[i]:.1f})" if comps else ""
        ax.set_title(f"{name} density{c}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(suptitle, fontsize=11)
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
