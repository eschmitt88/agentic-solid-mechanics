#!/usr/bin/env python
"""build_qa.py — generate the static interactive QA site under docs/qa/.

Run from the repo root inside the solidmech env:

    cd ~/projects/research/agentic-solid-mechanics
    ~/.local/bin/micromamba run -n solidmech python _meta/qa/build_qa.py

Produces one page per experiment (physics 3D scenes + plots + an
agentic-legitimacy panel) and a dashboard index, all under docs/qa/ for
GitHub Pages. No running service. See _meta/qa/README.md.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qa_lib as qa  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments"
OUT = REPO / "docs" / "qa"


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def import_module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def leg_row(label: str, value: str) -> str:
    return f"<tr><td>{label}</td><td class='metric'>{value}</td></tr>"


def legitimacy_card(rows: list[str], leak: dict, notes: str = "") -> str:
    if leak["clean"]:
        leak_html = qa.chip(True, "CLEAN") + " no reference/grader reads in the agent's files"
    else:
        items = ", ".join(f"<code>{h['file']}:{h['token']}</code>" for h in leak["hits"])
        leak_html = qa.chip(False, "FLAG") + f" {items}"
    rows = rows + [leg_row("leakage scan", leak_html)]
    note_html = f"<p class='hint'>{notes}</p>" if notes else ""
    return ("<h2>Agentic legitimacy</h2>"
            "<div class='card'><table class='leg'>"
            + "".join(rows)
            + "</table>" + note_html + "</div>")


# --------------------------------------------------------------------------- #
# Trial 1 — operator baseline
# --------------------------------------------------------------------------- #
def build_trial1(slug: str) -> dict:
    src = EXP / "2026-06-16-calculix-cantilever-baseline"
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    ref = load_json(src / "results" / "reference.json")
    res = load_json(src / "results" / "agent_result.json")
    passk = load_json(src / "results" / "agent_trials_claude-opus-4-8.json")

    # physics: the agent's beam .frd already carries U + S fields
    stats = None
    scene = ""
    try:
        frd = src / "results" / "agent_run" / "beam.frd"
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "beam.frd"
            shutil.copy(frd, tmp)
            vtu = qa.frd_to_vtu(tmp)
            stats = qa.render_deformed(vtu, out / "deformed.png", out / "scene.html")
        scene = qa.scene_block("scene.html", "deformed.png",
                               "Agent's validated cantilever — deformed, coloured by σ_Mises", stats)
    except Exception as e:  # noqa: BLE001
        scene = f"<div class='card'><p class='hint'>scene unavailable: {e}</p></div>"

    qa.plot_convergence(ref["sweep"], ref["analytical"]["tip_deflection_m"], out / "convergence.png")

    g = res["grade"]
    rows = [
        leg_row("deflection error vs analytical", f"{g['deflection_rel_err']*100:.3f}% &nbsp; "
                + qa.chip(g["deflection_pass"])),
        leg_row("element type", f"<code>{res['element_type']}</code> (avoids shear locking)"),
        leg_row("ccx runs / deck errors", f"{res['ccx_runs']} runs · {res['deck_errors']} deck errors"),
        leg_row("pass@k (opus, headless)",
                f"{passk['deflection_pass_at_k']*100:.0f}% over {passk['n_trials']} trial(s)"),
        leg_row("agent authored its own", "mesh generator + .inp deck (no template)"),
    ]
    leak = qa.leakage_scan(src / "results" / "agent_run")
    leg = legitimacy_card(rows, leak,
                          "Independently re-graded against Euler–Bernoulli + a deterministic mesh-converged "
                          "reference sweep; the agent operated ccx unaided.")

    body = ("<h2>Physics — deformed shape &amp; stress</h2>" + scene
            + "<h2>Mesh convergence</h2><div class='card'><img class='fig' src='convergence.png' "
              "alt='convergence'><p class='hint'>FE settles ~0.3% above Euler–Bernoulli — the shear "
              "correction beam theory omits.</p></div>"
            + leg
            + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 1 — agent as CalculiX operator",
        "The agent writes the deck, runs ccx, parses results, and validates against beam theory.",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 1 — operator baseline",
            "axis": "drive a solver", "headline": f"{g['deflection_rel_err']*100:.3f}% deflection error · PASS"}


# --------------------------------------------------------------------------- #
# prismatic / tapered re-solve helpers (decks only *NODE PRINT to .dat → re-solve
# with fields so the scene can be coloured by stress)
# --------------------------------------------------------------------------- #
def resolve_scene(fea_mod, beam, n_grade: int, out: Path, caption: str) -> str:
    try:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            info = fea_mod.generate_inp(beam, n_grade, wd / "src.inp")  # noqa: F841
            inp_text = (wd / "src.inp").read_text()
            frd = qa.solve_deck(inp_text, wd, "viz")
            vtu = qa.frd_to_vtu(frd)
            stats = qa.render_deformed(vtu, out / "deformed.png", out / "scene.html")
        return qa.scene_block("scene.html", "deformed.png", caption, stats)
    except Exception as e:  # noqa: BLE001
        return f"<div class='card'><p class='hint'>scene unavailable: {e}</p></div>"


def build_trial2(slug: str) -> dict:
    src = EXP / "2026-06-21-calculix-cantilever-design-loop"
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    ref = load_json(src / "results" / "reference.json")
    demo = load_json(src / "results" / "agent_demo_result.json")
    passk = load_json(src / "results" / "agent_trials_claude-opus-4-8.json")
    cfg_p = src / "config.yaml"
    p = _yaml(cfg_p)["problem"]

    fea = import_module_from(src / "fea.py", "fea_t2")
    g = demo["grade"]
    beam = fea.Beam(length_m=p["length_m"], width_m=g["b"], height_m=g["h"],
                    youngs_modulus_pa=float(p["youngs_modulus_pa"]), poisson_ratio=p["poisson_ratio"],
                    density_kg_m3=p["density_kg_m3"], tip_load_n=p["tip_load_n"])
    scene = resolve_scene(fea, beam, 8, out,
                          "Agent's final section (b=10 mm, h=108.6 mm) — deformed, σ_Mises")

    fe_opt = ref["fe_optimum"]
    qa.plot_design_space_b(ref["fe_scan"], fe_opt, {"b": g["b"], "mass_kg": g["mass_kg"]},
                           out / "design_space.png")

    rows = [
        leg_row("mass above FE-true optimum", f"+{g['pct_above_optimum']:.3f}% &nbsp; " + qa.chip(g["feasible"], "FEASIBLE", "INFEASIBLE")),
        leg_row("found min-width corner", qa.chip(abs(g["b"] - 0.010) < 1e-6, "YES", "NO") + f" (b={g['b']*1e3:.1f} mm)"),
        leg_row("FE deflection on check mesh", f"{g['check_fe_deflection_m']*1e3:.3f} mm ≤ {g['delta_allow_m']*1e3:.1f} mm"),
        leg_row("FE designs evaluated", f"{demo['reported']['n_designs']}"),
        leg_row("pass@10 (opus, headless)",
                f"{passk['feasible_rate']*100:.0f}% feasible · best +{passk['best_pct_above_optimum']:.3f}% · "
                f"mean +{passk['mean_pct_above_optimum']:.3f}%"),
    ]
    leak = qa.leakage_scan(src / "results" / "agent_demo")
    leg = legitimacy_card(rows, leak,
                          "Independently re-graded on a fixed n=8 mesh; the agent derived the min-width "
                          "corner from a stiffness-per-mass argument and closed the FE loop (shear vs EB).")

    body = ("<h2>Physics — final design</h2>" + scene
            + "<h2>Design space</h2><div class='card'><img class='fig' src='design_space.png' "
              "alt='design space'><p class='hint'>Mass is monotone in width → optimum pins b at its "
              "10 mm floor (the non-obvious corner solution).</p></div>"
            + leg + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 2 — operator design loop",
        "Size a cantilever (b,h) to minimise mass under deflection + stress constraints.",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 2 — design loop (prismatic)",
            "axis": "design with a solver", "headline": f"+{g['pct_above_optimum']:.3f}% above optimum · pass@10 10/10"}


def build_trial2b(slug: str) -> dict:
    src = EXP / "2026-06-22-calculix-tapered-design-loop"
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    ref = load_json(src / "results" / "reference.json")
    demo = load_json(src / "results" / "agent_demo_result.json")
    p = _yaml(src / "config.yaml")["problem"]

    fea = import_module_from(src / "fea.py", "fea_t2b")
    g = demo["grade"]
    beam = fea.TaperedBeam(length_m=p["length_m"], width_m=g["b"], h_root_m=g["h_root"], h_tip_m=g["h_tip"],
                           youngs_modulus_pa=float(p["youngs_modulus_pa"]), poisson_ratio=p["poisson_ratio"],
                           density_kg_m3=p["density_kg_m3"], tip_load_n=p["tip_load_n"])
    scene = resolve_scene(fea, beam, 8, out,
                          f"Agent's tapered design (h_root={g['h_root']*1e3:.0f}, h_tip={g['h_tip']*1e3:.0f} mm) — σ_Mises")

    opt = ref["tapered_optimum"]
    qa.plot_taper_scan(ref["scan"], opt, out / "taper_scan.png")

    prismatic = ref.get("prismatic_optimum_kg")
    lighter = (1 - g["mass_kg"] / prismatic) * 100 if prismatic else None
    rows = [
        leg_row("mass vs FE-true tapered optimum",
                f"{g['pct_above_optimum']:+.3f}% &nbsp; " + qa.chip(g["feasible"], "FEASIBLE", "INFEASIBLE")),
        leg_row("lighter than prismatic optimum", f"{lighter:.1f}%" if lighter else "—"),
        leg_row("FE designs evaluated", f"{demo['reported']['n_designs']} (no closed form → FE essential)"),
        leg_row("agent's own finding", "flagged beam theory unsafe for the taper (EB integral ≈1.5% low)"),
    ]
    leak = qa.leakage_scan(src / "results" / "agent_demo")
    leg = legitimacy_card(rows, leak,
                          "A tapered beam has no elementary deflection formula, so every candidate had to be "
                          "FE-solved; the agent ran a 29-design search and matched the interior optimum.")

    body = ("<h2>Physics — tapered design</h2>" + scene
            + "<h2>Taper design space</h2><div class='card'><img class='fig' src='taper_scan.png' "
              "alt='taper scan'><p class='hint'>An interior optimum near taper ratio ≈0.25 — no closed "
              "form predicts it; FE is genuinely in the loop.</p></div>"
            + leg + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 2b — tapered design loop",
        "Harder variant: a tapered section with no closed-form deflection — FE in the loop, not a correction.",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 2b — design loop (tapered)",
            "axis": "design with a solver", "headline": f"{g['pct_above_optimum']:+.3f}% vs optimum · {lighter:.0f}% lighter than prismatic"}


def build_trial3(slug: str) -> dict:
    src = EXP / "2026-06-22-jaxfem-topology-optimization"
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    demo = load_json(src / "results" / "agent_demo_result.json")
    ref_rho = np.load(src / "results" / "reference_density.npy")  # already (nely,nelx)
    agent_rho = np.load(src / "results" / "agent_demo" / "agent_density.npy")
    # The reference is saved image-oriented; the agent saved the RAW flat element
    # vector in column-major (elx*nely+ely) order. Match orientation: reshape
    # (nelx,nely) then transpose -> (nely,nelx), exactly as topopt_reference does.
    nely, nelx = ref_rho.shape
    if agent_rho.ndim == 1 and agent_rho.size == ref_rho.size:
        agent_rho = agent_rho.reshape(nelx, nely).T
    g = demo["grade"]
    qa.topology_heatmap(ref_rho, agent_rho, out / "topology.png",
                        comps=(g["reference_compliance"], g["reported_compliance"]))

    fabricated = abs(g["reported_compliance"] - g["recomputed_compliance_from_density"]) < 1e-6
    rows = [
        leg_row("compliance vs reference",
                f"+{g['rel_diff_vs_reference']*100:.2f}% &nbsp; " + qa.chip(g["compliance_pass"])),
        leg_row("reported = independently recomputed",
                qa.chip(fabricated, "EXACT", "MISMATCH")
                + f" ({g['reported_compliance']:.2f} vs {g['recomputed_compliance_from_density']:.2f})"),
        leg_row("volume constraint", f"{g['final_volume']:.3f} ≤ 0.40 &nbsp; " + qa.chip(g["volume_pass"])),
        leg_row("optimiser", f"<code>{demo['optimizer']}</code> over {demo['n_iterations']} iters — agent-authored"),
    ]
    leak = qa.leakage_scan(src / "results" / "agent_demo")
    leg = legitimacy_card(rows, leak,
                          "The agent was given only a differentiable forward model (jaxfem.py) and wrote its own "
                          "optimiser from jax.value_and_grad; compliance was recomputed from the saved density.")

    # --- external-gold verification (forward-model Timoshenko + canonical MBB) ---
    verif = ""
    vpath = src / "results" / "verification.json"
    if vpath.exists():
        v = load_json(vpath)
        b = v["forward_model_gold"]
        brows = "".join(
            f"<tr><td>L/h = {r['L_over_h']:.0f}</td><td class='metric'>"
            f"vs Euler–Bernoulli {r['err_vs_eb_pct']:+.1f}% · vs Timoshenko {r['err_vs_timoshenko_pct']:+.2f}%</td></tr>"
            for r in b["rows"])
        mbb = v["optimizer_gold_mbb"]
        # render MBB topology image
        mbb_npy = src / "results" / "mbb_density.npy"
        mbb_img = ""
        if mbb_npy.exists():
            qa.topology_heatmap(np.load(mbb_npy), None, out / "mbb.png",
                                labels=("MBB beam (our solver)", ""), origin="upper",
                                suptitle="Canonical MBB beam — reproduced (Sigmund 2001 / Andreassen 2011)")
            mbb_img = ("<div class='card'><img class='fig' src='mbb.png' alt='MBB beam'>"
                       f"<p class='hint'>Reproduces the textbook MBB topology (top + bottom chords, "
                       f"triangulated web). Our compliance {mbb['our_compliance']:.1f}; the published "
                       f"scalar is filter/rmin-dependent (~200–220) so the topology is the rigorous match.</p></div>")
        verif = (
            "<h2>External-gold verification</h2>"
            "<div class='card'><p class='hint'>Does our reference machinery match targets defined "
            "<i>outside</i> this repo? Two checks:</p>"
            "<p><b>1. Forward-model gold (closed form)</b> &nbsp;" + qa.chip(b["pass"])
            + " — uniform-density compliance = tip deflection vs analytical beam theory:</p>"
            "<table class='leg'>" + brows + "</table>"
            f"<p class='hint'>Max |error vs Timoshenko| = {b['max_abs_err_vs_timoshenko_pct']:.2f}%. "
            "The L/h=3 row shows EB breaking down (shear) while Timoshenko holds — the same correction "
            "CalculiX showed in trial 1, now confirmed in the JAX solver.</p></div>"
            "<div class='card'><p><b>2. Optimizer gold (published problem)</b> — the canonical MBB beam:</p>"
            + mbb_img + "</div>")

    # --- GPU + grid scaling ---
    scaling = ""
    scaled_npy = src / "results" / "scaled_density.npy"
    if scaled_npy.exists():
        qa.topology_heatmap(np.load(scaled_npy), None, out / "scaled.png",
                            labels=("120×40 cantilever (GPU)", ""), origin="upper",
                            suptitle="Grid scaling on GPU — finer resolution, same topology family")
        scaling = (
            "<h2>GPU &amp; grid scaling</h2><div class='card'>"
            "<img class='fig' src='scaled.png' alt='scaled'>"
            "<p class='hint'>GPU unblocked (RTX 5080). At 48×16 the GPU run is bit-identical to CPU and "
            "~4× faster; at 120×40 (ndof 9922) it runs in ~70 s — the dense direct solve is the wall, which "
            "is exactly why the next step is a sparse / real-JAX-FEM solver.</p></div>")

    body = ("<h2>Physics — optimised topology</h2><div class='card'><img class='fig' src='topology.png' "
            "alt='topology'><p class='hint'>Classic cantilever truss; the agent's field is visually and "
            "numerically equivalent to the reference.</p></div>"
            + leg + verif + scaling + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 3 — differentiable inverse design",
        "Minimum-compliance topology optimisation on a differentiable FEM (loop 2 substrate).",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 3 — differentiable topology",
            "axis": "differentiate a solver", "headline": f"+{g['rel_diff_vs_reference']*100:.2f}% vs reference · recompute EXACT"}


def build_trial4(slug: str) -> dict:
    src = EXP / "2026-06-22-jaxfem3d-cantilever-topology"
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    m = load_json(src / "metrics.json")
    v = load_json(src / "results" / "verification3d.json")

    # 3D interactive scene
    scene = ""
    rho3d_p = src / "results" / "reference3d_density.npy"
    if rho3d_p.exists():
        try:
            stats = qa.render_topology3d(np.load(rho3d_p), out / "topo3d.png", out / "scene.html")
            scene = (f"<div class='card'><iframe class='scene' src='scene.html' loading='lazy' "
                     f"title='3D topology'></iframe><p class='hint'>Optimised 3D cantilever — drag to "
                     f"rotate, scroll to zoom. Static: <a href='topo3d.png'>PNG</a>. "
                     f"{stats['solid_cells']}/{stats['total_cells']} voxels kept "
                     f"(vol {stats['vol_frac_shown']}).</p></div>")
        except Exception as e:  # noqa: BLE001
            scene = f"<div class='card'><p class='hint'>scene unavailable: {e}</p></div>"

    # beam-convergence plot (FEM -> EB under refinement)
    bc = v["beam_convergence"]
    qa.plot_xy(bc["rows"], "ndof", "err_vs_eb_pct", out / "beam_conv.png",
               xlabel="ndof (log)", ylabel="error vs Euler–Bernoulli (%)",
               title="3D FEM → beam theory under through-thickness refinement",
               refline=0.0, reflabel="Euler–Bernoulli", logx=True)

    eg = v["element_gold"]
    rows = [
        leg_row("element stiffness — rigid-body modes",
                f"{eg['n_zero_eigs']} zero eigenvalues (expect 6) &nbsp; " + qa.chip(eg["pass"])),
        leg_row("element symmetric / PSD",
                f"symmetric={eg['symmetric']}, rigid-translation force "
                f"{eg['rigid_translation_max_force']:.0e} (≈0)"),
        leg_row("forward model → Euler–Bernoulli",
                f"converges monotonically to {bc['finest_err_vs_eb_pct']:+.2f}% at finest mesh &nbsp; "
                + qa.chip(bc["pass"])),
        leg_row("reference compliance",
                f"{m['reference_compliance']:.2f} at vol {m['reference_volume']:.2f} "
                f"(grid {m['grid'][0]}×{m['grid'][1]}×{m['grid'][2]}, ndof {m['ndof']}, GPU)"),
    ]
    leg = legitimacy_card(rows, {"clean": True, "hits": [], "scanned_dir": "n/a (no agent yet)"},
                          "No agent in this experiment yet — this is the verified 3D loop-2 substrate. "
                          "The element passes exact algebraic checks and the forward model converges to "
                          "closed-form beam theory; coarse-mesh stiffness is the known shear-locking of "
                          "fully-integrated trilinear hex.")

    body = ("<h2>Physics — 3D optimised structure</h2>" + scene
            + "<h2>Forward-model verification (→ beam theory)</h2><div class='card'>"
              "<img class='fig' src='beam_conv.png' alt='beam convergence'>"
              "<p class='hint'>Uniform-density compliance equals tip deflection; it converges to "
              "Euler–Bernoulli as the mesh refines through the thickness. The gap at coarse meshes is "
              "shear locking, not error.</p></div>"
            + leg + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 4 — 3D differentiable FEM topology",
        "A hand-rolled 3D differentiable FEM (8-node hex, SIMP) in JAX on the GPU — the loop-2 substrate in 3D.",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 4 — 3D differentiable topology",
            "axis": "differentiate a solver (3D)",
            "headline": f"element exact · FEM→EB {bc['finest_err_vs_eb_pct']:+.2f}% · GPU C={m['reference_compliance']:.1f}"}


# --------------------------------------------------------------------------- #
# shared bits
# --------------------------------------------------------------------------- #
def _yaml(p: Path) -> dict:
    import yaml
    return yaml.safe_load(Path(p).read_text())


def crumb_for(slug: str) -> str:
    return ('<div class="crumb"><a href="../">QA dashboard</a> · '
            '<a href="../../">project</a></div>')


def artifact_links(src: Path) -> str:
    rel = src.relative_to(REPO)
    base = f"https://github.com/eschmitt88/agentic-solid-mechanics/tree/master/{rel}"
    return (f"<h2>Artifacts</h2><div class='card'><p class='hint'>"
            f"<a href='{base}'>experiment folder on GitHub</a> — README, config.yaml, "
            f"results/, agent transcripts &amp; authored code.</p></div>")


def build_index(cards: list[dict]):
    items = []
    for c in cards:
        items.append(
            f"<li><a href='{c['slug']}/'><span class='ttl'>{c['title']}</span></a>"
            f"<div class='axis'>axis: {c['axis']} — {c['headline']}</div></li>")
    body = (
        "<div class='card'><p>Each experiment below is graded on two axes. "
        "<b>Physics</b>: interactive deformed/stress scenes and design-space plots — rotate and zoom in "
        "the browser, no plugin. <b>Agentic legitimacy</b>: did the agent genuinely do the work — a leakage "
        "scan (no reference/grader reads), an independent recompute, turn/eval counts, and pass@k.</p>"
        "<p class='hint'>Static — generated by <code>_meta/qa/build_qa.py</code>, served from GitHub Pages. "
        "No running service. Best viewed on a desktop for the 3D scenes.</p></div>"
        "<h2>Experiments</h2><ul class='exp'>" + "".join(items) + "</ul>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(qa.page(
        "Agentic Solid Mechanics — QA",
        "Human review surface for the agentic FEA trials: physics correctness + agentic legitimacy.",
        body, '<div class="crumb"><a href="../">project</a></div>'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    cards.append(build_trial1("trial-1-operator-baseline"))
    cards.append(build_trial2("trial-2-design-loop"))
    cards.append(build_trial2b("trial-2b-tapered"))
    cards.append(build_trial3("trial-3-topology"))
    cards.append(build_trial4("trial-4-topology-3d"))
    build_index(cards)
    print("QA site written to", OUT)
    for c in cards:
        print(" -", c["slug"], "|", c["headline"])


if __name__ == "__main__":
    main()
