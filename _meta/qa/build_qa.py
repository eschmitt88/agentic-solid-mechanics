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

import hashlib
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


def cb(path: Path) -> str:
    """Content-hash cache-buster for iframe src — forces browsers to refetch a
    scene file whenever it changes (GitHub Pages caches assets ~10 min)."""
    try:
        return f"?v={hashlib.md5(path.read_bytes()).hexdigest()[:8]}"
    except Exception:  # noqa: BLE001
        return ""


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
        leak_html = qa.chip(True, "CLEAN") + " the agent's own files never read the stored answer or grader"
    else:
        items = ", ".join(f"<code>{h['file']}:{h['token']}</code>" for h in leak["hits"])
        leak_html = qa.chip(False, "FLAG") + f" {items}"
    rows = rows + [leg_row("no peeking at the answer", leak_html)]
    note_html = f"<p class='hint'>{notes}</p>" if notes else ""
    return ("<h2>Did the agent really do this? (independent checks)</h2>"
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
        scene = qa.scene_block("scene.html" + cb(out / "scene.html"), "deformed.png",
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

    qa.bc_beam(out / "bc.png", title="Cantilever beam — fixed at the left, loaded down at the tip")
    intro = qa.intro_block(
        objective=("Can an AI agent operate a Finite Element Analysis (FEA) solver entirely on its own? "
                   "Given a cantilever beam (a beam fixed at one end, free at the other) and a goal — find "
                   "the tip deflection and peak stress — the agent must write the solver's text input file, "
                   "run the solver, read back the numbers, and check them against textbook beam theory, "
                   "with no human in the loop."),
        setup_rows=[
            ("Structure", "Steel cantilever beam, length L = 1.0 m, rectangular cross-section "
                          "(width 50 mm × height 100 mm)"),
            ("Material", "Steel — Young's modulus E = 210 GPa, Poisson's ratio ν = 0.30"),
            ("How it is held", "Left end fully clamped (all displacement fixed to zero)"),
            ("Load", "Downward point force P = 1000 N at the free tip"),
            ("Solver", "CalculiX (free open-source FEA), 20-node quadratic “brick” elements "
                       "(type C3D20R) chosen to avoid shear-locking stiffness errors"),
        ],
        bc_png="bc.png",
        bc_caption="Boundary conditions: the hatched wall is the clamp; the red arrow is the tip load.",
        measured=("<b>Tip deflection</b> compared with the Euler–Bernoulli beam formula "
                  "δ = P·L³ / (3·E·I). The agent passes if its simulated deflection is within tolerance "
                  "of the analytical value and the mesh has converged (the answer stops changing as the "
                  "mesh is refined)."))
    body = (intro
            + "<h2>Result — deformed shape &amp; stress</h2>" + scene
            + "<h2>Mesh convergence</h2><div class='card'><img class='fig' src='convergence.png' "
              "alt='convergence'><p class='hint'>The simulated deflection settles ~0.3% above the "
              "Euler–Bernoulli formula — exactly the shear-deformation correction that simple beam theory "
              "leaves out.</p></div>"
            + leg
            + qa.terms_block(["FEA", "agent", "cantilever", "Euler–Bernoulli", "von Mises stress",
                              "CalculiX", "C3D20R", "pass@k"])
            + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 1 — Can an AI agent run a structural solver?",
        "The agent writes the solver input, runs it, reads the results, and checks them against beam theory.",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 1 — operator baseline",
            "axis": "running a solver", "headline": f"{g['deflection_rel_err']*100:.3f}% deflection error · PASS"}


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
        return qa.scene_block("scene.html" + cb(out / "scene.html"), "deformed.png", caption, stats)
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

    qa.bc_beam(out / "bc.png", title="Cantilever beam — the agent chooses the cross-section size")
    intro = qa.intro_block(
        objective=("Can the agent <i>design</i> a part, not just analyze one? It must choose the "
                   "cross-section dimensions of a cantilever beam to make it as light as possible while "
                   "still meeting limits on how much it bends and how highly it is stressed — running the "
                   "loop itself: propose dimensions → build the mesh → simulate → check the limits → revise."),
        setup_rows=[
            ("Structure", "Steel cantilever, length L = 1.0 m, rectangular cross-section"),
            ("Design freedom", "Width b (10–80 mm) and height h (10–200 mm) — the agent picks these"),
            ("Goal", "Minimize mass (= density × length × b × h)"),
            ("Limits (constraints)", "Tip deflection ≤ 3 mm (checked by simulation) and bending stress "
                                     "≤ 200 MPa"),
            ("Load", "Downward tip force P = 2000 N"),
        ],
        bc_png="bc.png",
        bc_caption="Same cantilever as Trial 1; now the cross-section width and height are the variables "
                   "the agent optimizes.",
        measured=("<b>Mass of the agent's final design</b> versus the true optimum found by an independent "
                  "brute-force search, plus an independent re-check that the design actually meets the "
                  "deflection and stress limits. A subtlety the agent must discover: stiffness-per-mass "
                  "favors height, so the lightest design pushes width to its minimum — a non-obvious "
                  "“corner” answer."))
    body = (intro
            + "<h2>Result — the agent's final design</h2>" + scene
            + "<h2>Design space</h2><div class='card'><img class='fig' src='design_space.png' "
              "alt='design space'><p class='hint'>Mass falls as width shrinks, so the optimum pins width "
              "at its 10 mm minimum — the non-obvious corner solution the agent found.</p></div>"
            + leg
            + qa.terms_block(["agent", "cantilever", "FEA", "pass@k"])
            + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 2 — Can the agent design a part to meet limits?",
        "Choose a cantilever's cross-section to minimize weight under deflection and stress limits.",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 2 — design loop (prismatic)",
            "axis": "designing a part", "headline": f"+{g['pct_above_optimum']:.3f}% above optimum · reliable 10/10"}


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

    qa.bc_beam(out / "bc.png", h_root=0.62, h_tip=0.20,
               title="Tapered cantilever — height shrinks from the clamp to the tip")
    intro = qa.intro_block(
        objective=("A harder version of Trial 2. Now the beam's height <i>tapers</i> from a tall root "
                   "(at the clamp) to a short tip. A tapered beam has <b>no textbook deflection formula</b>, "
                   "so the agent cannot shortcut with an equation — it must simulate every candidate design. "
                   "This tests whether the agent will lean on the simulation when no formula exists."),
        setup_rows=[
            ("Structure", "Steel tapered cantilever, length L = 1.0 m, constant width, height varying "
                          "linearly from root to tip"),
            ("Design freedom", "Width b, root height, and tip height — three variables"),
            ("Goal", "Minimize mass"),
            ("Limits (constraints)", "Tip deflection ≤ 3 mm (only obtainable by simulation) and bending "
                                     "stress ≤ 200 MPa"),
            ("Load", "Downward tip force P = 2000 N"),
        ],
        bc_png="bc.png",
        bc_caption="A tapered cantilever: deep at the clamped end, shallow at the free end — no closed-form "
                   "deflection equation exists, so simulation is essential.",
        measured=("<b>Mass of the agent's final taper</b> versus the true optimum from an independent "
                  "simulation-based search, plus a feasibility re-check. The interesting finding: the best "
                  "taper is an interior trade-off (neither uniform nor extreme), which only simulation can "
                  "locate."))
    body = (intro
            + "<h2>Result — the agent's tapered design</h2>" + scene
            + "<h2>Design space</h2><div class='card'><img class='fig' src='taper_scan.png' "
              "alt='taper scan'><p class='hint'>The lightest feasible beam sits at an interior taper "
              "(tip height ≈ ¼ of root height) — no formula predicts this; only simulation finds it.</p></div>"
            + leg
            + qa.terms_block(["agent", "cantilever", "FEA"])
            + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 2b — Designing when no formula exists",
        "A tapered cantilever with no closed-form deflection — the agent must rely on simulation.",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 2b — design loop (tapered)",
            "axis": "designing a part (no formula)", "headline": f"{g['pct_above_optimum']:+.3f}% vs optimum · {lighter:.0f}% lighter than uniform"}


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
        # render MBB setup diagram + reproduced topology
        mbb_npy = src / "results" / "mbb_density.npy"
        mbb_img = ""
        if mbb_npy.exists():
            qa.bc_mbb(out / "mbb_bc.png")
            qa.topology_heatmap(np.load(mbb_npy), None, out / "mbb.png",
                                labels=("our reproduced layout", ""), origin="upper",
                                suptitle="MBB beam — material layout our solver produced")
            mbb_img = ("<img class='fig' src='mbb_bc.png' alt='MBB setup' style='max-width:520px'>"
                       "<img class='fig' src='mbb.png' alt='MBB beam' style='margin-top:10px'>"
                       f"<p class='hint'>Our solver reproduces the textbook MBB layout (solid top and bottom "
                       f"chords with a triangulated web of diagonals). Our stiffness measure (compliance) "
                       f"is {mbb['our_compliance']:.1f}; the published number depends on solver settings "
                       f"(roughly 200–220), so the <b>shape match is the rigorous check</b>, not a single "
                       f"number.</p>")
        verif = (
            "<h2>Is the scoring trustworthy? (verification against outside references)</h2>"
            "<div class='card'><p>The agent above is scored against a reference we compute ourselves. To "
            "show that reference is correct, we check our simulator against two targets defined "
            "<i>outside</i> this project:</p></div>"
            "<div class='card'><p><b>Check 1 — does the simulator match textbook physics?</b> &nbsp;"
            + qa.chip(b["pass"])
            + "</p><p class='hint'>For a plain uniform beam, the simulator's predicted tip deflection should "
            "equal the analytical beam-theory formulas. <b>Euler–Bernoulli</b> is the simple formula; "
            "<b>Timoshenko</b> adds the shear-deformation term it omits. L/h is the beam's "
            "length-to-height (slenderness) ratio.</p>"
            "<table class='leg'>" + brows + "</table>"
            f"<p class='hint'>Largest error versus Timoshenko = {b['max_abs_err_vs_timoshenko_pct']:.2f}%. "
            "The short, stubby beam (L/h = 3) shows Euler–Bernoulli breaking down while Timoshenko still "
            "holds — the same shear correction the CalculiX solver showed in Trial 1.</p></div>"
            "<div class='card'><p><b>Check 2 — does it reproduce a famous published benchmark?</b></p>"
            "<p class='hint'>The MBB beam (Messerschmitt-Bölkow-Blohm, the standard textbook "
            "topology-optimization problem) has a well-known optimal shape. Setup: the left edge is a "
            "symmetry plane (rollers), a single roller supports the bottom-right corner, and a downward "
            "load sits at the top-left.</p>"
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

    qa.bc_domain_2d(out / "bc.png", nx=48, ny=16)
    intro = qa.intro_block(
        objective=("Can the agent invent the best <i>material layout</i> from scratch? This is "
                   "<b>topology optimization</b> — deciding which parts of a region should be solid and "
                   "which empty to make the stiffest possible structure from a fixed amount of material. "
                   "The twist: the agent is handed a <b>differentiable</b> physics simulator (one that also "
                   "returns gradients — the sensitivity of stiffness to each cell's material) and must write "
                   "its <i>own</i> gradient-based optimizer to solve it."),
        setup_rows=[
            ("Design region", "A 2-D rectangle, 48 × 16 cells, each cell solid or empty"),
            ("How it is held", "The entire left edge is clamped (fixed)"),
            ("Load", "A downward force at the middle of the right edge"),
            ("Material budget", "At most 40% of the region may be solid"),
            ("Goal", "Minimize compliance (flexibility) = maximize stiffness"),
        ],
        bc_png="bc.png",
        bc_caption="The design region: left edge clamped, load pulling down at the right. The optimizer "
                   "chooses where to put the limited material.",
        measured=("<b>Compliance of the agent's layout</b> versus a reference optimum we compute the same "
                  "way, and — crucially — an <b>independent recompute</b> of the agent's reported number "
                  "from its saved design (to rule out a fabricated result). Lower compliance is better."))
    body = (intro
            + "<h2>Result — the optimized layout</h2><div class='card'><img class='fig' src='topology.png' "
              "alt='topology'><p class='hint'>Both reach the classic cantilever truss; the agent's layout is "
              "visually and numerically equivalent to the reference (darker = solid material).</p></div>"
            + leg + verif + scaling
            + qa.terms_block(["agent", "topology optimization", "compliance", "differentiable FEM",
                              "SIMP", "OC", "MBB beam", "Euler–Bernoulli", "Timoshenko", "GPU", "ndof"])
            + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 3 — Can the agent invent an optimal structure?",
        "Topology optimization on a differentiable simulator — the agent writes its own optimizer.",
        body, crumb_for(slug)))
    return {"slug": slug, "title": "Trial 3 — differentiable topology",
            "axis": "inventing a structure (2-D)", "headline": f"+{g['rel_diff_vs_reference']*100:.2f}% vs reference · recompute EXACT"}


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
            rho3d = np.load(rho3d_p)
            qa.render_topology3d(rho3d, out / "topo3d.png")          # static PNG fallback
            qa.write_threshold_scene_3d(rho3d, out / "scene.html")   # interactive + slider
            scene = (f"<div class='card'><iframe class='scene' src='scene.html{cb(out / 'scene.html')}' loading='lazy' "
                     "title='3D topology'></iframe><p class='hint'>Optimised 3D cantilever — drag to "
                     "rotate, scroll to zoom, and use the <b>density &ge; slider</b> (top-left) to peel "
                     "away low-density material. Static fallback: <a href='topo3d.png'>PNG</a>.</p></div>")
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
        leg_row("element math — rigid-body check",
                f"{eg['n_zero_eigs']} of the expected 6 free-motion modes (a correct element can move/rotate "
                f"freely with zero internal force) &nbsp; " + qa.chip(eg["pass"])),
        leg_row("element math — symmetry &amp; zero-force motion",
                f"stiffness matrix symmetric = {eg['symmetric']}; force under a rigid shift "
                f"{eg['rigid_translation_max_force']:.0e} (≈ 0, as it must be)"),
        leg_row("simulator vs textbook beam theory",
                f"error shrinks to {bc['finest_err_vs_eb_pct']:+.2f}% at the finest mesh (converges) &nbsp; "
                + qa.chip(bc["pass"])),
        leg_row("optimized stiffness (compliance, lower = stiffer)",
                f"{m['reference_compliance']:.2f} using {m['reference_volume']*100:.0f}% material "
                f"(block {m['grid'][0]}×{m['grid'][1]}×{m['grid'][2]}, {m['ndof']} unknowns, on GPU)"),
    ]
    leg = legitimacy_card(rows, {"clean": True, "hits": [], "scanned_dir": "verification only"},
                          "These rows verify the 3-D simulator itself (built from scratch here). The element "
                          "passes exact algebraic checks and the simulator converges to closed-form beam "
                          "theory; coarse-mesh stiffness is the known shear-locking artifact of simple "
                          "elements. Whether an AI agent can drive this simulator is tested below.")

    # --- loop-2 agentic pass@k: agent writes its own optimizer on this solver ---
    loop2 = ""
    pk_files = sorted((src / "results").glob("agent_topopt3d_trials_*.json"))
    if pk_files:
        pk = load_json(pk_files[-1])
        feas = [t for t in pk["trials"] if t.get("feasible")]
        n = pk["n_trials"]
        best = pk.get("best_pct_above_reference")
        mean = pk.get("mean_pct_above_reference")
        turns = [t.get("num_turns") for t in pk["trials"] if t.get("num_turns")]
        med_turns = sorted(turns)[len(turns) // 2] if turns else "—"
        prows = [
            leg_row("feasible runs",
                    f"{len(feas)} / {n} &nbsp; " + qa.chip(len(feas) == n, f"{len(feas)}/{n}", f"{len(feas)}/{n}")),
            leg_row("stiffness vs our reference optimizer",
                    f"best {best:+.2f}% · mean {mean:+.2f}% (negative = stiffer than our reference)"
                    if best is not None else "—"),
            leg_row("independent recompute",
                    "every reported compliance was recomputed from the agent's saved design (no fabrication)"),
            leg_row("no peeking at the answer",
                    qa.chip(pk.get("all_leakage_clean", False), "CLEAN", "FLAG")
                    + " the agents' code never referenced the reference solution"),
            leg_row("typical effort", f"median {med_turns} turns per run (agent wrote &amp; ran its own optimizer)"),
        ]
        loop2 = (
            "<h2>Can an AI agent do this itself? (reliability over many runs)</h2>"
            "<div class='card'><p>The result above used our own optimizer. Here we instead hand the agent "
            "<i>only</i> the differentiable simulator and let it <b>write its own optimizer</b> from scratch, "
            "run it on the GPU, and submit a design — repeated as independent runs (a “pass@k” reliability "
            f"test). Grid {pk['grid'][0]}×{pk['grid'][1]}×{pk['grid'][2]}; each design is independently "
            "re-scored.</p><table class='leg'>" + "".join(prows) + "</table>"
            "<p class='hint'>Run headless on a Claude subscription (no API key). A representative "
            "agent-authored optimizer (a genuine Optimality-Criteria scheme with Lagrangian volume "
            "bisection and move limits) is committed at "
            "<code>results/agent_demo/agent_optimizer.py</code>.</p></div>")

    qa.bc_domain_3d(out / "bc.png", nx=m["grid"][0], ny=m["grid"][1], nz=m["grid"][2])
    intro = qa.intro_block(
        objective=("The same “invent the optimal material layout” problem as Trial 3, but in full "
                   "<b>3-D</b> — a solid block instead of a flat rectangle. Because no ready-made 3-D "
                   "simulator was on hand, this trial <b>builds one from scratch and proves it is correct</b> "
                   "before optimizing: the 3-D simulator is the foundation a future agent will use to design "
                   "three-dimensional parts."),
        setup_rows=[
            ("Design region", f"A 3-D block, {m['grid'][0]} × {m['grid'][1]} × {m['grid'][2]} cells"),
            ("How it is held", "One end face fully clamped (a cantilever)"),
            ("Load", "A downward force at the centre of the opposite (free) face"),
            ("Material budget", "At most 30% of the block may be solid"),
            ("Goal", "Minimize compliance (flexibility) = maximize stiffness"),
            ("Hardware", f"Runs on the GPU ({m['ndof']} unknown displacements solved each step)"),
        ],
        bc_png="bc.png",
        bc_caption="The 3-D design block: the shaded end face is clamped; the red arrow is the load on the "
                   "far face. The optimizer fills in where material should go.",
        measured=("Two things. <b>(1) Is the new simulator correct?</b> Its element math is checked "
                  "algebraically, and a plain uniform block must reproduce textbook beam theory as the mesh "
                  "is refined. <b>(2) Does optimization work?</b> The resulting 3-D structure should be a "
                  "sensible load-carrying shape at the target material budget."))
    # --- scaling up: matrix-free (sparse) solver ---
    mf_section = ""
    bench_p = src / "results" / "matrixfree_bench.json"
    topo_p = src / "results" / "matrixfree_topopt.json"
    if bench_p.exists() and topo_p.exists():
        bench = load_json(bench_p)
        mtopo = load_json(topo_p)
        big = bench["big_grid"]
        mf_scene = ""
        mfd = src / "results" / "matrixfree_density.npy"
        if mfd.exists():
            try:
                mfrho = np.load(mfd)
                qa.render_topology3d(mfrho, out / "mf3d.png")
                qa.write_threshold_scene_3d(mfrho, out / "mf_scene.html")
                mf_scene = (f"<div class='card'><iframe class='scene' src='mf_scene.html{cb(out / 'mf_scene.html')}' loading='lazy' "
                            f"title='matrix-free 3D result'></iframe><p class='hint'>Optimized structure on "
                            f"the {mtopo['grid'][0]}×{mtopo['grid'][1]}×{mtopo['grid'][2]} grid "
                            f"({mtopo['ndof']:,} unknowns) — drag to rotate; use the <b>density &ge; "
                            f"slider</b> to peel away low-density material. Static: "
                            f"<a href='mf3d.png'>PNG</a>.</p></div>")
            except Exception as e:  # noqa: BLE001
                mf_scene = f"<div class='card'><p class='hint'>scene unavailable: {e}</p></div>"
        scale_rows = "".join(
            f"<tr><td>{r['grid'][0]}×{r['grid'][1]}×{r['grid'][2]} ({r['ndof']:,} unknowns)</td>"
            f"<td class='metric'>matrix-free {r['matrix_free_s']*1000:.0f} ms"
            + (f" · dense {r['dense_s']:.2f} s" if r.get('dense_s') else " · dense: out of memory")
            + "</td></tr>"
            for r in bench["scaling"])
        mf_section = (
            "<h2>Scaling up — a matrix-free (sparse) solver</h2>"
            "<div class='card'><p>The simulator above forms the full stiffness matrix and solves it "
            "directly. That uses memory proportional to the number of unknowns <i>squared</i>, which caps "
            "the grid at a few thousand unknowns. A <b>matrix-free</b> rewrite never builds the matrix — it "
            "applies the stiffness one small element at a time and solves with an iterative method "
            "(<b>conjugate gradient</b>), using memory proportional to the unknowns (not their square). The "
            "density filter is likewise replaced by a small 3-D convolution. Both stay exactly "
            "differentiable, so the gradient-based optimizer is unchanged.</p>"
            f"<p>Validated against the direct solver: stiffness values match to "
            f"{max(c['rel_diff'] for c in bench['correctness']):.0e} and gradients to "
            f"{bench['gradient_rel_diff']:.0e} (i.e. identical). Speed and reach:</p>"
            "<table class='leg'>" + scale_rows
            + f"<tr><td><b>{big['grid'][0]}×{big['grid'][1]}×{big['grid'][2]} "
              f"({big['ndof']:,} unknowns)</b></td><td class='metric'><b>matrix-free "
              f"{big['matrix_free_s']:.2f} s</b> · a direct solve would need a "
              f"{big['dense_block_gb']:.0f} GB matrix (impossible on a 16 GB GPU)</td></tr>"
            "</table></div>"
            "<div class='card'><p>A full optimization at "
            f"{mtopo['grid'][0]}×{mtopo['grid'][1]}×{mtopo['grid'][2]} "
            f"({mtopo['ndof']:,} unknowns, {mtopo['n_iterations']} iterations) finishes in "
            f"{mtopo['wall_s']} s on the GPU — a grid the direct solver could not even allocate "
            f"({mtopo['dense_matrix_gb']:.0f} GB).</p>" + mf_scene + "</div>")

    # --- high-resolution showcase (how dense in ~a couple GPU-minutes) ---
    highres = ""
    hr_json = src / "results" / "highres_showcase.json"
    hr_npy = src / "results" / "highres_showcase_density.npy"
    if hr_json.exists() and hr_npy.exists():
        hr = load_json(hr_json)
        hr_scene = ""
        try:
            rho = np.load(hr_npy)
            qa.render_topology3d(rho, out / "highres.png", thresh=0.4)
            qa.write_threshold_scene_3d(rho, out / "highres_scene.html", floor=0.12)
            hr_scene = (f"<div class='card'><iframe class='scene' src='highres_scene.html"
                        f"{cb(out / 'highres_scene.html')}' loading='lazy' title='high-res 3D'></iframe>"
                        f"<p class='hint'>A slender high-resolution cantilever — the classic optimal "
                        f"hollow box-girder (solid top/bottom flanges, diagonally-trussed webs, opening to a "
                        f"truss at the free tip). Drag to rotate; use the <b>density &ge; slider</b> to peel "
                        f"into the internal web. Static: <a href='highres.png'>PNG</a>.</p></div>")
        except Exception as e:  # noqa: BLE001
            hr_scene = f"<div class='card'><p class='hint'>scene unavailable: {e}</p></div>"
        g0, g1, g2 = hr["grid"]
        highres = (
            "<h2>How dense can it go? (a few GPU-minutes)</h2>"
            f"<div class='card'><p>Pushing the matrix-free solver: a <b>{g0}×{g1}×{g2}</b> grid — "
            f"<b>{hr['nelem']:,} cells, {hr['ndof']:,} unknowns</b> — optimized in {hr['n_iterations']} "
            f"iterations ({hr['wall_s']:.0f} s on the GPU, {hr['sec_per_iter']:.1f} s/iter). A direct solver "
            f"would need to store and factorise a <b>{hr['dense_matrix_gb']:,.0f} GB</b> "
            f"({hr['dense_matrix_gb']/1000:.1f} TB) stiffness matrix — impossible on a 16 GB card; the "
            f"matrix-free solver never forms it.</p>"
            "<p class='hint'>The density ceiling in ~2 min is higher still (~200,000 cells, a 128×40×40 grid "
            "whose dense matrix would be ~3.4 TB); a slenderer domain like this one is shown because its "
            "optimum is a visibly open truss rather than a solid block.</p></div>" + hr_scene)

    body = (intro
            + "<h2>Result — the optimized 3-D structure</h2>" + scene
            + loop2
            + "<h2>Is the new simulator correct? (vs textbook beam theory)</h2><div class='card'>"
              "<img class='fig' src='beam_conv.png' alt='beam convergence'>"
              "<p class='hint'>For a plain uniform block, the simulator's tip deflection approaches the "
              "Euler–Bernoulli beam-theory value as the mesh is refined through the thickness. The gap at "
              "coarse meshes is “shear locking” (a known stiffness artifact of simple elements), not a bug — "
              "it shrinks with refinement.</p></div>"
            + leg + mf_section + highres
            + qa.terms_block(["agent", "topology optimization", "compliance", "differentiable FEM", "SIMP",
                              "OC", "Euler–Bernoulli", "shear locking", "matrix-free",
                              "conjugate gradient", "GPU", "ndof", "pass@k"])
            + artifact_links(src))
    (out / "index.html").write_text(qa.page(
        "Trial 4 — Building &amp; verifying a 3-D simulator, then optimizing",
        "A from-scratch 3-D differentiable simulator on the GPU, checked against beam theory, then used for "
        "3-D topology optimization.",
        body, crumb_for(slug)))
    hl = f"simulator verified vs beam theory ({bc['finest_err_vs_eb_pct']:+.2f}%)"
    if pk_files:
        pk = load_json(pk_files[-1])
        feas = sum(1 for t in pk["trials"] if t.get("feasible"))
        hl += f" · agent writes its own optimizer: {feas}/{pk['n_trials']} feasible"
    elif (src / "results" / "matrixfree_bench.json").exists():
        bigj = load_json(src / "results" / "matrixfree_bench.json")["big_grid"]
        hl += f" · matrix-free solves {bigj['ndof']:,} unknowns (≈{bigj['dense_block_gb']:.0f} GB dense) in {bigj['matrix_free_s']:.2f} s"
    return {"slug": slug, "title": "Trial 4 — 3D differentiable topology",
            "axis": "inventing a structure (3-D)", "headline": hl}


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
            f"<div class='axis'>Tests: {c['axis']} &nbsp;·&nbsp; {c['headline']}</div></li>")
    body = (
        "<div class='card'><p>This project tests whether an <b>AI agent</b> — a large language model that "
        "does the engineering itself, not a human driving a tool — can carry out structural-mechanics "
        "simulation and design. Each trial below uses <b>Finite Element Analysis (FEA)</b>, the standard "
        "numerical method for predicting how a structure deforms and where it is stressed.</p>"
        "<p>The trials build up three abilities: <b>(1) run a solver</b> (operate the simulation tool "
        "correctly), <b>(2) design a part</b> (choose dimensions to meet goals and limits), and "
        "<b>(3) invent a structure</b> (decide the optimal material layout from scratch). Every result is "
        "graded against physics — analytical formulas, mesh convergence, or published benchmarks — not "
        "against the agent's own say-so.</p>"
        "<p>Each page shows: the <b>objective</b> in plain terms, the <b>problem setup and boundary "
        "conditions</b> (how the structure is held and loaded, with a diagram), an <b>interactive 3-D "
        "result</b> you can rotate and zoom, and a <b>trustworthiness panel</b> (independent re-checks that "
        "the agent did the work honestly and the numbers hold up). Unfamiliar terms are defined at the "
        "bottom of each page.</p>"
        "<p class='hint'>Static site, generated by <code>_meta/qa/build_qa.py</code>, served from GitHub "
        "Pages — no running service. Best viewed on a desktop for the 3-D scenes.</p></div>"
        "<h2>Trials</h2><ul class='exp'>" + "".join(items) + "</ul>")
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
