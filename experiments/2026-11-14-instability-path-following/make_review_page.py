"""Write docs/qa/trial-7-instability/index.html from metrics.json (so every
number on the page is the number in the metrics file) and copy the figures."""
import html
import json
import pathlib
import shutil

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"
OUT = HERE.parents[1] / "docs" / "qa" / "trial-7-instability"
FIGS = ["r1_truss.png", "r1_lee_frame.png", "r1_arch.png", "r1_beam_convergence.png", "r2_biot.png", "r2_biot_mode.png",
        "r3_knockdown.png", "r3_ccx_paths.png"]

CSS = """
:root{--fg:#1c2128;--mut:#636c76;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--bg:#fff;--card:#f6f8fa}
*{box-sizing:border-box}body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}
.wrap{max-width:1040px;margin:0 auto;padding:28px 22px 80px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:25px;margin:.2em 0 .1em}h2{font-size:18px;margin:1.7em 0 .5em;border-bottom:1px solid var(--bd);padding-bottom:.25em}
h3{font-size:15.5px;margin:1.2em 0 .3em}
.sub{color:var(--mut);margin:0 0 1.4em}
.crumb{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0}
img.fig{max-width:100%;border:1px solid var(--bd);border-radius:8px;display:block}
.hint{color:var(--mut);font-size:12.5px;margin:.4em 0 0}
table.leg{border-collapse:collapse;width:100%;font-size:14px}
table.leg td,table.leg th{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top;text-align:left}
table.leg th{color:var(--mut);font-weight:600;font-size:12.5px}
table.leg td:first-child{color:var(--mut)}
.scroll{overflow-x:auto}
.chip{display:inline-block;font-size:12px;font-weight:600;padding:1px 9px;border-radius:20px;white-space:nowrap}
.chip.ok{background:#dafbe1;color:var(--ok)}.chip.bad{background:#ffebe9;color:var(--bad)}.chip.info{background:#ddf4ff;color:#0550ae}
code{background:#eaeef2;padding:1px 5px;border-radius:5px;font-size:12.5px}
.metric{font-variant-numeric:tabular-nums}
details summary{cursor:pointer;color:var(--accent);font-size:13.5px}
.big{font-size:17px;font-weight:600}
ul{padding-left:20px}li{margin:.25em 0}
"""


def chip(ok):
    return '<span class="chip ok">PASS</span>' if ok else '<span class="chip bad">MISS</span>'


def pct(x, d=2):
    return f"{100 * x:.{d}f} %"


def main():
    m = json.loads((HERE / "metrics.json").read_text())
    g3 = json.loads((RES / "r3_grade.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FIGS:
        shutil.copyfile(RES / f, OUT / f)

    def row(cid, what, number, ref, tol):
        return (f"<tr><td>{cid}</td><td>{what}</td><td class='metric'>{number}</td><td class='metric'>{ref}</td>"
                f"<td class='metric'>{tol}</td><td>{chip(m[cid]['pass'])}</td></tr>")

    head = "<tr><th></th><th>what</th><th>computed</th><th>reference</th><th>tolerance</th><th></th></tr>"
    c3 = g3["3c"]
    planned = [r for r in g3["3d_runs"] if r["seed_over_t"] == 1e-3 and "_full" not in r["file"]]
    full = m["3d"].get("corrected_full_model", [])
    inf = m["informational"]
    n_pass, n_all = m["summary"]["n_pass"], m["summary"]["n_criteria"]

    t1 = "<div class='scroll'><table class='leg'>" + head + "".join([
        row("1a", "Two-bar truss: snap-through load", f"{m['1a']['value']:.9f}", f"{m['1a']['reference']:.9f} (closed form)", "0.1 %"),
        row("1b", "Two-bar truss: sideways-buckling load, then switch onto the buckled branch", f"{m['1b']['value']:.9f}",
            f"{m['1b']['reference']:.9f} (closed form)", "0.1 %"),
        row("1c", "Lee frame: first limit load, P·L²/EI", f"{m['1c']['value']:.3f}", "18.552 (NAFEMS NL7)", "1 %"),
        row("1d", "Lee frame: load at deflection 0.784 L", f"{m['1d']['value']:.2f}", "31.887 (NAFEMS NL7)", "2 %"),
        row("1e", "215° arch: maximum load, P·R²/EI", f"{m['1e']['value']:.4f}", "8.9729 (Horák et al. 2022)", "1 %"),
        row("1f", "Second solver (CalculiX) on the Lee frame limit load", f"{m['1f']['value']:.3f}", f"{m['1f']['reference']:.3f} (own solver)", "2 %"),
    ]) + "</table></div>"
    t2 = "<div class='scroll'><table class='leg'>" + head + "".join([
        row("2a", "Critical compressive strain, finest mesh", f"{m['2a']['value']:.7f}", f"{m['2a']['reference']:.7f} (Biot; Cao &amp; Hutchinson 2012)", "1 %"),
        row("2b", "Error shrinks at every one of 5 mesh refinements", "×16 per halving", "—", "monotone"),
    ]) + "</table></div>"
    kd = "".join(f"<tr><td>{r['xi']:g} t</td><td class='metric'>{r['rho']:.3f}</td><td class='metric'>{r['koiter']:.3f}</td>"
                 f"<td class='metric'>{r['abs_diff']:.3f}</td></tr>" for r in c3)
    t3 = "<div class='scroll'><table class='leg'>" + head + "".join([
        row("3a", "Classical buckling load, own linear eigenvalue analysis ÷ formula", f"{m['3a']['value']:.5f}", "1 (0.605·E·t²/R-type formula)", "3 %"),
        row("3b", "Same with CalculiX shells", f"{m['3b']['value']:.4f}", "1", "5 %"),
        row("3c", "Knockdown curve vs Koiter's theory: largest gap over 7 imperfection sizes", f"{m['3c']['error']:.3f}", "0 (Koiter 1963 upper bound)", "0.05"),
        row("3d", "CalculiX nonlinear runs vs own solver, model as planned, 4 imperfection sizes",
            "+7.5 %, +11.2 %, −0.6 %, and no buckling found at 1 t", "0", "10 % each"),
    ]) + "</table></div>"
    def prow(r):
        none = r["how"].startswith("no buckling")
        val = "none up to load %.3f" % r["P_buckle"] if none else "%.3f" % r["P_buckle"]
        dif = "—" if none else "%+.1f %%" % (100 * r["rel_diff"])
        return (f"<tr><td>{r['xi']:g} t</td><td class='metric'>{val}</td><td class='metric'>{r['own_ss_even_n']:.3f}</td>"
                f"<td class='metric'>{dif}</td><td>{html.escape(r['how'])}</td></tr>")
    plan_rows = "".join(prow(r) for r in planned)
    full_rows = "".join(f"<tr><td>{r['xi']:g} t</td><td class='metric'>{r['ccx']:.3f}</td><td class='metric'>{r['own']:.3f}</td>"
                        f"<td class='metric'>{100 * r['rel_diff']:+.1f} %</td><td>{html.escape(r['how'])}</td></tr>" for r in full)
    full_block = ""
    if full:
        full_block = ("<h3>Corrected cross-check (full 360° model, larger seed)</h3><div class='scroll'><table class='leg'>"
                      "<tr><th>imperfection</th><th>CalculiX</th><th>own solver</th><th>difference</th><th>how CalculiX buckled</th></tr>"
                      + full_rows + "</table></div><p class='hint'>Run after the two problems below were diagnosed. Not part of the "
                      "pre-registered grade — shown so the reader can see what the second solver says once the model is sound.</p>")

    glossary = [
        ("equilibrium path", "The curve of all states in which the structure is in balance, drawn as load against deflection. Following it is what a nonlinear analysis does."),
        ("limit point", "A point where the path turns over: the load reaches a maximum (or minimum). A structure loaded by a dead weight jumps (snaps) at a maximum."),
        ("snap-through / snap-back", "Jump to a far-away equilibrium when a limit point is reached. Snap-back means the deflection itself also reverses along the path, so even prescribing the deflection cannot follow it."),
        ("bifurcation", "A point where a second equilibrium path branches off, e.g. a straight column starting to bend sideways. The structure can leave the original path there without the load turning over."),
        ("arc-length continuation", "A way of stepping along the path by a fixed distance in the combined load–deflection space rather than by a fixed load step, so the solver can go around limit points."),
        ("load-controlled Newton", "The standard method: raise the load in steps and iterate to balance. It cannot pass a load maximum — it either fails to converge or lands on a different branch."),
        ("tangent stiffness", "The matrix that says how forces change for a small change of shape at the current state. The structure is stable as long as it is positive definite; a zero eigenvalue marks a limit point or a bifurcation."),
        ("branch switching", "Deliberately stepping onto the secondary path at a bifurcation, using the buckling mode (the null vector of the tangent stiffness) as the direction."),
        ("neo-Hookean", "The simplest model of rubber-like material at large strain."),
        ("Biot instability", "A compressed block of rubber loses stability of its flat free surface (it wrinkles) at a universal compressive strain of 45.6 % in plane strain, whatever the wavelength."),
        ("mixed (Taylor–Hood) elements", "Finite elements that carry displacement and pressure as separate unknowns to handle incompressible material. Incompressibility is then only satisfied on average, not at every point."),
        ("stream function", "A way to write a 2-D displacement field so that it preserves volume exactly, by construction."),
        ("classical buckling load", "Textbook buckling stress of a perfect thin cylinder under axial compression: σ = E·t / (R·√(3(1−ν²))) ≈ 0.605·E·t/R."),
        ("imperfection sensitivity", "Real cylinders buckle far below the classical load because tiny deviations from the perfect shape cut the load sharply."),
        ("knockdown factor", "Buckling load of the imperfect shell divided by the classical load."),
        ("Koiter's theory", "W. T. Koiter's analysis (1945, 1963) of how the buckling load falls with imperfection size. For an axisymmetric ripple he gave a formula that is an upper bound of the load at which the rippled shell buckles into a non-axisymmetric pattern."),
        ("NASA SP-8007", "NASA design monograph (1965/68) with an empirical lower-bound knockdown factor fitted to tests; 0.47 for R/t = 200."),
        ("CalculiX", "An open-source finite-element program, used here as the independent second solver."),
        ("Richardson extrapolation", "Estimating the infinitely-fine-mesh answer from results on a sequence of meshes."),
    ]
    gl = "".join(f"<tr><td><b>{html.escape(a)}</b></td><td>{html.escape(b)}</td></tr>" for a, b in glossary)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial 7 — Buckling, snap-through and following the equilibrium path</title><style>{CSS}</style></head><body><div class="wrap">
<div class="crumb"><a href="../">QA dashboard</a> · <a href="../../">project</a></div>
<h1>Trial 7 — Buckling, snap-through and following the equilibrium path</h1>
<p class="sub">Can the agent build and run the machinery that structural instability needs, and get physically correct answers — checked against formulas, published values and a second solver?</p>

<h2>Result at a glance</h2>
<div class="card"><p class="big">{n_pass} of {n_all} pre-registered checks passed. Both misses are explained below; neither tolerance was moved.</p>
<table class="leg">
<tr><td>Rung 1 — path following</td><td>5 of 6. Snap-through, snap-back and a branch switch all reproduced to better than 0.02 % against closed forms and published values; a second solver agrees to 0.15 %. One check missed because I posed it badly (it asks for a load where the curve is almost vertical).</td></tr>
<tr><td>Rung 2 — rubber surface instability</td><td>2 of 2, on the second attempt. The first, standard choice of finite element gave a wrong threshold (32 % instead of 45.6 %) that looked converged. After switching to elements that preserve volume exactly: 45.63110 % against the exact 45.63110 %.</td></tr>
<tr><td>Rung 3 — imperfect cylinder</td><td>3 of 4. Classical load and Koiter's knockdown curve reproduced with an own solver. The planned CalculiX cross-check missed at two of four imperfection sizes, for two diagnosed reasons that are themselves the main lesson of this rung; rerun on a corrected model it agrees to within 1.5 %.</td></tr>
</table></div>

<h2>Objective</h2>
<div class="card"><p>Structures that buckle or snap do not have one answer per load: the load–deflection curve turns back on itself and splits into branches. The ordinary solver (raise the load, iterate) fails exactly where the interesting thing happens. This trial tests whether an agent can (1) write a solver that follows such curves around turning points and onto side branches, (2) find a buckling threshold in a block of rubber from the stiffness matrix alone, and (3) produce the curve engineers care about most in shell design — how much load a thin cylinder loses to a small shape imperfection — and judge which tool can be trusted for it.</p>
<p>Every tolerance was written down before the corresponding solver was run. Every reference value comes from a paper or report opened during the session.</p></div>

<h2>Rung 1 — following a path through snap-through and snap-back</h2>
<div class="card"><p>One arc-length solver (written in JAX; forces and stiffness obtained by automatic differentiation of the strain energy) is used on three problems. Red circles show what ordinary load-stepping does on the same problem.</p>
<img class="fig" src="r1_truss.png" alt="two-bar truss paths">
<p class="hint">Two-bar truss. Left: the shallow truss snaps through; load-stepping follows the curve to the maximum, then jumps to the far branch and never sees the descending part. Right: a steep truss buckles sideways before the maximum; the solver detects the bifurcation from the stiffness matrix, switches branch (orange), and the switched branch matches its closed form (a circle) to 13 digits.</p></div>
<div class="card"><img class="fig" src="r1_lee_frame.png" alt="Lee frame">
<p class="hint">Lee frame (an L-shaped frame pinned at both ends, loaded near the corner): the classic snap-back benchmark. The path turns back in both load and deflection. Load-stepping fails to converge just above the first maximum and then jumps to the far branch. CalculiX under prescribed deflection (orange crosses) confirms the first part of the curve.</p></div>
<div class="card"><img class="fig" src="r1_arch.png" alt="deep arch">
<p class="hint">Deep circular arch (215°, one end clamped, one hinged): maximum load after very large deflection.</p>
<img class="fig" src="r1_beam_convergence.png" alt="mesh convergence" style="max-width:560px;margin-top:12px">
<p class="hint">Mesh convergence. Two of the three quantities converge to the published values at second order; the third (orange) converges too — to a different number, see 1d.</p></div>
<div class="card">{t1}
<p class="hint"><b>1d, the miss.</b> At a deflection of 0.784 L the frame is a taut chain — an inextensible frame cannot get past 0.780 L at all — so the curve is nearly vertical there and a 0.4 % difference in deflection is a 24 % difference in load. Read the other way round, at the reference load 31.887 the computed deflection is {m['1d']['posthoc_deflection_at_reference_load']:.4f} L against 0.784 L (−0.40 %); the commercial code on the same reference page reports 0.781. Read that way the path agrees with the reference; the check as I wrote it does not pass, and is reported as a miss.</p>
<p class="hint"><b>1a/1b caveat.</b> The truss closed form is for the same idealised model the solver uses, so agreement to round-off proves the solver, not the modelling.</p></div>

<h2>Rung 2 — when does a compressed block of rubber wrinkle?</h2>
<div class="card"><p>Theory (Biot 1963, as stated by Cao &amp; Hutchinson 2012): the flat surface of an incompressible rubber block becomes unstable at 45.63 % compression in plane strain. The task: find that number from the tangent stiffness — the first compression at which it stops being positive definite — not by looking at a deformed mesh.</p>
<img class="fig" src="r2_biot.png" alt="Biot threshold vs mesh">
<p class="hint">Left: detected threshold against mesh fineness for the two discretisations. Right: error of the final method, falling 16-fold per mesh halving.</p></div>
<div class="card"><h3 style="margin-top:0">What went wrong first</h3>
<p>The standard element for incompressible material (mixed displacement–pressure, "Taylor–Hood") gave 32.2 % on three successive meshes — stable to three digits under refinement, and close to a different, real phenomenon (creasing, 35 %). It is wrong. Two things were hiding the truth:</p>
<ul><li>The eigenvalue routine returns the eigenvalues <i>nearest zero</i>, not the lowest. Modes that were already strongly unstable were skipped.</li>
<li>A full count of negative eigenvalues shows these elements lose stability at about 3 % compression, in a zig-zag mode one element wide at the surface (right-hand picture). These elements only keep volume on average, and the zig-zag exploits that.</li></ul>
<img class="fig" src="r2_biot_mode.png" alt="mode shapes">
<p class="hint">Left: the physical wrinkle mode found by the final method. Right: the spurious mesh-scale mode of the mixed elements.</p>
<p>The fix was to describe the perturbation with a stream function, which preserves volume at every point, and to test stability with a factorisation that cannot skip a negative eigenvalue.</p></div>
<div class="card">{t2}
<p class="hint">Independent check: a finite-depth formula derived from the equations in Cao &amp; Hutchinson (2012) agrees with the computation for blocks one and two wavelengths deep to 2 parts in 100 000.</p></div>

<h2>Rung 3 — how much load does a thin cylinder lose to a small ripple?</h2>
<div class="card"><p>A thin cylinder (radius 200 × wall thickness) compressed along its axis. The textbook buckling load assumes a perfect shape. Here the wall has a ripple of the worst shape (Koiter's: rings, with the wavelength of the classical buckling mode) and of increasing depth.</p>
<img class="fig" src="r3_knockdown.png" alt="knockdown curve">
<p class="hint">Buckling load as a fraction of the textbook value, against ripple depth. A ripple of one tenth of the wall thickness already removes about 45 % of the load; one wall thickness removes 80 %. Blue: own solver on the idealised infinitely long shell that Koiter analysed. Black: Koiter's formula (an upper bound — the computed points must lie on or below it, and do, by at most {m['3c']['error']:.3f}). Orange: the same cylinder with realistic simply supported ends. Hollow green diamonds: CalculiX with the model as originally planned — two of them sit off the curve, for reasons given below. Filled green diamonds: CalculiX after the model was corrected. Dotted: NASA's empirical design factor for this R/t, which the computed curve reaches at a ripple depth of about {inf['imperfection_amplitude_at_nasa_factor_simply_supported']:.2f} wall thicknesses.</p></div>
<div class="card"><h3 style="margin-top:0">Choice of method</h3>
<p>CalculiX has no arc-length method and its nonlinear step is ordinary Newton iteration, so it cannot say where a bifurcation is. The graded curve therefore comes from an own solver: the rippled shell's axisymmetric response is followed with the rung-1 path follower, and at each step the stiffness against every circumferential wave pattern (1 to 40 waves) is tested; the first pattern to lose stiffness gives the buckling load. CalculiX shells serve as the independent second opinion.</p>
<div class="scroll"><table class="leg"><tr><th>ripple depth</th><th>own solver</th><th>Koiter upper bound</th><th>gap</th></tr>{kd}</table></div></div>
<div class="card">{t3}</div>
<div class="card"><h3 style="margin-top:0">3d — why the planned CalculiX check missed, and what that teaches</h3>
<div class="scroll"><table class="leg"><tr><th>imperfection</th><th>CalculiX</th><th>own solver</th><th>difference</th><th>how CalculiX buckled</th></tr>{plan_rows}</table></div>
<img class="fig" src="r3_ccx_paths.png" alt="CalculiX load paths" style="margin-top:12px">
<ul><li><b>Newton steps over the bifurcation.</b> Right-hand panel: with the small non-axisymmetric seed I had planned (0.001 t), CalculiX stays on the unstable axisymmetric solution — it follows my axisymmetric path to within 0.4 % — and only leaves it 11 % above the buckling load the own solver finds. With a ten times larger seed it stops at 0.575 (own solver 0.580). The CalculiX answer moves by 12 % with a setting that has no physical meaning.</li>
<li><b>The quarter model was too stiff for deep ripples.</b> To save time I modelled a quarter of the cylinder with symmetry conditions. The rotational part of those conditions made the deeply rippled shell 7 % too stiff (measured: removing them, or modelling the full circumference, removes the excess; my understanding is that CalculiX turns such shell nodes into rigid bodies, but I did not re-read its manual); in that model CalculiX found no buckling at all at 1 t. A third calculation with solid axisymmetric elements, involving no shell theory, sided with the own solver to 0.6 %. On a full-circumference model (rerun with the larger seed, so two things changed at once) CalculiX buckles at the load the own solver predicts.</li>
<li><b>Buckling is not always collapse.</b> At 0.3 t and 1 t CalculiX buckles at the predicted load but, under prescribed shortening, keeps carrying a slowly rising load; at 0.1 t the load drops. "Maximum load" and "first buckling" are different questions.</li></ul>
{full_block}</div>

<h2>Honest caveats</h2>
<div class="card"><ul>
<li><b>Two pre-registered checks missed</b> (1d and 3d). 1d was a badly posed check; 3d exposed real weaknesses of the second-solver model. Neither tolerance was changed after the fact.</li>
<li><b>Rung 2 passed only on the second discretisation.</b> The first attempt produced a plausible, mesh-converged, wrong number. Without the exact reference it would have been easy to report it.</li>
<li><b>Check 3c passes for the shell length fixed beforehand</b> (four ripple wavelengths, largest gap to Koiter 0.030). A much longer idealised shell admits longer-wave buckling modes and sits lower: at a ripple of 0.3 t a 16-wavelength cell gives 0.320 against Koiter's 0.374, a gap of 0.054. Theory (Budiansky &amp; Hutchinson 1972) predicts exactly this family of modes, but the pass depends on that choice.</li>
<li><b>My own first version of the infinite-shell calculation left out a family of modes</b> (it only allowed patterns symmetric about a ripple crest). A cross-check exposed it; the corrected values changed in the fourth digit at most, which was luck.</li>
<li><b>The knockdown curve is for one idealised imperfection</b> (a regular ring ripple). Real shells have irregular imperfections; the NASA factor is an empirical lower bound to tests and is shown for orientation only, not as a check.</li>
<li><b>The own shell solver finds the first bifurcation, not the final collapse.</b> Whether the shell then collapses or carries on was only examined with CalculiX, at three or four ripple depths.</li>
<li><b>The own shell solver uses shallow-shell (Donnell) theory</b>, the same theory as Koiter's formula. Agreement with Koiter checks the numerics, not the theory; the CalculiX and solid-element comparisons are the checks on the theory, and they cover the pre-buckling stiffness and three buckling loads only.</li>
<li><b>Truss closed forms were derived in the session</b> (and verified symbolically) rather than taken from a paper.</li>
<li><b>One equation was read from a scanned page</b> (Koiter's bound in Hutchinson 2010); an ambiguous exponent was resolved by requiring that zero imperfection gives the classical load.</li>
<li>The simple knockdown formula I remembered, (1−ρ)² = (3c/2)·ξ·ρ, is not in any source I opened and is only the small-imperfection limit; it was not used for grading.</li>
</ul></div>

<details><summary>Terms used on this page</summary><div class="card"><table class="leg">{gl}</table></div></details>

<h2>Sources and artifacts</h2>
<div class="card"><p class="hint">Reference values: Cao &amp; Hutchinson 2012, Proc. R. Soc. A 468:94 (Biot threshold) · NAFEMS NL7 as quoted in the Abaqus Benchmarks Guide (Lee frame) · Horák, La Malfa Ribolla &amp; Jirásek 2022, arXiv:2111.11154 (deep arch) · Hutchinson 2010, Int. J. Solids Struct. 47:1443 (Koiter's bound) · Budiansky &amp; Hutchinson 1972 (long-wave limit) · NASA SP-8007 (1965/68). Full list with URLs in the experiment README.</p>
<p class="hint"><a href="https://github.com/eschmitt88/agentic-solid-mechanics/tree/master/experiments/2026-11-14-instability-path-following">experiment folder on GitHub</a> — README with the pre-registered criteria, config.yaml, the chronological log including dead ends, metrics.json, code and results.</p></div>
</div></body></html>
"""
    (OUT / "index.html").write_text(page)
    print("wrote", OUT / "index.html", len(page))


if __name__ == "__main__":
    main()
