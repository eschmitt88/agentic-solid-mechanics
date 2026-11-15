"""Build docs/qa/trial-10-contact/index.html from metrics.json and results/prediction.json."""
import json, os, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "qa", "trial-10-contact"))
os.makedirs(OUT, exist_ok=True)
M = json.load(open(os.path.join(HERE, "metrics.json")))
P = json.load(open(os.path.join(HERE, "results", "prediction.json")))
CH = {c["id"]: c for c in M["checks"]}
FIGS = ["r1_hertz_pressure.png", "r1_convergence.png", "r1_patch.png", "r2_mindlin.png", "r2_ironing_mesh.png", "r2_ironing_forces.png",
        "r2_ccx_settings.png", "r3_joint.png", "r3_prediction.png", "r3_histories.png"]
for f in FIGS:
    shutil.copy(os.path.join(HERE, "results", f), os.path.join(OUT, f))

pct = lambda v, d=1: f"{100*v:+.{d}f} %"
apct = lambda v, d=1: f"{100*abs(v):.{d}f} %"


def chip(c):
    return '<span class="chip ok">PASS</span>' if c["passed"] else '<span class="chip bad">MISS</span>'


def table(rows):
    """rows: (id, what, computed, reference, tolerance) with id in CH."""
    h = "<div class='tbl'><table class='leg'><thead><tr><th></th><th>what</th><th>computed</th><th>reference</th><th>tolerance</th><th></th></tr></thead><tbody>"
    for cid, what, comp, ref, tol in rows:
        c = CH[cid]
        h += (f"<tr><td data-l='check'>{cid}</td><td data-l='what'>{what}</td><td data-l='computed' class='metric'>{comp}</td>"
              f"<td data-l='reference' class='metric'>{ref}</td><td data-l='tolerance' class='metric'>{tol}</td><td data-l='result'>{chip(c)}</td></tr>")
    return h + "</tbody></table></div>"


def count(prefix):
    cs = [c for k, c in CH.items() if k.startswith(prefix)]
    return sum(1 for c in cs if c["passed"]), len(cs)


c = CH
r1 = count("1"); r2 = count("2")
iron = M["ironing_summaries"]
own_iron_key = [k for k in ("r2_ironing_own_h0.05.json", "r2_ironing_own_h0.1_half.json") if k in iron][0]
so = iron[own_iron_key]
def load(f):
    p = os.path.join(HERE, "results", f)
    return json.load(open(p)) if os.path.exists(p) else None
import rung2_ironing_own as RI
def ccx_iron(f):
    r = load(f)
    if r is None: return None
    r["sum"] = RI.summary(r["hist"]) if r["t_reached"] >= 1.6 else None
    return r
cx_fine = ccx_iron("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.05_dm0.001.json")
cx_0005 = ccx_iron("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.0005.json")
cx_001 = ccx_iron("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.001.json")
cx_k5 = ccx_iron("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_k5.json")
best_cx = next((x for x in (cx_0005, cx_001) if x and x["sum"]), None)
q1, q2 = P["Q1_min_dV_any_fastener"], P["Q2_min_dV_all_fasteners"]
EX = M["ironing_supplementary"]
sc = P["selfchecks"]

# ------------------------------------------------------------------ rung tables
t1 = table([
    ("1a-own", "Patch test (two blocks, mismatched meshes, uniform squeeze): worst deviation from uniform stress — own solver", apct(c["1a-own"]["value"], 2), "0 (exact)", "1 %"),
    ("1a-ccx", "Same — CalculiX", apct(c["1a-ccx"]["value"], 2), "0 (exact)", "1 %"),
    ("1b-a-own", "Cylinder: half-width of the contact strip — own", f"{c['1b-a-own']['value']:.5f}", f"{c['1b-a-own']['reference']:.5f} (Hertz)", "5 %"),
    ("1b-a-ccx", "Same — CalculiX", f"{c['1b-a-ccx']['value']:.5f}", f"{c['1b-a-ccx']['reference']:.5f}", "5 %"),
    ("1b-p0-own", "Cylinder: peak pressure — own", f"{c['1b-p0-own']['value']:.3f} ({pct(c['1b-p0-own']['error'], 2)})", f"{c['1b-p0-own']['reference']:.3f} (Hertz)", "3 %"),
    ("1b-p0-ccx", "Same — CalculiX", f"{c['1b-p0-ccx']['value']:.3f} ({pct(c['1b-p0-ccx']['error'], 2)})", f"{c['1b-p0-ccx']['reference']:.3f}", "3 %"),
    ("1b-dist-own", "Cylinder: whole pressure curve, average error — own", apct(c["1b-dist-own"]["value"]), "Hertz curve", "5 %"),
    ("1b-dist-ccx", "Same — CalculiX", apct(c["1b-dist-ccx"]["value"]), "Hertz curve", "5 %"),
    ("1c-F-own", "Sphere: force needed for a given squeeze — own", f"{c['1c-F-own']['value']:.5f} ({pct(c['1c-F-own']['error'])})", f"{c['1c-F-own']['reference']:.5f} (Hertz)", "3 %"),
    ("1c-F-ccx", "Same — CalculiX", f"{c['1c-F-ccx']['value']:.5f} ({pct(c['1c-F-ccx']['error'])})", f"{c['1c-F-ccx']['reference']:.5f}", "3 %"),
    ("1c-a-own", "Sphere: contact radius — own", f"{c['1c-a-own']['value']:.5f}", f"{c['1c-a-own']['reference']:.5f} (Hertz)", "5 %"),
    ("1c-a-ccx", "Same — CalculiX", f"{c['1c-a-ccx']['value']:.5f}", f"{c['1c-a-ccx']['reference']:.5f}", "5 %"),
    ("1c-p0-own", "Sphere: peak pressure — own", f"{c['1c-p0-own']['value']:.2f} ({pct(c['1c-p0-own']['error'])})", f"{c['1c-p0-own']['reference']:.2f} (Hertz)", "3 %"),
    ("1c-p0-ccx", "Same — CalculiX", f"{c['1c-p0-ccx']['value']:.1f} ({pct(c['1c-p0-ccx']['error'], 0)})", f"{c['1c-p0-ccx']['reference']:.2f}", "3 %"),
    ("1d", "Errors shrink with finer mesh, four mesh levels (own)", f"pressure {apct(c['1d']['value'][0], 2)} → {apct(c['1d']['value'][1], 2)}; sphere force {apct(c['1d']['value'][2], 2)} → {apct(c['1d']['value'][3], 2)}", "—", "finest &lt; coarsest"),
    ("1e", "The two solvers against each other, same mesh", f"pressure {pct(c['1e']['value'][0])}, sphere force {pct(c['1e']['value'][1], 2)}", "—", "2 %"),
])

def mrow(tag):
    a, b, cc = c[f"2a-{tag}"], c[f"2b-{tag}"], c[f"2c-{tag}"]
    name = "own" if tag == "own" else "CalculiX"
    if a["value"] is None:
        nr = "no result"
        return [(f"2a-{tag}", f"Size of the stuck zone at three loads — {name}", nr, "0.866, 0.707, 0.500 × a", "±0.05 a"),
                (f"2b-{tag}", f"Friction stress at the centre — {name}", nr, "closed form", "5 %"),
                (f"2c-{tag}", f"Whole friction-stress curve at half the sliding load — {name}", nr, "closed form", "8 %")]
    return [(f"2a-{tag}", f"Size of the stuck zone at three loads — {name}", ", ".join(f"{v:.3f}" for v in a["value"]) + " × a", "0.866, 0.707, 0.500 × a", "±0.05 a"),
            (f"2b-{tag}", f"Friction stress at the centre — {name}", ", ".join(pct(v, 2) for v in b["error"]), "closed form", "5 %"),
            (f"2c-{tag}", f"Whole friction-stress curve at half the sliding load, average error — {name}", apct(cc["value"]), "closed form", "8 %")]

def band_str(s):
    return f"vertical {s['Fv_t1']/1e8:.2f} / {s['Fv_mid']/1e8:.2f}, pulling {s['Ft_t1']/1e8:.2f} / {s['Ft_mid']/1e8:.2f}"

g = c["2g"]["value"] if c["2g"]["value"] else None
t2 = table(mrow("own") + mrow("ccx") + [
    ("2d", "Stuck-zone problem: the two solvers against each other", "no CalculiX result at the agreed mesh" if c["2d"]["value"] is None else pct(c["2d"]["value"]), "—", "5 %"),
    ("2e-own", "Ironing: four forces inside the range of published curves — own (×1e8; end of pressing / mid-slide)", band_str(so), "vertical 2.85–3.35 / 3.55–4.23, pulling 0.55–0.74 / 1.40–2.00", "inside"),
    ("2e-ccx", "Same — CalculiX in the agreed setting", "run fails 4 % into the slide" if c["2e-ccx"]["value"] is None else band_str(c["2e-ccx"]["value"]), "same ranges", "inside"),
    ("2f", "Ironing: the two solvers against each other", "no CalculiX result in the agreed setting" if c["2f"]["value"] is None else ", ".join(pct(v) for v in c["2f"]["value"].values()), "—", "5 %"),
    ("2g", "Ironing, own solver: change when the step is halved / the mesh is halved",
     (f"step: {max(abs(v) for v in g['step_halving'].values())*100:.1f} %; mesh 0.1→0.05: " + (f"{max(abs(v) for v in g['mesh_0p1_to_0p05'].values())*100:.1f} %" if g["mesh_0p1_to_0p05"] else "not finished")
      + f"; mesh 0.2→0.1: {max(abs(v) for v in g['mesh_0p2_to_0p1'].values())*100:.1f} %") if g else "—", "—", "3 %"),
])

q3rows = "".join(
    f"<tr><td data-l='test'>{r['test']}</td><td data-l='velocity change' class='metric'>{r['dV_ms']:.1f} m/s ({r['dV_fts']:.1f} ft/s)</td><td data-l='peak' class='metric'>{r['peak_g']:.0f} g</td>"
    f"<td data-l='broken (nominal)' class='metric'>{r['n_fractured_nominal']}" + (f" ({', '.join(map(str, r['order_nominal']))})" if r["order_nominal"] else "") + "</td>"
    f"<td data-l='chance any breaks' class='metric'>{100*r['prob_any']:.0f} %</td><td data-l='chance all six' class='metric'>{100*r['prob_all']:.0f} %</td></tr>" for r in P["Q3_per_test"])

n_pass, n_all = M["summary"]["passed"], M["summary"]["checks"]
best_txt = ""
if best_cx:
    s = best_cx["sum"]
    best_txt = (f"With increments {best_cx['dmax2']:g} on the h = {best_cx['h']:g} mesh CalculiX completes and gives a pulling force of {s['Ft_mid']/1e8:.2f}e8, "
                f"{(s['Ft_mid']/EX['own_Ft_mid_h0p1']-1)*100:+.0f} % from the own solver on the same mesh ({EX['own_Ft_mid_h0p1']/1e8:.2f}e8); the vertical force agrees to {abs(s['Fv_mid']/so['Fv_mid']-1)*100:.1f} %.")

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial 10 — Contact and friction</title><style>
:root{{--fg:#1c2128;--mut:#636c76;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--bg:#fff;--card:#f6f8fa}}
*{{box-sizing:border-box}}body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}}
.wrap{{max-width:1040px;margin:0 auto;padding:28px 22px 80px}}
a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}
h1{{font-size:25px;margin:.2em 0 .1em}}h2{{font-size:18px;margin:1.7em 0 .5em;border-bottom:1px solid var(--bd);padding-bottom:.25em}}
h3{{font-size:15.5px;margin:1.2em 0 .3em}}
.sub{{color:var(--mut);margin:0 0 1.4em}}.crumb{{color:var(--mut);font-size:13px;margin-bottom:18px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0}}
img.fig{{max-width:min(100%,660px);border:1px solid var(--bd);border-radius:8px;display:block;margin-top:8px}}
.hint{{color:var(--mut);font-size:13px;margin:.5em 0 0}}
table.leg{{border-collapse:collapse;width:100%;font-size:14px}}
table.leg td,table.leg th{{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top;text-align:left}}
table.leg th{{color:var(--mut);font-weight:600;font-size:12.5px}}
table.leg td:first-child{{color:var(--mut);white-space:nowrap}}
.chip{{display:inline-block;font-size:12px;font-weight:600;padding:1px 9px;border-radius:20px;white-space:nowrap}}
.chip.ok{{background:#dafbe1;color:var(--ok)}}.chip.bad{{background:#ffebe9;color:var(--bad)}}.chip.info{{background:#ddf4ff;color:#0550ae}}
code{{background:#eaeef2;padding:1px 5px;border-radius:5px;font-size:12.5px}}
.metric{{font-variant-numeric:tabular-nums}}.big{{font-size:17px;font-weight:600}}
ul{{padding-left:20px}}li{{margin:.3em 0}}
.kpis{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}}.kpi{{flex:1 1 150px;background:#fff;border:1px solid var(--bd);border-radius:10px;padding:10px 12px}}
.kpi b{{display:block;font-size:22px}}.kpi span{{color:var(--mut);font-size:12.5px}}
@media (max-width:640px){{
 .wrap{{padding:18px 12px 60px}}h1{{font-size:21px}}
 table.leg thead{{display:none}}
 table.leg,table.leg tbody,table.leg tr,table.leg td{{display:block;width:100%}}
 table.leg tr{{border:1px solid var(--bd);border-radius:8px;margin:8px 0;padding:6px 8px;background:#fff}}
 table.leg td{{border:0;padding:2px 0}}
 table.leg td[data-l]::before{{content:attr(data-l) ": ";color:var(--mut);font-size:12px}}
 table.leg td:first-child{{font-weight:600;color:var(--fg)}}
}}
</style></head><body><div class="wrap">
<div class="crumb"><a href="../">QA dashboard</a> · <a href="../../">project</a></div>
<h1>Trial 10 — Contact and friction</h1>
<p class="sub">Can the agent get contact problems right — two bodies pressing and rubbing on each other — when graded against exact formulas, published results, a second solver and, at the end, a real drop test it has not seen the result of?</p>

<h2>Result at a glance</h2>
<div class="card"><p class="big">{n_pass} of {n_all} pre-registered checks passed. One blind prediction is filed and waits for grading. No tolerance was moved.</p>
<div class="kpis">
<div class="kpi"><b>{r1[0]} / {r1[1]}</b><span>Rung 1 — pressing, no friction. Both solvers match the exact answers; one CalculiX output is wrong at a symmetry axis.</span></div>
<div class="kpi"><b>{r2[0]} / {r2[1]}</b><span>Rung 2 — friction. The own solver matches the exact stick–slip answer. CalculiX did not finish in the settings fixed in advance, and the two solvers disagree on one sliding force.</span></div>
<div class="kpi"><b>{q1['p50']:.1f} m/s</b><span>Rung 3 — blind prediction of the drop speed at which a bolted joint first loses a bolt (range {q1['p5']:.1f}–{q1['p95']:.1f}). Not yet graded.</span></div>
</div>
<p class="hint">Most of the misses are CalculiX runs that failed or could not be completed, counted as misses by a rule set beforehand. The misses that belong to the agent's own solver are 2e (one force outside the published range) and, in rung 3, a self-check that showed the first model was too simple.</p></div>

<h2>Objective</h2>
<div class="card"><p>Contact is where structural simulation most often fails quietly. Which parts of two surfaces touch is unknown in advance, friction makes the answer depend on the loading history, and the solver has to guess and re-guess which points stick and which slide. The trial climbs three rungs:</p>
<ul><li><b>Pressing without friction</b>, where a 140-year-old formula (Hertz) gives the exact answer.</li>
<li><b>Friction</b>: a problem with an exact answer for which part of the contact sticks and which part creeps, and a standard hard test ("ironing": a stiff die pressed into a soft slab and dragged along it).</li>
<li><b>A real experiment, predicted blind</b>: the Sandia Mechanics Challenge 2023, a bolted joint in a drop test.</li></ul>
<p>Each problem is solved twice: with CalculiX (the open-source solver this project operates) and with a contact solver the agent wrote for this trial. Pass marks were written down before anything was run.</p></div>

<h2>Rung 1 — pressing two bodies together</h2>
<div class="card"><img class="fig" src="r1_hertz_pressure.png" alt="contact pressure against the Hertz formula">
<p class="hint">Pressure across the contact, from the centre to the edge. Black: the exact formula. Blue dots: own solver. Orange squares: CalculiX. Top: a cylinder pressed on a flat. Bottom: a sphere. Everything lies on the formula, except CalculiX's reported pressure on the few points next to the sphere's axis, which is wrong by a factor of up to five (most of those points are off the top of the chart).</p></div>
<div class="card"><img class="fig" src="r1_convergence.png" alt="mesh convergence and finite size effect">
<p class="hint">Top: what a finer mesh buys. The error of the whole pressure curve falls steadily; the peak pressure is already within 0.3 % on the coarsest mesh; the width of the contact is only as good as the mesh spacing and does not improve steadily; CalculiX's peak pressure stays about 1 % high on every mesh. Bottom: the sphere needs 2.6 % more force than the formula says. That is not a solver error. The formula assumes the sphere is huge compared with the contact; here the contact is 5 % of the radius, and the excess shrinks in proportion as the contact gets smaller. The check passes its 3 % mark, but narrowly and for a reason the agent had not thought of when setting the mark.</p></div>
<div class="card"><img class="fig" src="r1_patch.png" alt="patch test">
<p class="hint">The patch test: two blocks with meshes that do not line up are squeezed uniformly; the stress should be exactly uniform. CalculiX's surface-to-surface contact is exact. Its older node-to-surface contact is off by 22 %. The agent's own solver is not exact either: its error depends on how finely it integrates over the contact, 0.5 % at the setting used.</p></div>
<div class="card">{t1}</div>

<h2>Rung 2 — friction</h2>
<div class="card"><h3 style="margin-top:0">Stick and slip under a sideways push</h3>
<p>Press a cylinder onto a block, then push it sideways with less force than it takes to slide. The middle of the contact stays stuck; the edges creep. Theory (Cattaneo, Mindlin) gives the size of the stuck zone and the friction stress everywhere.</p>
<img class="fig" src="r2_mindlin.png" alt="friction traction against the closed form">
<p class="hint">Friction stress across the contact at three sideways loads. Shaded: the stuck zone predicted by theory. Black: exact. Blue: own solver. Orange: CalculiX on a mesh twice as coarse as agreed (its run on the agreed mesh did not finish within an hour). CalculiX gets the size of the stuck zone and the centre value right to 2 %, but its friction stress scatters inside the stuck zone, and after pressing alone it already reports friction stress of up to half the limit where theory has none (own solver: 3 % of the limit).</p></div>
<div class="card"><h3 style="margin-top:0">Ironing: press a die into a slab and drag it</h3>
<img class="fig" src="r2_ironing_mesh.png" alt="deformed mesh of the ironing problem">
<img class="fig" src="r2_ironing_forces.png" alt="ironing forces against time">
<p class="hint">Force on the die against time (top: vertical, bottom: pulling). Grey bars: the spread of four published solutions (they differ by up to 30 %; there is no exact answer). Crosses: tabulated Code_Aster values. A red × marks where a CalculiX run stopped.</p>
<p>What happened:</p>
<ul><li>The vertical force agrees everywhere: own solver, CalculiX and the published curves.</li>
<li>While the die is only being pressed down, both solvers reproduce the tabulated values to within 4 %.</li>
<li>Once the die slides, the pulling force of the own solver ({so['Ft_mid']/1e8:.2f}e8) lies below every published curve (1.45–1.95e8). This is a <b>miss</b>.</li>
<li>CalculiX in the setting fixed in advance stops 4 % into the slide. In settings that do finish, its pulling force ranges from 1.36 to 1.78e8 depending on the step size and the contact stiffness. {best_txt}</li>
<li>CalculiX's pulling force falls steadily as its steps get smaller: 1.40e8 at steps of 0.001, 1.36e8 at 0.0005. Continuing that line to zero step size gives {EX['ccx_Ft_mid_linear_extrapolation_to_zero_increment']/1e8:.3f}e8; the own solver on the same mesh gives {EX['own_Ft_mid_h0p1']/1e8:.3f}e8. This is a two-point extrapolation, done after the fact, so it is evidence, not a graded check.</li>
<li>Part of the gap to the published curves is the rubber law, not the contact. Both solvers were deliberately run with CalculiX's variant of the law so they could be compared; with the textbook variant the own solver gives {EX['own_h0p1_textbook_neo_hookean']['Ft_mid']/1e8:.2f}e8, against Code_Aster's 1.45e8.</li></ul>
<img class="fig" src="r2_ccx_settings.png" alt="CalculiX result by setting">
<p class="hint">The same problem in CalculiX with different numerical settings. The physics is identical in every bar. Smaller steps move CalculiX towards the own solver's value.</p>
<p class="hint">Reading: the own solver's pulling force is converged (it changes by 1.3 % when the step is halved and by 0.5 % when the mesh is halved) and CalculiX approaches it from above as its steps shrink. It still counts as a <b>miss</b> against the published range, which was the mark set in advance. The whole dispute is small in physical terms: it is whether the pressure under the die leans forward by 1° or by 5°, and the four published codes disagree with each other by more than that.</p></div>
<div class="card">{t2}</div>

<h2>Rung 3 — a blind prediction: when do the bolts break?</h2>
<div class="card"><p>Sandia National Laboratories dropped a block with a 1.5 lb aluminium rod bolted to its side (six small steel screws through a flange) onto a hard stop, in 19 documented tests of increasing speed. The shock throws the rod downward; at some speed the screws break. Teams were asked to predict, without seeing the result, the lowest speed that breaks one screw, the lowest that breaks all six, and how the parts move.</p>
<p><b>What the agent had:</b> the public information packet (a PDF). The data files behind it need a login that is not available here, so the screw strength curves were read off the packet's figures and the shock pulses were rebuilt from three numbers per test. <b>What it did not have:</b> any result. The results report exists; it was not opened.</p>
<p><b>Model:</b> not a full 3D crash simulation but a small dynamic model: the rod as a beam, the six screws as springs that stretch, yield and snap, the flange pressing and sliding on the block with friction. CalculiX 3D models were used to check and correct it.</p>
<img class="fig" src="r3_joint.png" alt="joint behaviour and the failed self-check">
<p class="hint">Bottom: the check the first version of the model failed. It treated the flange as rigid and came out 44 % too stiff. Top: a CalculiX 3D push on the joint shows why — the thin flange bends, and starts to bend permanently at a quarter of the load of the mildest test. The model was rebuilt with a bending flange before the prediction was frozen. That one change moved the predicted breaking speed up by about a quarter (median 19.4 → 23.7 m/s).</p></div>
<div class="card"><img class="fig" src="r3_prediction.png" alt="predicted probability of fracture per test">
<p class="hint">The frozen prediction. Blue: share of 400 model variants (screw strength, friction, hole clearance and other unknowns varied over plausible ranges) in which at least one screw breaks in that test. Orange: all six break. Numbers under the bars: screws broken in the single best-guess model.</p>
<div class="kpis">
<div class="kpi"><b>{q1['p50']:.1f} m/s</b><span>first screw breaks (median; {q1['p50']/0.3048:.0f} ft/s). Range {q1['p5']:.1f}–{q1['p95']:.1f} m/s. Best-guess model: {q1['nominal']:.1f} m/s, test 9.</span></div>
<div class="kpi"><b>{q2['p50']:.1f} m/s</b><span>all six break (median). From {q2['p5']:.1f} m/s; in {100*q2['fraction_of_samples_never_all_six']:.0f} % of variants the bottom screw survives every test.</span></div>
<div class="kpi"><b>1 → 2, 6 → 3, 5 → 4</b><span>order: top screw first, pulled apart; the rest follow within the same shock.</span></div>
</div>
<img class="fig" src="r3_histories.png" alt="time histories of two tests">
<p class="hint">Two neighbouring tests in the best-guess model, one that survives and one that does not.</p>
<p class="hint">Also predicted, although not asked for: the flange is bent permanently in every test, including the mildest. Not predicted by the model (answered by judgment only): no cracks in the block or in the rod.</p></div>
<div class="card"><h3 style="margin-top:0">Checks on the model itself (the outcome is graded by someone else)</h3>
<div class='tbl'><table class='leg'><thead><tr><th></th><th>what</th><th>result</th><th>tolerance</th><th></th></tr></thead><tbody>
<tr><td data-l='check'>3a</td><td data-l='what'>Vibration frequency of the rod, small model vs CalculiX 3D</td><td data-l='result' class='metric'>{sc['3a']['rom_hz']:.0f} vs {sc['3a']['ccx_hz']:.0f} Hz ({pct(sc['3a']['err'])})</td><td data-l='tolerance'>5 %</td><td><span class="chip ok">PASS</span></td></tr>
<tr><td data-l='check'>3b</td><td data-l='what'>Static joint response, small model as first written vs CalculiX 3D</td><td data-l='result' class='metric'>tip deflection {pct(sc['3b']['registered_model']['err_tip'], 0)}, top-screw force {pct(sc['3b']['registered_model']['err_T1'], 1)}; after rebuilding: {pct(sc['3b']['recalibrated']['err_tip'])}, {pct(sc['3b']['recalibrated']['err_T1'])}</td><td data-l='tolerance'>15 %</td><td><span class="chip bad">MISS</span></td></tr>
<tr><td data-l='check'>3c</td><td data-l='what'>Energy bookkeeping of the time stepping</td><td data-l='result' class='metric'>{100*sc['3c']['worst']:.2f} % unaccounted</td><td data-l='tolerance'>2 %</td><td><span class="chip ok">PASS</span></td></tr>
<tr><td data-l='check'>3d</td><td data-l='what'>Halving the time step</td><td data-l='result' class='metric'>{100*sc['3d']['change']:.2f} % change</td><td data-l='tolerance'>2 %</td><td><span class="chip ok">PASS</span></td></tr>
</tbody></table></div></div>
<div class="card"><details><summary>Prediction for each of the 19 tests</summary>
<div class='tbl'><table class='leg'><thead><tr><th>test</th><th>velocity change</th><th>peak</th><th>broken, best guess (order)</th><th>chance any</th><th>chance all six</th></tr></thead><tbody>{q3rows}</tbody></table></div></details></div>

<h2>What CalculiX contact is like to operate</h2>
<div class="card"><ul>
<li><b>Frictionless contact with the surface-to-surface option is good.</b> Exact patch test, Hertz within about 1 %, and the peak pressure moves by only 2 % when the contact stiffness is changed 10,000-fold (the softest setting makes the contact patch 7 % too wide).</li>
<li><b>The older node-to-surface option is unreliable</b>: fine on the 2D Hertz problem, but 22 % error on the patch test with mismatched meshes, a crash ("double free") on the axisymmetric model, and failure in the ironing problem before sliding even starts.</li>
<li><b>The mortar option refuses 2D models</b> outright.</li>
<li><b>Reported contact pressure is wrong next to a symmetry axis</b> in axisymmetric models, although the total force is right.</li>
<li><b>Friction with large sliding is fragile and setting-dependent.</b> Whether the run finishes, and what force it reports, depends on the step size and the contact stiffness. A stiffer contact fails; a softer one finishes with a pulling force 24 % higher; five and ten times smaller steps finish, 7 % and 3 % above the own solver. The error falls in proportion to the step size; its cause inside CalculiX was not investigated.</li>
<li><b>Pressing with friction leaves stray friction stress</b> in the contact where theory has none, and it barely shrinks on a finer mesh.</li>
<li><b>It is slow on fine 2D contact meshes with friction</b>: the finest stick–slip model was stopped after an hour, still in its second load step, and a ten times stiffer stick setting on the coarser mesh after 45 minutes.</li>
<li><b>Output traps</b>: one output request with a different frequency silently thins all the others; forces of axisymmetric models are per 1/180 of the circle.</li></ul></div>

<h2>What went wrong, and what caught it</h2>
<div class="card"><ul>
<li><b>A rigid flange.</b> The first drop-test model was 44 % too stiff. Caught by the pre-registered comparison with a 3D model, before the prediction was frozen.</li>
<li><b>Two wrong derivatives in the own solver.</b> A library function reported a slope of one where the true slope is zero, and the slip measure ignored that the surface it is measured on stretches. Both showed up as a solver that would not converge, not as a wrong answer.</li>
<li><b>A first contact formulation that could not pass the patch test at all</b> (more constraints than unknowns). Replaced.</li>
<li><b>A tolerance set on a wrong assumption.</b> The sphere's 3 % mark assumed the finite-size effect was negligible; it is 2.6 %.</li>
<li><b>A CalculiX run that looked fine and was not</b>: the first ironing run (coarse mesh) finished and landed inside the published range at 1.78e8. With a finer mesh and ten times smaller steps the same code gives 1.36e8.</li>
<li><b>A reference range that mixes different physics.</b> The four published ironing curves were computed with different rubber laws, which the agent only found out after setting "inside the published range" as a pass mark.</li>
<li><b>Operating slips</b>: a time stamp typed from memory instead of the clock, twice (corrected within minutes); one command that stopped the agent's own shell; too many processor cores used for about half an hour.</li></ul></div>

<h2>Honest caveats</h2>
<div class="card"><ul>
<li>The ironing reference is a figure with four curves that disagree with each other. "Inside the published range" is a weak test. The stronger test, agreement between the two solvers, <b>failed</b> for the sliding force.</li>
<li>The own solver's miss on the ironing pulling force could still hide an error in it. No bug was found and CalculiX moves towards the same value as its steps shrink, but that comparison was assembled after the planned one failed.</li>
<li>The own solver passes the patch test only approximately (0.5 %), and its stick–slip pass on the traction curve is close to the mark (see table).</li>
<li>The drop-test prediction rests on curves read off figures, rebuilt shock pulses, an assumed rule for screws loaded in tension and shear together, and a model that was changed once after its first self-check. The range (18–30 m/s) is wide. A hit inside it means less than a hit inside a narrow range would.</li>
<li>Before the flange was allowed to bend permanently, the same model predicted first fracture at a median of 19.4 m/s (range 16.5–20.2, from a 60-sample trial run). The very first version, with a rigid flange, put it at about 20 m/s in single best-guess runs that still contained a bug in the shock-pulse shape. The agent had decided to add flange bending before seeing the 19.4 and had seen no experimental result at any point, but a reader should know all three numbers exist.</li>
<li>The model cannot say whether the last screw breaks once five are gone (the rod is then swinging freely), and it underestimates how evenly the flange spreads load between the top three screws.</li></ul></div>

<h2>Glossary</h2>
<div class="card"><table class="leg">
<tr><td>Hertz contact</td><td>The exact solution (from the 1880s) for two smooth elastic bodies pressed together: size of the contact patch, peak pressure and pressure distribution.</td></tr>
<tr><td>Patch test</td><td>A sanity test: squeeze two flat blocks uniformly; any contact method worth using should return exactly uniform stress.</td></tr>
<tr><td>Stick / slip</td><td>Under friction, touching points either move together (stick) or slide with a friction force proportional to pressure (slip).</td></tr>
<tr><td>Cattaneo–Mindlin</td><td>The exact solution for a contact pushed sideways below the sliding load: a stuck centre with creeping edges.</td></tr>
<tr><td>Ironing</td><td>A standard hard test for contact software: a stiff die pressed into a soft slab and dragged along it.</td></tr>
<tr><td>Penalty stiffness</td><td>A numerical spring that pushes overlapping surfaces apart. Too soft and the surfaces overlap; too stiff and the solver struggles.</td></tr>
<tr><td>Surface-to-surface / node-to-surface</td><td>Two ways of deciding where contact forces act: averaged over patches of surface (newer, smoother) or at single mesh points (older).</td></tr>
<tr><td>Blind prediction</td><td>A prediction written down and frozen before the experimental result is looked at.</td></tr>
<tr><td>Velocity change</td><td>How much speed the falling carriage loses in the impact; the measure of how severe each drop test is.</td></tr>
<tr><td>Flange</td><td>The flat rim at the end of the rod through which the six screws pass.</td></tr>
</table></div>
<p class="hint">Experiment folder: <code>experiments/2026-11-15-contact-ladder/</code> — pass marks in <code>PREREGISTRATION.md</code>, chronology including dead ends in <code>log.md</code>, frozen prediction in <code>results/prediction.json</code>.</p>
</div></body></html>"""
open(os.path.join(OUT, "index.html"), "w").write(html)
print("written", os.path.join(OUT, "index.html"), len(html))
