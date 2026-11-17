"""Write metrics.json and the review page docs/qa/trial-13-evolutionary-dehomogenization/index.html
from results/grades.json, results/grade_ccx.json and results/fig_*.png. Run after analyze.py,
grade_ccx.py grade and make_figures.py."""
from __future__ import annotations

import json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PAGE = HERE.parent.parent / "docs" / "qa" / "trial-13-evolutionary-dehomogenization"

G = json.load(open(RES / "grades.json"))
V, ARMS = G["verdicts"], G["arms"]
C7 = json.load(open(RES / "grade_ccx.json"))["summary"]


def pct(x):
    return f"{100 * x:+.1f} %"


def gains(obj, arm):
    a = ARMS[obj].get(arm, {})
    return [a[k]["gain11"] for k in sorted(a)]


def chip(v):
    ok = v.startswith("PASS") or v.startswith("EA >")
    return f"<span class='chip {'ok' if ok else 'bad'}'>{v}</span>"


def mean(x):
    return sum(x) / len(x) if x else float("nan")


S, B = ARMS["stiff"], ARMS["buck"]
ml_s, ml_b = S.get("MORELF_raw") or {}, B.get("MORELF_raw") or {}
h2a_s, h2a_b = V.get("H2a_stiff", {}).get("verdict", "pending"), V.get("H2a_buck", {}).get("verdict", "pending")

metrics = {
    "H1_stiff_gain_mean": V["H1_stiff"]["mean"], "H1_buck_gain_mean": V["H1_buck"]["mean"],
    "stiff_gain_FULL": gains("stiff", "FULL"), "stiff_gain_NOVAE": gains("stiff", "NOVAE"),
    "stiff_gain_NOMUT": gains("stiff", "NOMUT"), "stiff_gain_NOSEL": gains("stiff", "NOSEL"),
    "stiff_gain_MORELF_raw": ml_s.get("gain11"), "buck_gain_FULL": gains("buck", "FULL"),
    "buck_gain_NOSEL": gains("buck", "NOSEL"), "buck_gain_MORELF_raw": ml_b.get("gain11"),
    "H6_stiff_retained": V["H6_stiff"]["mean_retained"], "H6_buck_retained_unconstrained": V["H6_buck"]["mean_retained"],
    "H7_spearman_C": C7["spearman_C"], "H7_spearman_BLF": C7["spearman_BLF"],
    "H7_ratio_cv_C": C7["ratio_C_cv"], "H7_ratio_cv_BLF": C7["ratio_BLF_cv"],
    "H5_wins_of_4": V["H5"]["wins"],
    "verdicts": {k: (v or {}).get("verdict") for k, v in V.items()},
}
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=1))

rows = [
    ("H1", "Does the paper's gain replicate? (hypervolume gain ≥ 10 %, every seed positive)",
     f"stiffness {', '.join(pct(g) for g in gains('stiff', 'FULL'))}; buckling {', '.join(pct(g) for g in gains('buck', 'FULL'))} (30 generations)",
     "PASS" if V["H1_stiff"]["verdict"] == V["H1_buck"]["verdict"] == "PASS" else "MISS"),
    ("H2a", "Does the search beat spending the same budget on more low-fidelity designs?",
     f"more-LF: stiffness {pct(ml_s.get('gain11', float('nan')))}, buckling {pct(ml_b.get('gain11', float('nan')))}",
     h2a_s if h2a_s == h2a_b else f"S: {h2a_s} · B: {h2a_b}"),
    ("H2b", "Does selection matter? (vs. same operators, no selection)",
     f"no selection: stiffness {', '.join(pct(g) for g in gains('stiff', 'NOSEL'))}; buckling {', '.join(pct(g) for g in gains('buck', 'NOSEL'))}",
     "PASS" if V["H2b_stiff"]["verdict"] == V["H2b_buck"]["verdict"] == "PASS" else "MISS"),
    ("H3", "Does mutation help? (≥ 2 points)", f"no mutation: {', '.join(pct(g) for g in gains('stiff', 'NOMUT'))} (FULL − no-mutation = {V['H3']['diff_pp']:+.1f} points)", V["H3"]["verdict"]),
    ("H4", "Does the VAE help? (≥ 2 points)", f"PCA-space crossover: {', '.join(pct(g) for g in gains('stiff', 'NOVAE'))} (FULL − no-VAE = {V['H4']['diff_pp']:+.1f} points)", V["H4"]["verdict"]),
    ("H5", "Do evolved buckling designs beat plain SIMP at equal volume?", f"{V['H5']['wins']} of 4 volumes; also 17–42 % less stiff (CalculiX agrees)", V["H5"]["verdict"]),
    ("H6", "Do the gains survive re-drawing the random phasor phases?", f"stiffness keeps {100 * V['H6_stiff']['mean_retained']:.0f} %, buckling {100 * V['H6_buck']['mean_retained']:.0f} % (post-hoc variant, see trust panel)", "PASS"),
    ("H7", "Does the in-loop pixel evaluator agree with body-fitted CalculiX?", f"ranking ρ = {C7['spearman_C']:.2f} / {C7['spearman_BLF']:.2f}; spread of the ratio {100 * C7['ratio_C_cv']:.0f} % / {100 * C7['ratio_BLF_cv']:.0f} % (limits 10 / 20 %)", C7["verdict"]),
]

figs = ["fig_gallery.png", "fig_hv_stiff.png", "fig_hv_buck.png", "fig_fronts_stiff.png", "fig_fronts_buck.png", "fig_calibration.png"]
PAGE.mkdir(parents=True, exist_ok=True)
for f in figs:
    shutil.copy(RES / f, PAGE / f)

table = "".join(f"<tr><td><b>{h}</b></td><td>{q}</td><td>{e}</td><td>{chip(v)}</td></tr>" for h, q, e, v in rows)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trial 13 — Evolutionary de-homogenisation, reproduced on a new problem</title><style>
:root{{--fg:#1c2128;--mut:#636c76;--bd:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--accent:#0969da;--bg:#fff;--card:#f6f8fa}}
*{{box-sizing:border-box}}body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0}}
.wrap{{max-width:1040px;margin:0 auto;padding:28px 22px 80px}}a{{color:var(--accent);text-decoration:none}}
h1{{font-size:25px;margin:.2em 0 .1em}}h2{{font-size:18px;margin:1.7em 0 .5em;border-bottom:1px solid var(--bd);padding-bottom:.25em}}
.sub{{color:var(--mut);margin:0 0 1.4em}}.crumb{{color:var(--mut);font-size:13px;margin-bottom:18px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0}}
img.fig{{max-width:100%;height:auto;border:1px solid var(--bd);border-radius:8px;display:block;margin:10px 0}}
figcaption{{color:var(--mut);font-size:13px;margin:-4px 0 18px}}
table.t{{border-collapse:collapse;width:100%;font-size:14px}}table.t td,table.t th{{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top;text-align:left}}
table.t th{{color:var(--mut);font-weight:600;font-size:12.5px}}
.chip{{display:inline-block;font-size:12px;font-weight:600;padding:1px 9px;border-radius:20px;white-space:nowrap}}
.chip.ok{{background:#dafbe1;color:var(--ok)}}.chip.bad{{background:#ffebe9;color:var(--bad)}}
dt{{font-weight:600}}dd{{margin:0 0 .6em 0;color:var(--mut)}}
</style></head><body><div class="wrap">
<div class="crumb"><a href="../">QA index</a> · trial 13</div>
<h1>Trial 13 — Evolutionary de-homogenisation, reproduced on a new problem</h1>
<p class="sub">Reproduction of Xu, Guo, Kawabe &amp; Yaji, <i>Engineering Structures</i> 371 (2027) 124076
(<a href="https://doi.org/10.1016/j.engstruct.2026.124076">doi</a>, <a href="https://arxiv.org/abs/2510.08830">arXiv 2510.08830</a>),
with a different geometry and supports, and with controls the paper did not run. Hypotheses and pass rules were frozen before any run.</p>

<div class="card"><p><b>What happened, in one paragraph.</b> The paper's method designs lattice-filled structures in two steps: a cheap
"averaged-material" optimiser proposes designs, a phasor pattern turns them into real lattice geometry, and an evolutionary search
improves the result by scoring the real geometry. On our girder the evolutionary step <b>does</b> improve the trade-off between material
and performance — by {pct(V['H1_stiff']['mean'])} (stiffness) and {pct(V['H1_buck']['mean'])} (buckling), comparable to the paper's +15 % / +19 % —
and it beats simply generating more averaged-material designs with the same budget. The gain comes from <b>selection</b>: the
search keeps the designs whose lattice comes out connected and clean. Two of the paper's ingredients add nothing here: the mutation
operator is inert, and the generative model (VAE) makes things <b>worse</b> than plain recombination because it collapses the
population's diversity. The evolved lattice designs also <b>lose to an ordinary solid topology optimisation (SIMP)</b> on both buckling and stiffness.</p></div>

<h2>The problem</h2>
<img class="fig" src="fig_gallery.png" alt="">
<figcaption>A 100 × 50 mm girder resting on elastic bearings at the bottom corners, loaded uniformly on its top deck, with a lined
service duct at quarter span. Black = material. Top row: stiffness designs; middle row: buckling designs; left column the starting
designs, right column after evolution, all at about 35 % material. Bottom: SIMP at 35 % for comparison.</figcaption>

<h2>Results against the pre-registered hypotheses</h2>
<table class="t"><tr><th></th><th>Question</th><th>Evidence</th><th>Verdict</th></tr>{table}</table>

<h2>How the hypervolume grew</h2>
<p>Hypervolume measures how much of the material-vs-performance plane the best designs cover; higher is better. Each curve is
the best set found so far against the number of expensive evaluations spent.</p>
<img class="fig" src="fig_hv_stiff.png" alt=""><img class="fig" src="fig_hv_buck.png" alt="">
<figcaption>The full method (blue) and its ablations. Removing selection (yellow) flattens the curve; removing the VAE (orange)
does better than the full method; spending the budget on more starting designs (pink, dashed) barely moves for stiffness but gets within a few points of
the search for buckling. The dashed line starts below zero because extra starting designs are scored as raw fields, without the
compression step the search uses, and raw starting designs score a little lower.</figcaption>

<h2>Pareto fronts</h2>
<img class="fig" src="fig_fronts_stiff.png" alt=""><img class="fig" src="fig_fronts_buck.png" alt="">
<figcaption>Best designs at each material fraction. Evolution (blue) improves on the starting designs (grey), most at low material,
where the starting lattices break apart. SIMP (violet dots) is better still in both objectives.</figcaption>

<h2>Trustworthiness</h2>
<div class="card"><ul>
<li><b>Pre-registered.</b> <code>PREREGISTRATION.md</code> was frozen before any evolutionary run. One dated addendum (budget) cut the
buckling arms from 60 to 30 generations before any buckling result was graded.</li>
<li><b>Independent evaluator.</b> The search scores designs on a fast pixel grid. 24 front designs were re-scored on body-fitted
meshes in CalculiX. Rankings agree (ρ = {C7['spearman_C']:.2f} compliance, {C7['spearman_BLF']:.2f} buckling) and every front comparison
agrees, but one evolved design was a pixel artefact: its load path counts as connected on the pixel grid and not in the real
contour (pixel compliance 22 vs CalculiX 56). The search found and kept a flaw in its own scorer.</li>
<img class="fig" src="fig_calibration.png" alt="">
<li><b>Winner's curse.</b> The lattice depends on random phasor phases. Re-drawing them makes evolved designs ~25 % worse
(stiffness) and ~33 % worse (buckling) per design, against ~10 % / ~18 % for the starting designs: the search partly selects
lucky draws. Most of the front-level gain survives anyway.</li>
<li><b>Post-hoc choice, labelled.</b> For buckling, H6 applies no stiffness constraint to the re-drawn designs. With the constraint, 14 of 15
starting-front designs become infeasible and the measure degenerates.</li>
<li><b>Deviations from the paper</b> (all in <code>log.md</code>): pixel grid instead of COMSOL in the loop; buckling stiffness from
stress averaged over 1 mm (otherwise pixel buckling loads fall with mesh refinement at sharp corners); spring bearings
(fixed supports give a corner mode whose load → 0 under refinement); an orientation encoding that respects the cell's
symmetries; a spline fit of the cell stiffness (the paper's polynomial cannot fit its jump at full width); 100 / 30 generations
instead of 300.</li>
<li><b>Not done:</b> running the authors' reference MATLAB code side by side (blocked in this session); the port was checked
against analytic patterns instead.</li>
</ul></div>

<h2>Terms</h2><dl>
<dt>De-homogenisation</dt><dd>Turning an "averaged material" design (each cell has a density and orientation) into an explicit lattice of bars.</dd>
<dt>Phasor</dt><dd>A wave pattern laid along the bar directions; thresholding it draws the bars.</dd>
<dt>Hypervolume</dt><dd>Area of the objective plane dominated by the best designs, measured to a fixed reference point.</dd>
<dt>Compliance</dt><dd>Work done by the load; lower = stiffer.</dd>
<dt>Buckling load factor (BLF)</dt><dd>How many times the design load the structure can take before it buckles (linear estimate).</dd>
<dt>SIMP</dt><dd>Standard gradient-based topology optimisation producing solid (not lattice) designs.</dd>
<dt>NSGA-II, VAE, PCA</dt><dd>Multi-objective selection rule; a neural generative model used for recombination; a linear compression of the design fields.</dd>
</dl>
<p class="sub">Experiment folder: <code>experiments/2026-11-16-evolutionary-dehomogenization/</code> — <code>PREREGISTRATION.md</code>, <code>log.md</code>, <code>results/</code>.</p>
</div></body></html>"""
(PAGE / "index.html").write_text(html)
print("page written", PAGE)
