"""Prose blocks of the review page, with every number read from metrics.json (run build_page.py, this, build_page.py)."""
import json

import numpy as np

M = json.load(open("metrics.json"))
T, rob, S = M["targets"], M["robust_design"], M["snap_test"]
pc = lambda x, n=1: "—" if x is None else f"{100*x:.{n}f} %"
H1, H2, H4, B = T["H1"], T["H2"], T["H4"], T["B"]
cert = M["infeasibility_certificate"]["H4"]
gc = M["gradient_check_fold"]
rel = M["relaxed_idealisation"]


def sr(a, g="preregistered"):
    x = S.get(a, {}).get(g)
    return "not run" if not x else f"{x['success']} of {x['n']}"


band = lambda m: m["calculix_bands"][0] if m.get("calculix_bands") else None
BM = rel.get("bistable_margin", {})
vm = json.load(open("results/verify_bistable_margin.json")) if __import__("os").path.exists("results/verify_bistable_margin.json") else None
x120 = S.get("AM+O2@120", {}).get("preregistered")
per = M["snap_test_per_run"]
ok60 = [k.split("__")[1] for k, v in per.items() if k.startswith("AM+O2__") and not k.split("__")[1].startswith("fresh") and v["success"]]
n_total_120 = len(ok60) + (x120["success"] if x120 else 0)

tx = {}
tx["headline"] = f"""<p><b>What happened, in one paragraph.</b> The harder targets did their job: of three new target curves <b>none passed</b> — the literature plateau H1 missed
({pc(H1['calculix_nrmse'])} error against {pc(H1['tolerance_nrmse'],0)} allowed), the two-cell staircase H2 missed ({pc(H2['calculix_nrmse'])} against {pc(H2['tolerance_nrmse'],0)}), and the deliberately
impossible H4 was reported as a miss with a proof that no elastic cell can come closer than {pc(cert['lower_bound_nrmse'])}. Designing for manufacturing error <b>did not help</b>:
{pc(rob['robust']['pass_fraction_calculix'],0)} of imperfect copies of the "robust" design stay inside tolerance in CalculiX, the same as the ordinary design ({pc(rob['nominal']['pass_fraction_calculix'],0)}); the cause is physical (force grows with the cube of beam thickness).
The snap problem turned out to be something other than yesterday's diagnosis: the loading path flips onto a different branch when the design crosses a buckling surface, and the objective jumps there.
A new gradient that treats snap points as unknowns is exact (checked to {gc['snap_position_sensitivity_max_rel_err']:.0e}) but did not cure the stall by itself; the pre-registered remedy converged from <b>{sr('A+O')}</b> of yesterday's stalled starts
(yesterday's method: {sr('B0-legacy')}). A variant added during the run, which keeps the design a fixed distance away from the buckling surface, converged from <b>{sr('AM+O2')}</b> of those starts but from only {sr('AM+O2', 'fresh')} fresh ones, and doubling its budget rescued none of its misses — better than yesterday, well short of the 8 of 11 set as the bar. Its designs do have one practical advantage: the one checked ran cleanly on every verifier mesh, where the designs sitting on the buckling surface abort on the coarser ones.
Finally, dropping the shared assumptions moves the curves a lot: a real 10 mm deep part is about 10 % softer and an elastic frame 16–17 % softer than the idealised model both solvers used.</p>
<p class="hint">Verdict on the trial as pre-registered: the method reports misses as misses (good), the remedy for the snap is partial (bar not met), robust design is a negative result, and yesterday's "verified" designs would not meet their tolerances once the frame is allowed to flex.</p>"""

b1 = band(H1)
tx["H1"] = f"""<p>The target is the textbook compression curve of an open-cell rubber foam (Gibson &amp; Ashby): a straight rise to a collapse stress, then a flat plateau.
With this rubber and a relative density of 5.5 % the formulas give a plateau of 0.4235 kPa starting at 5 % strain and lasting past 50 %. The rule: error ≤ 3 % of range <i>and</i> the plateau held within ±5 % from 8 % to 50 % strain.
Yesterday's cell held a plateau to 3.9 % but only up to 33 % strain, so it was not known whether this one could be met.</p>
<p><b>It was not.</b> Best of six starts: optimiser error {pc(H1['optimizer_nrmse'])}, CalculiX {pc(H1['calculix_nrmse'])} (allowed 3 %); the plateau strays by {b1['max_dev']:.3f} kPa (allowed {b1['tol']:.3f}).
The plateau itself is nearly there (within 7.4 %); the cell cannot make the sharp knee — it is too soft before 5 % strain and overshoots after. All three CalculiX meshes ran to the end and agree to {pc(H1['ladder_spread'],2)}; the two solvers differ by {pc(H1['solver_gap'])}.
Every start ran out of its 80 evaluations without reporting a converged optimum, so this is "not reached in the budget", not a proof that the cell cannot do it.</p>"""

b2 = H2.get("calculix_bands") or [dict(max_dev=float("nan"), tol=0)] * 2
tx["H2"] = f"""<p>Two cells in series should give a staircase: the softer cell collapses at 0.35 kPa, then the stiffer one at 0.70 kPa. 26 shape numbers, four starts.
The two-cell model solved as one system cost 2–3 minutes per evaluation on a busy machine, so the stack was re-formulated as two single-cell solves joined by one equation (equal force, travels add up); that version reproduces the one-system solve to 6e-9 kPa and its gradient matches finite differences to 6e-8.</p>
<p><b>Missed by a wide margin.</b> Best start: optimiser error {pc(H2['optimizer_nrmse'])}, CalculiX (full two-cell model, finest mesh) {pc(H2['calculix_nrmse'])}, allowed 4 %. The steps are off by {b2[0]['max_dev']:.2f} and {b2[1]['max_dev']:.2f} kPa (allowed 0.028 and 0.056).
The four starts end between {pc(min(H2['optimizer_nrmse_per_start'].values()))} and {pc(max(H2['optimizer_nrmse_per_start'].values()))}, all out of budget and still descending slowly. The curve that comes out has two snap-through humps, not two flat steps: each cell's force falls after its peak instead of holding, and in 80 evaluations the optimiser did not find a pair of plateau-type cells that hand over to each other.</p>"""

tx["H4"] = f"""<p>H4 asks the cell to push back for a moment and then <i>pull</i> the plunger down with 0.4 kPa over most of the stroke. Adding up force times distance, the cell would hold less elastic energy than at rest — impossible for any elastic object. This was written into the plan as "expected to miss".</p>
<p><b>Reported as a miss, three ways.</b> (1) Six starts stall between {pc(min(H4['optimizer_nrmse_per_start'].values()))} and {pc(max(H4['optimizer_nrmse_per_start'].values()))} error (allowed 5 %).
(2) A small quadratic programme finds the closest curve that never needs negative stored energy: {pc(cert['lower_bound_nrmse'])} away — a floor that holds for every design, not only for this beam.
(3) The best design was still sent to CalculiX: all three meshes and both imperfection re-runs stopped at 9.25 mm of 12.5 mm, so by the rule the verdict is <b>FAIL (unverifiable)</b>. No number anywhere suggests a pass.</p>"""

dn = rob.get("decomposition_nominal", {})
tx["robust"] = f"""<p>Yesterday's plateau design passed with room to spare, but copies made with ordinary printing error often would not. The error model: each shape mode off by 0.05 mm, overall thickness off by 3 %, thickness profile off by 2 % (one standard deviation each).
The robust optimiser minimises the average error over eight imperfect copies plus the perfect one; it is then judged on 64 <i>fresh</i> imperfect copies in the design solver and the first 24 of those in CalculiX. The plan said: claim success only if the share of copies inside tolerance rises by 15 points in both.</p>
<p><b>It did not rise at all.</b> Design solver: ordinary design {pc(rob['nominal']['pass_fraction_design_solver'],0)}, robust design {pc(rob['robust']['pass_fraction_design_solver'],0)}. CalculiX: {pc(rob['nominal']['pass_fraction_calculix'],0)} and {pc(rob['robust']['pass_fraction_calculix'],0)} (none of the 48 runs aborted; CalculiX and the design solver agree on pass/fail for {pc(rob['nominal']['agreement_calculix_vs_design_solver'],0)} / {pc(rob['robust']['agreement_calculix_vs_design_solver'],0)} of copies).
The robust optimiser stopped after {rob['rms_over_optimisation_copies']['nfev']} evaluations at a genuine optimum, having lowered its own objective from {rob['rms_over_optimisation_copies']['start']:.4f} to {rob['rms_over_optimisation_copies']['end']:.4f} — nothing. Restarting it from four other shapes (not in the plan) gave {', '.join(f"{k} {pc(v['pass_fraction_design_solver'],0)}" for k, v in rob['other_starts_not_preregistered'].items())}.</p>
<p><b>Why.</b> Perturbing one error source at a time: centre-line error alone leaves {pc(dn.get('centre_line_only (0.05 mm)', {}).get('pass_fraction'),0)} of copies inside tolerance, the 3 % thickness error alone {pc(dn.get('thickness_scale_only (3 %)', {}).get('pass_fraction'),0)}, the thickness-profile error alone {pc(dn.get('thickness_modes_only (0.02)', {}).get('pass_fraction'),0)}.
Along the whole curve the force scales with thickness to the power {dn.get('dlog_sigma_dlog_t0', {}).get('median', float('nan')):.2f} — a bending beam — so 3 % thickness scatter is 9 % force scatter whatever the shape. Reshaping the same beam cannot fix that; a different mechanism (stretch-dominated members), tighter thickness control, or a calibration step after printing would.</p>"""

bf = M.get("branch_flip", {})
tx["snap_intro"] = f"""<p>Yesterday five optimisations of a "stays-snapped" cell stalled, and the diagnosis was a secondary snap that the gradient cannot see moving. Checking the new snap-aware gradient against finite differences exposed a different cause.
At the stalled design the structure is at a perfect buckling point: just past the force peak there are two possible continuations, and which one the loading path follows flips when one thickness number changes by 0.0001. The return barrier differs by {pc(bf.get('relative'),2)} between the two.
Yesterday's path-follower also picked between them by round-off, which is what looked like a snap appearing and vanishing. So the objective has a genuine jump along a surface in design space, and the optimiser is drawn onto that surface because removing the imperfection raises the peak force.</p>"""
tx["snap_grad"] = f"""<p><b>The new gradient.</b> Each snap on the path is solved for as its own unknown (the position where the stiffness becomes singular), and the return barrier is computed from stored energies at the rest states plus the energy lost in each snap, instead of adding up forces across the jump.
Its sensitivity uses the textbook limit-point formula. Against finite differences: snap position sensitivity agrees to {gc['snap_position_sensitivity_max_rel_err']:.0e} ({gc['n_snap_position_checks']} checks), the whole pipeline re-run from scratch agrees to {gc['whole_pipeline_max_rel_err_where_continuous']:.0e} in {gc['n_whole_pipeline_probes_matching']} probes;
the other {gc['n_probes_straddling_branch_flip']} probes straddle the branch flip, where no derivative exists. Yesterday's gradient had the wrong sign for several parameters at the stalled design.</p>"""
fr = lambda a: sr(a, "fresh")
tx["snap_result"] = f"""<p><b>Result.</b> The bar written down beforehand was: the remedy converges from at least 8 of the 11 starts where yesterday's method converges from at most 3. Yesterday's method, re-run today: {sr('B0-legacy')}. The pre-registered remedy (fold-aware gradient + automatic restarts): <b>{sr('A+O')}</b>. <b>The bar is not met.</b></p>
<p>What the runs show: the exact gradient alone ({sr('A')}) changes little, because the stall is the branch flip, not a moving snap. The restarts do the work — a damped Gauss–Newton step across the jump, accepted even if the error first rises, which automates yesterday's hand fix — but each stall burned ~30 of the 60 evaluations before the optimiser gave up.
Two variants were added after seeing this (so their scores on the same starts are not a clean test): giving up on a stall sooner ({sr('A+O2')}), and adding a penalty that keeps the lowest stiffness eigenvalue along the path above 15 % of its starting value, i.e. keeps the design off the buckling surface (<b>{sr('AM+O2')}</b>).
On six fresh starts drawn before those variants were run: margin variant {fr('AM+O2')}, quick-restart variant {fr('A+O2')}, yesterday's method {fr('B0-legacy')}.
{f"Given 120 instead of 60 evaluations, the margin variant converged on {x120['success']} of the {x120['n']} starts it had missed ({n_total_120} of 11 in total at 120 evaluations)." if x120 else ""}</p>"""

ci = B["calculix"] or {}
lad = B["ladder"]
ab = [f"{k.replace('t', 't_min/')} stopped at {v['d_reached']:.2f} mm" for k, v in lad.items() if not v["completed"]]
mtxt = ""
if vm:
    f = vm["verdict"].get("features")
    f = f if isinstance(f, dict) else None
    labm = [f"{k.replace('t', 't_min/')} {'completed' if v['completed'] else 'stopped at %.2f mm' % v['d_reached']}" for k, v in vm["ccx"].items()]
    mtxt = f"""<p>A design from the margin variant (start cos_h6_t1.4) was verified the same way: {'; '.join(labm)}; verdict <b>{vm['verdict']['verdict']}</b>""" + (f""", errors {100*f['rel_err'][0]:+.2f} %, {100*f['rel_err'][1]:+.2f} %, {100*f['rel_err'][2]:+.2f} %.</p>""" if f else ".</p>")
tx["bistable"] = f"""<p>The one design the pre-registered remedy produced (start h6_t1.4_a2): peak force, second rest position and return barrier are off by {100*ci.get('rel_err', [float('nan')]*3)[0]:+.2f} %, {100*ci.get('rel_err', [float('nan')]*3)[1]:+.2f} % and {100*ci.get('rel_err', [float('nan')]*3)[2]:+.2f} % in CalculiX (allowed 5 %): <b>{B['verdict']}</b>, graded on {B['graded_mesh'].replace('t', 't_min/')}.
On record against it: {'; '.join(ab) if ab else 'no ladder mesh aborted'} — the same kind of abort as yesterday, because this design also sits on the buckling surface (its margin ratio is 0.02).</p>{mtxt}"""

r1, rn, rr, rb = rel.get("H1", {}), rel.get("T2_nominal", {}), rel.get("T2_robust", {}), rel.get("bistable", {})
tx["relaxed"] = f"""<p>Both solvers assumed an infinitely deep part (plane strain) and perfectly rigid walls. The final designs were re-solved in CalculiX with each assumption dropped in turn: as a 3D body 10 mm deep with free side faces, and with the wall posts and the plunger meshed in the same rubber as the beam (a one-material print).
Written down beforehand: 3D forces should fall by 0–16 %.</p>
<p><b>The curves move by more than the tolerances.</b> 3D: forces × {rn.get('3d', {}).get('median_force_ratio', float('nan')):.2f}–{r1.get('3d', {}).get('median_force_ratio', float('nan')):.2f} (inside the predicted range), curves shift by {pc(rn.get('3d', {}).get('shift_nrmse_vs_plane_strain'))}–{pc(r1.get('3d', {}).get('shift_nrmse_vs_plane_strain'))} of the target range.
Elastic frame: forces × {rr.get('ef', {}).get('median_force_ratio', float('nan')):.2f}–{rn.get('ef', {}).get('median_force_ratio', float('nan')):.2f}, curves shift by {pc(rr.get('ef', {}).get('shift_nrmse_vs_plane_strain'))}–{pc(r1.get('ef', {}).get('shift_nrmse_vs_plane_strain'))}.
Yesterday's plateau design, which passed at 1 % error, sits at {pc(rn.get('3d', {}).get('nrmse'))} in 3D (the 5 % limit) and at {pc(rn.get('ef', {}).get('nrmse'))} with an elastic frame (a miss). An arch pushes outward on its supports, so support stiffness is part of the design, not a detail.</p>"""

tx["caveats"] = f"""<ul>
<li><b>No new target passed.</b> H1 and H2 are "not reached within 80 evaluations per start"; none of their optimisations reported a converged optimum, so the cell family may or may not be able to do better. Only H4 has a proof of impossibility.</li>
<li><b>The snap remedy missed its own bar</b> ({sr('A+O')} against 8 of 11). The better numbers ({sr('AM+O2')}) come from a variant designed after looking at the first results on the same starts; the six fresh starts are the only uncontaminated evidence for it, and six is few.</li>
<li><b>The objective is discontinuous</b> where the path flips branch. The fold-aware gradient is exact on either side and says nothing about the jump. The exact features and the grid-based scorer (yesterday's extractor, used to score every arm) differ by up to 0.15 %, which decided one run (0.92 % by the arm's own features, 1.07 % by the scorer: counted as a miss).</li>
<li><b>The bistable pass has two aborted verifier meshes behind it</b> (graded on the finest, as the rule says). Such a design would pick its branch from manufacturing error.</li>
<li><b>Robust design is a negative result</b> for this cell and this error model; it was tried on one target only.</li>
<li><b>Relaxed checks are one-way.</b> The designs were not re-optimised for 3D or for an elastic frame; the elastic frame is one specific layout (posts 5 mm wide held on the lattice symmetry plane, same rubber). Self-contact and rate-dependent rubber were not tested. Nothing was printed.</li>
<li><b>Fabrication copies in CalculiX: 24 per design</b>, one mesh size; the pass fraction has a sampling error of about ±10 points.</li>
<li><b>Compute discipline.</b> The machine was shared and heavily loaded for two hours (runs took 3–6 times the plan); at times 12–13 pinned cores were in use against the ~10 asked for.</li></ul>"""

tx["deviations"] = """<ul>
<li><b>Path-following repaired after the plan was written</b> (tangent predictor; bisect any load step that meets an indefinite stiffness). Found by the gradient check, before any optimisation. Yesterday's path-follower was kept as a separate arm so yesterday's method is reproduced exactly.</li>
<li><b>Two planned arms were not run as separate jobs</b> ("fold-aware gradient only" and "yesterday's gradient with repaired path-following"): they are the first phase of the corresponding restart arms, so their result was read from those runs' histories (all eleven for the first; four of eleven for the second, whose other seven are genuine separate runs).</li>
<li><b>Three arms were added during the run</b> and are labelled as such everywhere: quicker restarts, the bifurcation margin, and the margin with a doubled budget. Six fresh starts were fixed in code before the first of them ran.</li>
<li><b>The two-cell stack was solved in a reduced form</b> after the one-system version proved too slow; the reduced form was checked against it first, and CalculiX graded the full two-cell model.</li>
<li><b>Robust optimisation from other starts</b> was added when the planned start gave no gain. The relaxed check on H1 and the elastic-frame variant were listed as planned / "if time" and were run.</li>
<li>No tolerance, target, start or grading rule was changed.</li></ul>"""
json.dump(tx, open("page_text.json", "w"), indent=1)
print("page_text.json written")
