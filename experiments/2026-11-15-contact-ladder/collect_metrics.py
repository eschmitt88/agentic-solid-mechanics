"""Grade the pre-registered checks from the result files and write metrics.json."""
import json, os, glob
import numpy as np
import rung2_ironing_own as RI

R = "results/"
def j(f):
    return json.load(open(R + f)) if os.path.exists(R + f) else None

checks = []
def add(cid, what, value, ref, err, tol, passed, solver, note=""):
    checks.append(dict(id=cid, what=what, value=value, reference=ref, error=err, tolerance=tol, passed=bool(passed) if passed is not None else None, solver=solver, note=note))

# ---------------- rung 1
po, pc = j("r1_patch_own.json"), j("r1_patch_ccx.json")
v = max(po["same_material"]["p_dev"], po["same_material"]["syy_dev"])
add("1a-own", "patch test: max deviation of contact pressure / stress", v, 0.0, v, 0.01, v <= 0.01, "own", "depends on quadrature: 7.5 % with 2 points per edge, 0.03 % with 20")
s2s = [c for c in pc if c["ctype"] == "SURFACE TO SURFACE"][0]; v = max(s2s["p_dev"], s2s["syy_dev"])
add("1a-ccx", "patch test: max deviation of contact pressure / stress", v, 0.0, v, 0.01, v <= 0.01, "ccx")
cyl, sph, cx = j("r1_cyl_own.json"), j("r1_sphere_own.json"), j("r1_hertz_ccx.json")
c3, s3 = cyl[-1], sph[-1]
cc = [c for c in cx if c.get("ctype") == "SURFACE TO SURFACE" and c.get("kfac") == 50.0 and c.get("level") == 3]
c2d = [c for c in cc if c["ref"]["F"] > 1][0]; cax = [c for c in cc if c["ref"]["F"] < 1][0]
for tag, r in (("own", c3), ("ccx", c2d)):
    add(f"1b-a-{tag}", "2D Hertz contact half-width", r["a"], r["ref"]["a"], r["err_a"], 0.05, abs(r["err_a"]) <= 0.05 and r["h_c"] <= r["ref"]["a"] / 15, tag, f"resolution ±{r['a_resolution']/2/r['ref']['a']*100:.1f} %")
    add(f"1b-p0-{tag}", "2D Hertz peak pressure", r["p0"], r["ref"]["p0"], r["err_p0"], 0.03, abs(r["err_p0"]) <= 0.03, tag)
    add(f"1b-dist-{tag}", "2D Hertz pressure distribution, relative L2 error", r["l2"], 0.0, r["l2"], 0.05, r["l2"] <= 0.05, tag)
add("1c-F-own", "sphere: force at prescribed approach", s3["F"], s3["ref"]["F"], s3["err_F"], 0.03, abs(s3["err_F"]) <= 0.03, "own", "the 2.6 % is the finite sphere (scales with a/R), not solver error; passes with 0.4 % to spare")
add("1c-F-ccx", "sphere: force at prescribed approach", cax["F"], cax["ref"]["F"], cax["err_F"], 0.03, abs(cax["err_F"]) <= 0.03, "ccx", "same finite-size effect")
add("1c-a-own", "sphere: contact radius", s3["a"], s3["ref"]["a"], s3["err_a"], 0.05, abs(s3["err_a"]) <= 0.05, "own")
add("1c-a-ccx", "sphere: contact radius", cax["a"], cax["ref"]["a"], cax["err_a"], 0.05, abs(cax["err_a"]) <= 0.05, "ccx")
add("1c-p0-own", "sphere: peak pressure", s3["p0"], s3["ref"]["p0"], s3["err_p0"], 0.03, abs(s3["err_p0"]) <= 0.03, "own")
add("1c-p0-ccx", "sphere: peak pressure", cax["p0"], cax["ref"]["p0"], cax["err_p0"], 0.03, abs(cax["err_p0"]) <= 0.03, "ccx",
    f"CalculiX contact output is wrong on the nodes next to the symmetry axis (max {cax['p0_max']:.0f} vs Hertz {cax['ref']['p0']:.1f}); away from the axis it is within 1.5 %")
e0, e3 = abs(cyl[0]["err_p0"]), abs(cyl[-1]["err_p0"]); f0, f3 = abs(sph[0]["err_F"]), abs(sph[-1]["err_F"])
add("1d", "mesh convergence over 4 levels (own): 2D peak-pressure error and sphere force error, coarsest -> finest", [e0, e3, f0, f3], None, None, "finest < coarsest and in tolerance",
    e3 < e0 and e3 <= 0.03 and f3 < f0 and f3 <= 0.03, "own", "sphere force error barely moves (2.68 -> 2.62 %) because it is not a mesh error")
d1 = c2d["p0"] / c3["p0"] - 1; d2 = cax["F"] / s3["F"] - 1
add("1e", "own vs CalculiX on the same mesh: 2D peak pressure, sphere force", [d1, d2], None, max(abs(d1), abs(d2)), 0.02, max(abs(d1), abs(d2)) <= 0.02, "both")

# ---------------- rung 2a Cattaneo-Mindlin
own = None
for f in ("r2_mindlin_own_3-15.json", "r2_mindlin_own_2-15_3-15.json", "r2_mindlin_own_2-15.json"):
    d = j(f)
    if d:
        own = d[-1]; break
ccxg = j("r2_mindlin_ccx_graded.json"); ccx2 = j("r2_mindlin_ccx_sweep.json")
cg = ccxg[0] if ccxg and ccxg[0].get("loads") else None
def grade_m(tag, r, note=""):
    if r is None:
        for cid, what, tol in (("2a", "stick-zone half-width at Q/(fP) = 0.25, 0.5, 0.75", "0.05 a"), ("2b", "tangential traction at the centre", 0.05), ("2c", "traction distribution L2 at Q/(fP) = 0.5", 0.08)):
            add(f"{cid}-{tag}", what, None, None, None, tol, False, tag, note or "no result")
        return
    L = r["loads"]
    add(f"2a-{tag}", "stick-zone half-width at Q/(fP) = 0.25, 0.5, 0.75 (error / a)", [l["c"] / 0.05 for l in L], [l["c_ref"] / 0.05 for l in L], [l["c_err_over_a"] for l in L], "0.05 a", all(abs(l["c_err_over_a"]) <= 0.05 for l in L), tag, note)
    add(f"2b-{tag}", "tangential traction at the centre, three loads", [l["q0"] for l in L], [l["q0_ref"] for l in L], [l["q0_err"] for l in L], 0.05, all(abs(l["q0_err"]) <= 0.05 for l in L), tag, note)
    add(f"2c-{tag}", "traction distribution, relative L2 error at Q/(fP) = 0.5", L[1]["l2"], 0.0, L[1]["l2"], 0.08, L[1]["l2"] <= 0.08, tag, note)
grade_m("own", own, f"mesh level {own['level']} (h = a/{int(round(0.05/own['h_c']))})" if own else "")
grade_m("ccx", cg, "registered configuration (level 3) did not complete: 61 min, 8 increments, 62 divergence messages, stopped by me" if cg is None else "")
if own and cg:
    d = cg["loads"][1]["q0"] / own["loads"][1]["q0"] - 1
    add("2d", "own vs CalculiX: centre traction at Q/(fP) = 0.5", d, 0.0, d, 0.05, abs(d) <= 0.05, "both")
else:
    add("2d", "own vs CalculiX: centre traction at Q/(fP) = 0.5", None, None, None, 0.05, False, "both", "no CalculiX result at the registered level; at level 2 the two differ by "
        + (f"{(ccx2[0]['loads'][1]['q0']/j('r2_mindlin_own_2-15.json')[-1]['loads'][1]['q0']-1)*100:+.1f} %" if ccx2 and j('r2_mindlin_own_2-15.json') else "n/a"))

# ---------------- rung 2 ironing
bands = dict(Fv_t1=(2.85e8, 3.35e8), Fv_mid=(3.55e8, 4.23e8), Ft_t1=(0.55e8, 0.74e8), Ft_mid=(1.40e8, 2.00e8))
ownf = [f for f in ("r2_ironing_own_h0.05.json", "r2_ironing_own_h0.1_half.json", "r2_ironing_own_h0.1.json") if j(f)]
so = RI.summary(j(ownf[0])["hist"]) if ownf else None
inb = lambda s: {k: bool(bands[k][0] <= s[k] <= bands[k][1]) for k in bands}
if so:
    add("2e-own", f"ironing forces inside the published band ({ownf[0]})", {k: so[k] for k in bands}, {k: list(v) for k, v in bands.items()}, inb(so), "in band (4 numbers)", all(inb(so).values()), "own",
        "tangential force during sliding is below the lowest published curve")
cgr = j("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.05_graded.json")
if cgr is not None and cgr["t_reached"] >= 1.6:
    sc = RI.summary(cgr["hist"])
    add("2e-ccx", "ironing forces inside the published band (registered CalculiX configuration)", {k: sc[k] for k in bands}, {k: list(v) for k, v in bands.items()}, inb(sc), "in band (4 numbers)", all(inb(sc).values()), "ccx")
else:
    sc = None
    add("2e-ccx", "ironing forces inside the published band (registered CalculiX configuration)", None, {k: list(v) for k, v in bands.items()}, None, "in band (4 numbers)", False, "ccx",
        f"registered configuration fails at t = {cgr['t_reached']:.3f} ('too many cutbacks')" if cgr else "not run")
if so and sc:
    d = {k: sc[k] / so[k] - 1 for k in bands}
    add("2f", "ironing, own vs CalculiX (registered configuration)", d, None, max(abs(v) for v in d.values()), 0.05, all(abs(v) <= 0.05 for v in d.values()), "both")
else:
    alt = j("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.05_dm0.001.json") or j("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.0005.json") or j("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.001.json")
    note = "no CalculiX result in the registered configuration"
    if alt and alt["t_reached"] >= 1.6 and so:
        sa = RI.summary(alt["hist"]); note += f"; with {alt['dmax2']} increments at h = {alt['h']} CalculiX gives " + ", ".join(f"{k} {sa[k]/so[k]-1:+.1%}" for k in bands) + " relative to own"
    add("2f", "ironing, own vs CalculiX (registered configuration)", None, None, None, 0.05, False, "both", note)
# 2g: mesh and step sensitivity of the own solver
s = {f: RI.summary(j(f)["hist"]) for f in ("r2_ironing_own_probe.json", "r2_ironing_own_h0.1.json", "r2_ironing_own_h0.1_half.json", "r2_ironing_own_h0.05.json") if j(f)}
if "r2_ironing_own_h0.1.json" in s and "r2_ironing_own_h0.1_half.json" in s:
    dstep = {k: s["r2_ironing_own_h0.1_half.json"][k] / s["r2_ironing_own_h0.1.json"][k] - 1 for k in bands}
    fine = "r2_ironing_own_h0.05.json" if "r2_ironing_own_h0.05.json" in s else None
    dmesh = {k: s[fine][k] / s["r2_ironing_own_h0.1_half.json"][k] - 1 for k in bands} if fine else None
    dcoarse = {k: s["r2_ironing_own_h0.1.json"][k] / s["r2_ironing_own_probe.json"][k] - 1 for k in bands}
    worst = max([abs(v) for v in dstep.values()] + ([abs(v) for v in dmesh.values()] if dmesh else [1.0]))
    add("2g", "ironing, own solver: change under step halving (h = 0.1) and under mesh refinement h = 0.1 -> 0.05", dict(step_halving=dstep, mesh_0p1_to_0p05=dmesh, mesh_0p2_to_0p1=dcoarse), None, worst, 0.03, worst <= 0.03, "own",
        "the coarser pair (0.2 -> 0.1) changes the sliding tangential force by more than 3 %")

# ---------------- rung 3 self-checks (the blind outcome itself is graded by the orchestrator)
p3 = j("prediction.json")
if p3:
    sc = p3["selfchecks"]
    add("3a", "reduced model vs CalculiX 3D: first bending frequency of the rod (Hz)", sc["3a"]["rom_hz"], sc["3a"]["ccx_hz"], sc["3a"]["err"], 0.05, sc["3a"]["passed"], "both")
    add("3b", "reduced model as registered (rigid flange) vs CalculiX 3D, static elastic joint: tip deflection, top-fastener force",
        [sc["3b"]["registered_model"]["tip"], sc["3b"]["registered_model"]["T1"]], [sc["3b"]["ccx_tip"], sc["3b"]["ccx_T1"]],
        [sc["3b"]["registered_model"]["err_tip"], sc["3b"]["registered_model"]["err_T1"]], 0.15, sc["3b"]["passed_as_registered"], "both",
        f"model rebuilt with a flange hinge and an effective bearing radius: {sc['3b']['recalibrated']['err_tip']:+.1%}, {sc['3b']['recalibrated']['err_T1']:+.1%}")
    add("3c", "energy balance of the explicit integration, worst of the 19 nominal tests", sc["3c"]["worst"], 0.0, sc["3c"]["worst"], 0.02, sc["3c"]["passed"], "own")
    add("3d", f"time-step halving, peak plastic elongation of fastener 1, test {sc['3d']['test']}", sc["3d"]["ep1_half_dt"], sc["3d"]["ep1_dt"], sc["3d"]["change"], 0.02, sc["3d"]["passed"], "own")

# ---------------- CalculiX characterisation (not graded)
char = dict(ironing=[], mindlin=[])
for f in sorted(glob.glob(R + "r2_ironing_ccx_*.json")):
    r = json.load(open(f)); sm = RI.summary(r["hist"]) if r["t_reached"] >= 1.6 else None
    char["ironing"].append(dict(file=os.path.basename(f), h=r["h"], contact=r["ctype"], K_over_E_h=r["kfac"], stick_over_K=r["stickfac"], max_increment_slide=r["dmax2"], finished=r["finished"],
                                t_reached=r["t_reached"], increments=r["increments"], iterations=r["iterations"], divergence_messages=r["divergence"], seconds=r["seconds"], summary=sm))
for f in ("r2_mindlin_ccx_probe.json", "r2_mindlin_ccx_sweep.json", "r2_mindlin_ccx_graded.json"):
    d = j(f)
    for r in (d or []):
        if "ctype" in r:
            char["mindlin"].append(dict(level=r["level"], contact=r["ctype"], K_over_E_h=r["kfac"], stick_over_K=r["stickfac"], increments_per_step=r["ninc"], finished=r["finished"], seconds=r["seconds"],
                                        iterations=r["iterations"], divergence_messages=r["divergence"], q_max_after_normal=r.get("q_max_after_normal"),
                                        loads=[dict(c_err_over_a=l["c_err_over_a"], q0_err=l["q0_err"], l2=l["l2"]) for l in r.get("loads", [])]))
# step-size extrapolation of the CalculiX ironing force at h = 0.1 (supplementary)
a, b = j("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.001.json"), j("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.0005.json")
extra = {}
if a and b and b["t_reached"] >= 1.6 and so:
    sa, sb = RI.summary(a["hist"]), RI.summary(b["hist"])
    own01 = s.get("r2_ironing_own_h0.1_half.json")
    extra = dict(ccx_Ft_mid_inc_0p001=sa["Ft_mid"], ccx_Ft_mid_inc_0p0005=sb["Ft_mid"], ccx_Ft_mid_linear_extrapolation_to_zero_increment=2 * sb["Ft_mid"] - sa["Ft_mid"],
                 own_Ft_mid_h0p1=own01["Ft_mid"] if own01 else None, own_Ft_mid_finest=so["Ft_mid"],
                 ccx_inc0p0005_vs_own_h0p1={k: sb[k] / own01[k] - 1 for k in bands} if own01 else None)
std = j("r2_ironing_own_h0.1_nhstd.json")
if std:
    extra["own_h0p1_textbook_neo_hookean"] = RI.summary(std["hist"])
for f, k in (("r2_ironing_own_h0.2_nhstd.json", "own_h0p2_textbook_neo_hookean"), ("r2_ironing_own_h0.2_eps20.json", "own_h0p2_penalty_x4"), ("r2_ironing_own_h0.2_ngp12.json", "own_h0p2_12_gauss_points"), ("r2_ironing_own_h0.2_half.json", "own_h0p2_steps_halved")):
    if j(f): extra[k] = RI.summary(j(f)["hist"])

npass = sum(1 for c in checks if c["passed"]); nmiss = sum(1 for c in checks if c["passed"] is False)
out = dict(summary=dict(checks=len(checks), passed=npass, missed=nmiss), checks=checks,
           ironing_summaries={k: v for k, v in s.items()}, ironing_supplementary=extra, calculix_characterisation=char)
if p3:
    out["rung3"] = dict(Q1=p3["Q1_min_dV_any_fastener"], Q2=p3["Q2_min_dV_all_fasteners"], selfchecks={k: v.get("passed", v.get("passed_as_registered")) for k, v in p3["selfchecks"].items()}, graded_by="orchestrator (blind)")
json.dump(out, open("metrics.json", "w"), indent=1)
for c in checks:
    print(f"{c['id']:10s} {'PASS' if c['passed'] else 'MISS'}  {c['what'][:70]:70s} err={c['error']} tol={c['tolerance']}")
print(out["summary"])
