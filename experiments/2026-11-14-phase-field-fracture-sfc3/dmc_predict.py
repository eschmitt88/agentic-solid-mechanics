"""Blind prediction for the Damage Mechanics Challenge geometry + calibration/validation figures.
Writes results/dmc_prediction.json, results/dmc_prediction_Q1.txt, results/dmc_prediction_crack_surface.csv and figures."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_pf import C, ORDER, SEQ
import dmc_data
from dmc import GEOMS, W, Hh, XS
from dmc_post import res, crack_surface, calib_table

R = "results/dmc"
fit = json.load(open(f"{R}/calibration_fit.json"))
con = json.load(open(f"{R}/contact_fit.json"))
exp = json.load(open(f"{R}/calibration_data_summary.json"))
base = fit["base l=1.0 h=0.5"]; s = base["load_scale"]


def total_disp(P, dFE):
    return con["toe_mean"] + dFE + con["a"] * np.maximum(P, 1e-9) ** con["n"]


cand_vals = [v["CH_P_pred"] for v in fit.values()]
F_ENS = float(np.mean(cand_vals) / base["CH_P_pred"])     # point prediction = mean over the re-calibrated model variants


def model_curve(geom, suffix="", scale=None):
    """Returns P, structural displacement, measured-style displacement. The test is displacement controlled through a
    soft contact, so after the peak the record drops vertically until it meets the post-peak branch again."""
    r = res(geom, suffix=suffix) or res(geom, 1.0, 0.5, 1, "_slab_full") or res(geom)
    sc = s if scale is None else scale
    rec = r["rec"]; P = rec[:, 1] * sc; dFE = rec[:, 0] * sc
    dT = total_disp(P, dFE)
    i = int(P.argmax())
    keep = np.ones(len(P), bool); keep[i + 1:] = dT[i + 1:] > dT[i]
    return P[keep], dFE[keep], dT[keep]


# ---------------- prediction numbers
ch = res("CH", suffix="_full") or res("CH")
P, dFE, dT = model_curve("CH", "_full", scale=s * F_ENS)
ip = int(P.argmax())
P_pred = float(P[ip])
cands = {k: v["CH_P_pred"] for k, v in fit.items()}
num_spread = float(np.std(list(cands.values()), ddof=1) / P_pred) if len(cands) > 1 else 0.0
u_model = base["rms_log_misfit"]                         # scatter of the 4 calibration geometries about the fit
u_num = max(num_spread, 0.02)
u_cal = float(np.mean([exp[g]["P_std"] / exp[g]["P_mean"] for g in exp]) / 2.0)   # sd of a 4-beam mean, per geometry
u_mean = float(np.sqrt(u_model ** 2 + u_num ** 2 + u_cal ** 2))
cov_spec = float(np.mean([exp[g]["P_std"] / exp[g]["P_mean"] for g in exp]))
u_single = float(np.sqrt(u_mean ** 2 + cov_spec ** 2))
k_tot = float(np.polyfit(dT[(P > 0.3 * P_pred) & (np.arange(len(P)) < ip) & (P < 0.8 * P_pred)], P[(P > 0.3 * P_pred) & (np.arange(len(P)) < ip) & (P < 0.8 * P_pred)], 1)[0])
d_peak = float(dT[ip])
u_dpeak = float(np.sqrt((con["amp_cov"] * con["a"] * P_pred ** con["n"]) ** 2 + con["toe_std"] ** 2 + (u_mean * dFE[ip]) ** 2))

# crack surface
X = crack_surface(ch, 0.6); ys, zs = ch["ys"], ch["zs"]
def at(j, z):
    k = int(np.argmin(np.abs(zs - z))); return None if np.isnan(X[j, k]) else round(float(X[j, k]), 1)
faces = {"front (y=-6.35)": 0, "mid-width (y=0)": len(ys) // 2, "back (y=+6.35)": len(ys) - 1}
trace = {name: {f"z={z:.1f}": at(j, z) for z in (-20.0, -17.5, -15.0, -12.5, -10.0, -7.5, -5.0, -2.5, -1.0)} for name, j in faces.items()}
top_k = [k for k in range(len(zs)) if zs[k] > -3.0 and zs[k] < -0.9]
x_top = {name: round(float(np.nanmean(X[j, top_k])), 1) for name, j in faces.items()}
# post-peak: load at which crack front reaches given heights is not tracked; report drop
drop = float(P[min(ip + 1, len(P) - 1)] / P_pred)
_raw = ch["rec"]; _i = int(_raw[:, 1].argmax()); drop_struct = float(_raw[min(_i + 1, len(_raw) - 1), 1] / _raw[_i, 1])

pred = dict(
    challenge="Damage Mechanics Challenge (Purdue/LLNL/Sandia, information packet of 15 Jan 2022) — fallback for SFC3",
    status="FROZEN blind prediction, made 2026-11-14/15 (UTC) from calibration data only; not graded",
    Q1_load_displacement=dict(
        peak_load_N=round(P_pred, 1),
        peak_load_1sigma_N=round(u_mean * P_pred, 1),
        peak_load_80pct_interval_mean_of_specimens_N=[round(P_pred * (1 - 1.28 * u_mean), 0), round(P_pred * (1 + 1.28 * u_mean), 0)],
        peak_load_80pct_interval_single_specimen_N=[round(P_pred * (1 - 1.28 * u_single), 0), round(P_pred * (1 + 1.28 * u_single), 0)],
        displacement_at_peak_mm=round(d_peak, 3), displacement_at_peak_1sigma_mm=round(u_dpeak, 3),
        loading_stiffness_30_80pct_N_per_mm=round(k_tot, 0),
        structural_only_displacement_at_peak_mm=round(float(dFE[ip]), 3),
        post_peak="abrupt (unstable): the structural solution falls to %.0f %% of peak within one increment of %.4f mm; in the measured record this is a vertical drop to about %.0f %% of peak followed by a low tail" % (100 * drop_struct, float(dFE[1] - dFE[0]), 100 * drop),
        curve_file="results/dmc_prediction_Q1.txt"),
    Q2_crack_path=dict(
        initiation="at the notch root; first on the deep (front, y=-6.35) side of the tapered notch",
        crack_x_mm_at_heights=trace,
        crack_x_mm_near_top_surface=x_top,
        description="single crack; twists from the 26.6-degree skewed notch plane towards a plane normal to the beam axis and drifts towards the loading rod (x=0)",
        surface_file="results/dmc_prediction_crack_surface.csv"),
    Q3_surface_displacement_fields="not submitted (the solver output exists but was not post-processed into the requested 100-micron grids)",
    model=dict(method="AT2 phase-field fracture, hybrid formulation (isotropic degradation, spectral tensile driving force, history-field irreversibility), staggered to convergence",
               solver="own JAX code pf3d.py, hex8 tensor grid, GPU Jacobi-PCG", E_MPa=ch["meta"]["E"], nu=ch["meta"]["nu"],
               l_mm=ch["meta"]["l"], Gc_N_per_mm=round(base["Gc_calibrated"], 4), h_mm=ch["meta"]["h"], elements=ch["meta"]["ne"],
               contact_compliance="delta = %.3f + delta_FE + %.4f * P^%.3f (mm, N), fitted to the 16 calibration beams" % (con["toe_mean"], con["a"], con["n"])),
    calibration_fit={g: dict(model_N=round(v["P_model_scaled"], 1), measured_mean_N=round(v["P_exp"], 1), error_pct=round(v["err_pct"], 1)) for g, v in base["calib"].items()},
    sensitivity_peak_load_N={k: round(v, 1) for k, v in cands.items()},
    uncertainty=dict(model_form_rel=round(u_model, 3), numerical_rel=round(u_num, 3), calibration_mean_rel=round(u_cal, 3), specimen_scatter_cov=round(cov_spec, 3),
                     total_rel_for_mean=round(u_mean, 3), total_rel_single_specimen=round(u_single, 3)),
)
json.dump(pred, open("results/dmc_prediction.json", "w"), indent=1)
with open("results/dmc_prediction_Q1.txt", "w") as f:
    f.write("Point Number\tForce (Newtons)\tDisplacement (mm)\n")
    keep = dT > 0
    for i, (p, d) in enumerate(zip(P[keep], dT[keep]), 1):
        f.write(f"{i}\t{p:.2f}\t{d:.4f}\n")
with open("results/dmc_prediction_crack_surface.csv", "w") as f:
    f.write("# predicted crack surface, challenge coordinates, mm: y,z,x\n")
    for j in range(len(ys)):
        for k in range(len(zs)):
            if not np.isnan(X[j, k]):
                f.write(f"{ys[j]:.3f},{zs[k]:.3f},{X[j,k]:.3f}\n")
print(json.dumps(pred["Q1_load_displacement"], indent=1)); print(json.dumps(pred["Q2_crack_path"]["crack_x_mm_near_top_surface"])); print(pred["uncertainty"], pred["sensitivity_peak_load_N"])

# ---------------- figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e4e0",
                     "grid.linewidth": 0.6, "figure.dpi": 130, "savefig.bbox": "tight", "axes.titlesize": 11, "lines.linewidth": 1.8})
names = {"HC": "HC — centre notch (mode I)", "HB": "HB — notch 9.5 mm off centre", "HA": "HA — notch 19 mm off centre", "H45": "H45 — centre notch rotated 45°"}
fig, axs = plt.subplots(1, 5, figsize=(17, 3.7), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.1]))
for ax, g in zip(axs, names):
    for nme in dmc_data.BEAMS[g]:
        d, Pm = dmc_data.load_ld(nme); ax.plot(d, dmc_data.smooth(Pm), color="#b9b8b0", lw=1.0)
    Pm, dfe, dt = model_curve(g, "_full" if g == "H45" else "")
    ax.plot(dt[dt > 0], Pm[dt > 0], color=C["blue"]); ax.set_title(names[g], fontsize=9.5); ax.set_xlabel("displacement [mm]"); ax.set_xlim(0, 0.8); ax.set_ylim(0, 450)
axs[0].set_ylabel("load [N]")
axs[0].plot([], [], color="#b9b8b0", label="4 lab tests"); axs[0].plot([], [], color=C["blue"], label="calibrated model"); axs[0].legend(frameon=False, fontsize=8, loc="upper left")
ax = axs[4]
for i, g in enumerate(names):
    v = base["calib"][g]
    ax.errorbar([i - 0.12], [v["P_exp"]], yerr=[v["P_exp_std"]], fmt="o", color=C["mute"], capsize=3, ms=6)
    ax.plot([i + 0.12], [v["P_model_scaled"]], "s", color=C["blue"], ms=7)
    ax.text(i + 0.2, v["P_model_scaled"], f"{v['err_pct']:+.0f}%", fontsize=8, va="center", color=C["ink"])
ax.set_xticks(range(4)); ax.set_xticklabels(list(names)); ax.set_xlim(-0.5, 3.8); ax.set_ylim(150, 450); ax.set_title("Peak load: tests (mean ± sd) vs model", fontsize=9.5)
fig.suptitle("Calibration (not blind): one fracture energy fitted to four notch geometries", fontsize=11)
fig.tight_layout(); fig.savefig("results/fig_dmc_calibration.png"); plt.close(fig)

# crack path validation on calibration beams (laser profilometry)
SF = os.environ.get("DMC_SF_DIR", "/tmp/dmc_work/FractureSurface")
path_metrics = {}
fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
variants = [((1.0, 0.5), C["blue"], "model, l=1, h=0.5 (prediction mesh)"), ((1.0, 0.25), C["aqua"], "model, l=1, h=0.25"),
            ((1.0, 0.125), C["violet"], "model, l=1, h=0.125"), ((0.5, 0.125), C["magenta"], "model, l=0.5, h=0.125")]
for ax, (g, xn) in zip(axs, (("HB", 9.525), ("HA", 19.05))):
    tops = []
    for nme in dmc_data.BEAMS[g]:
        a_ = np.genfromtxt(f"{SF}/{nme}_SF.csv", delimiter=","); zrow = np.arange(a_.shape[0]) * 0.1
        prof = np.median(a_, axis=1); ax.plot(prof[56:236] - prof[56], zrow[56:236], color="#b9b8b0", lw=1.0)
        tops.append(float(prof[225] - prof[56]))
    path_metrics[g] = dict(measured_lateral_shift_at_22p5mm=float(np.mean(tops)), model={})
    for (l_, h_), col, lab in variants:
        r = res(g, l_, h_, 1, "_slab_full")
        if r is None:
            continue
        Xm = crack_surface(r, 0.6)[0]; zz = r["zs"] + Hh; m = ~np.isnan(Xm) & (zz > 5.3)
        ax.plot(Xm[m] - xn, zz[m], color=col, lw=1.8, label=lab)
        k = int(np.argmin(np.abs(zz - 22.5)))
        path_metrics[g]["model"][f"l={l_},h={h_}"] = None if np.isnan(Xm[k]) else float(Xm[k] - xn)
    ax.plot([], [], color="#b9b8b0", label="4 lab tests (laser scan)")
    ax.set_title(f"{g}: crack path from the notch root", fontsize=10)
    ax.set_xlabel("sideways shift towards the load rod [mm]"); ax.set_ylabel("height above bottom face [mm]"); ax.set_ylim(0, 25.4); ax.set_xlim(-11.5, 1)
axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
ax = axs[2]
for nme in dmc_data.BEAMS["H45"]:
    a_ = np.genfromtxt(f"{SF}/{nme}_SF.csv", delimiter=","); zrow = np.arange(a_.shape[0]) * 0.1
    tw = np.abs(a_[:, 112] - a_[:, 8]); ax.plot(tw[56:236], zrow[56:236], color="#b9b8b0", lw=1.0)
r = res("H45", suffix="_full") or res("H45")
Xm = crack_surface(r, 0.6); zz = r["zs"] + Hh
j0 = int(np.argmin(np.abs(r["ys"] + 5.2))); j1 = int(np.argmin(np.abs(r["ys"] - 5.2)))
tw = np.abs(Xm[j1] - Xm[j0]); m = ~np.isnan(tw) & (zz > 5.3)
ax.plot(tw[m], zz[m], color=C["blue"], lw=1.8)
ax.set_title("H45: how fast the crack untwists\n(sideways offset between the two faces of the crack)", fontsize=9.5)
ax.set_xlabel("offset across 10.4 mm of width [mm]"); ax.set_ylabel("height above bottom face [mm]"); ax.set_ylim(0, 25.4)
fig.suptitle("Crack-path check against laser-scanned fracture surfaces of the calibration beams (not blind)", fontsize=11)
fig.tight_layout(); fig.savefig("results/fig_dmc_crack_path_validation.png"); plt.close(fig)
json.dump(path_metrics, open(f"{R}/crack_path_validation.json", "w"), indent=1)
# empirical correction of the mesh-biased upper crack path: mean (measured - model on the prediction mesh) on HB and HA at 22.5 mm height
_c = [path_metrics[g]["measured_lateral_shift_at_22p5mm"] - path_metrics[g]["model"]["l=1.0,h=0.5"] for g in ("HB", "HA")]
corr = -float(np.mean(_c))          # mm towards the loading rod
pred = json.load(open("results/dmc_prediction.json"))
xt_raw = pred["Q2_crack_path"]["crack_x_mm_near_top_surface"]
pred["Q2_crack_path"]["crack_x_mm_near_top_surface_raw_model"] = xt_raw
pred["Q2_crack_path"]["top_exit_bias_correction_mm_towards_load"] = round(corr, 2)
pred["Q2_crack_path"]["top_exit_correction_basis"] = {g: dict(measured=round(path_metrics[g]["measured_lateral_shift_at_22p5mm"], 2), model=path_metrics[g]["model"]) for g in ("HB", "HA")}
pred["Q2_crack_path"]["crack_x_mm_near_top_surface"] = {k: round(v + corr, 1) for k, v in xt_raw.items()}
pred["Q2_crack_path"]["crack_x_top_range_mm"] = [round(min(xt_raw.values()) - 0.3, 1), round(max(xt_raw.values()) + max(-c for c in _c) + 0.5, 1)]
pred["Q2_crack_path"]["crack_x_mm_at_heights_note"] = "raw model values; reliable up to about z = -12 mm, above that biased towards vertical (see correction)"
json.dump(pred, open("results/dmc_prediction.json", "w"), indent=1)
print("top exit corrected:", pred["Q2_crack_path"]["crack_x_mm_near_top_surface"], pred["Q2_crack_path"]["crack_x_top_range_mm"], "corr", corr)
print("path metrics", json.dumps(path_metrics))

# challenge prediction figure
fig = plt.figure(figsize=(15, 4.6))
ax = fig.add_subplot(1, 3, 1)
keep = dT > 0
ax.fill_between(dT[keep][: ip + 1], P[keep][: ip + 1] * (1 - 1.28 * u_single), P[keep][: ip + 1] * (1 + 1.28 * u_single), color="#cfe0f6", lw=0, label="80 % band, single specimen")
ax.plot(dT[keep], P[keep], color=C["blue"], label="blind prediction")
ax.plot([d_peak], [P_pred], "o", color=C["blue"], ms=7); ax.annotate(f"{P_pred:.0f} N at {d_peak:.2f} mm", (d_peak, P_pred), xytext=(d_peak - 0.33, P_pred + 22), fontsize=9)
ax.set_xlabel("load-point displacement [mm]"); ax.set_ylabel("load [N]"); ax.set_xlim(0, 0.9); ax.set_ylim(0, 420); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Challenge geometry: predicted load–displacement")
d = np.nan_to_num(ch["d"], nan=0.0)
for n, (j, lab) in enumerate(((0, "front face (y = −6.35 mm), deep notch end"), (len(ys) - 1, "back face (y = +6.35 mm), shallow notch end"))):
    ax = fig.add_subplot(2, 3, 2 + 3 * n)
    ax.pcolormesh(ch["xs"], zs, np.clip(d[:, j, :], 0, 1).T, cmap=SEQ, vmin=0, vmax=1, shading="gouraud", rasterized=True)
    g = GEOMS["CH"]; xn = g["xf"] if j == 0 else g["xb"]; dep = g["df"] if j == 0 else g["db"]
    ax.add_patch(plt.Rectangle((xn - 0.56, -Hh), 1.12, dep, fc="white", ec="#888", lw=0.8))
    ax.plot([0], [0.6], "v", color=C["ink"], ms=7, clip_on=False); ax.plot([-XS, XS], [-Hh - 0.6] * 2, "^", color=C["ink"], ms=7, clip_on=False)
    ax.set_xlim(-38.1, 38.1); ax.set_ylim(-Hh, 0); ax.set_aspect("equal"); ax.grid(False); ax.set_title(lab, fontsize=9); ax.set_ylabel("z [mm]")
    if n == 1:
        ax.set_xlabel("x [mm]")
    else:
        ax.set_xticklabels([])
ax = fig.add_subplot(1, 3, 3)
cm = plt.get_cmap("viridis")
for k in range(len(zs)):
    if zs[k] > -Hh + 2.6 and not np.all(np.isnan(X[:, k])) and k % 3 == 0:
        ax.plot(X[:, k], ys, color=SEQ(0.25 + 0.75 * (zs[k] + Hh) / Hh), lw=1.4)
g = GEOMS["CH"]; ax.plot([g["xf"], g["xb"]], [-W / 2, W / 2], color=C["orange"], lw=2.2, label="notch (plan view)")
ax.axvline(0, color=C["ink"], lw=1, ls=":"); ax.text(0.2, -5.9, "load rod", fontsize=8, rotation=90)
_q2 = json.load(open("results/dmc_prediction.json"))["Q2_crack_path"]
if "crack_x_top_range_mm" in _q2:
    lo_, hi_ = _q2["crack_x_top_range_mm"]; xc_ = _q2["crack_x_mm_near_top_surface"]["mid-width (y=0)"]
    ax.axvspan(lo_, hi_, color="#cfe0f6", alpha=0.6, lw=0); ax.axvline(xc_, color=C["blue"], lw=1.6, ls="--")
    ax.text(xc_ + 0.15, 4.2, "predicted exit\nat the top\n(bias-corrected)", fontsize=7.5, color=C["ink"])
ax.set_xlabel("x [mm]"); ax.set_ylabel("y through the width [mm]"); ax.set_xlim(-15, 3); ax.set_title("Predicted crack seen from above\n(light = near notch, dark = near top)", fontsize=9.5); ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.set_aspect("equal")
fig.suptitle("BLIND PREDICTION — Damage Mechanics Challenge geometry (awaiting grading)", fontsize=12)
fig.tight_layout(); fig.savefig("results/fig_dmc_prediction.png"); plt.close(fig)
print("figures written")
