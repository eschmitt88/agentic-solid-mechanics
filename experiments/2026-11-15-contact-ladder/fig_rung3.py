"""Rung 3 figures."""
import json, numpy as np, plotstyle
plt = plotstyle.setup(); C = plotstyle.C
R = "results/"
p = json.load(open(R + "prediction.json")); q = p["Q3_per_test"]
dv = np.array([r["dV_ms"] for r in q])
fig, ax = plt.subplots(figsize=(6.6, 5.0))
w = 0.45
ax.bar(dv - w / 2, [r["prob_any"] for r in q], width=w, color=C["blue"], label="at least one fastener breaks")
ax.bar(dv + w / 2, [r["prob_all"] for r in q], width=w, color=C["orange"], label="all six break")
q1 = p["Q1_min_dV_any_fastener"]
ax.axvspan(q1["p5"], q1["p95"], color="#e9f1fb", zorder=0); ax.axvline(q1["p50"], color=C["blue"], lw=1.2, ls="--")
ax.text(12.4, 1.13, f"first fracture: median {q1['p50']:.1f} m/s\n(5–95 %: {q1['p5']:.1f}–{q1['p95']:.1f})", fontsize=10, color=C["blue"], va="bottom")
for r in q:
    ax.text(r["dV_ms"], -0.075, str(r["n_fractured_nominal"]), ha="center", fontsize=8, color="#52514e")

ax.set_ylim(-0.13, 1.55); ax.set_xlim(12, 40.5); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("velocity change of the drop (m/s); one pair of bars per test\nsmall numbers: screws broken in the best-guess model"); ax.set_ylabel("fraction of model samples")
ax.set_title("Blind prediction: which drops break the screws"); ax.legend(loc="upper right", fontsize=9.5)
fig.tight_layout(); fig.savefig(R + "r3_prediction.png"); plt.close(fig)

# joint behaviour: pushover and static check
h = json.load(open(R + "r3_pushover_ccx.json"))["hist"]; hinge = json.load(open(R + "r3_flange_hinge.json"))
sc = p["selfchecks"]["3b"]
fig, axs = plt.subplots(2, 1, figsize=(6.6, 9.4))
axs[0].plot([d["rot_x"] * 1e3 for d in h], [d["F"] * 2.881 / 1e3 for d in h], color=C["orange"], label="CalculiX 3D, elastic-plastic flange")
e = json.load(open(R + "r3_pushover_ccx_elastic.json"))["hist"][-1]
xs = np.array([0, 40.0]); axs[0].plot(xs, xs * (e["F"] * 2.881 / 1e3) / (e["rot_x"] * 1e3), color=C["mute"], ls="--", label="elastic")
axs[0].axhline(5.56, color=C["bad"], lw=0.8); axs[0].text(22, 6.0, "flange starts to yield", color=C["bad"], fontsize=10)
axs[0].set_ylim(0, 26); axs[0].set_xlim(0, 40); axs[0].set_xlabel("rotation of the rod, 2 in from the block (mrad)"); axs[0].set_ylabel("moment at the joint (1000 in·lbf)")
axs[0].set_title("Static push on the joint: the flange bends first"); axs[0].legend(fontsize=9.5, loc="lower right")
labs = ["tip deflection", "top-fastener force"]
x = np.arange(2)
axs[1].bar(x - 0.27, [1, 1], width=0.25, color=C["orange"], label="CalculiX 3D")
axs[1].bar(x, [1 + sc["registered_model"]["err_tip"], 1 + sc["registered_model"]["err_T1"]], width=0.25, color=C["bad"], label="reduced model as registered (rigid flange)")
axs[1].bar(x + 0.27, [1 + sc["recalibrated"]["err_tip"], 1 + sc["recalibrated"]["err_T1"]], width=0.25, color=C["blue"], label="reduced model with flange hinge")
axs[1].axhspan(0.85, 1.15, color="#e6e5df", zorder=0); axs[1].set_xticks(x); axs[1].set_xticklabels(labs); axs[1].set_ylim(0, 1.45)
axs[1].set_ylabel("relative to CalculiX 3D"); axs[1].set_title("The check the first model failed"); axs[1].legend(fontsize=9, loc="upper center"); axs[1].set_ylim(0, 1.75)
fig.tight_layout(); fig.savefig(R + "r3_joint.png"); plt.close(fig)

# nominal histories of two tests
d = np.load(R + "r3_nominal_hist.npz"); t = d["t"] * 1e3
import smc_rom as S
fig, axs = plt.subplots(3, 1, figsize=(6.6, 11.0))
for i, col, lab in ((7, C["blue"], "test 8 (77.9 ft/s): no fracture"), (9, C["orange"], "test 10 (85.2 ft/s): five break")):
    A, W, n = S.pulse_params(*S.TESTS[i, 1:])
    x = np.clip(d["t"] / W, 0, 1); axs[0].plot(t, A / S.G * (0.5 * (1 - np.cos(2 * np.pi * x))) ** n / 1e3, color=col, label=lab)
    axs[1].plot(t, d["T"][:, i, 0] / 1e3, color=col); axs[1].plot(t, d["T"][:, i, 3] / 1e3, color=col, ls=":")
    axs[2].plot(t, d["vt"][:, i] * 25.4, color=col)
axs[0].set_title("Shock on the block"); axs[0].set_ylabel("acceleration (1000 g)"); axs[0].legend(fontsize=9)
axs[1].set_title("Fastener tension (solid: top, dotted: bottom)"); axs[1].set_ylabel("1000 lbf")
axs[2].set_title("Tip of the rod, relative to the block"); axs[2].set_ylabel("downward displacement (mm)"); axs[2].set_ylim(0, 25)
for ax in axs: ax.set_xlim(0, 2.0)
axs[2].set_xlabel("time (ms)")
fig.tight_layout(); fig.savefig(R + "r3_histories.png"); plt.close(fig)
print("ok")
