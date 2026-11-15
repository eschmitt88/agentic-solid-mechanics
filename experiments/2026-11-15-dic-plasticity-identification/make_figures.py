"""All result figures for README and the review page."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data, femu, baseline
from plotlib import *
from plots_results import _outline, fig_force, FD, ST_LABEL

R = "results"
J = lambda f: json.load(open(os.path.join(R, f)))
fz = J("frozen_models.json"); PRI, BASE = fz["primary"], fz["baseline"]
ho = J("heldout_scores.json"); dev = J("dev_scores.json")
NAMES = {"O15-3": "Plate O15 (slot + notch)", "XT10": "X-specimen, same direction", "XR5": "X-specimen, rolled direction", "O5-4": "Plate O5 (development)"}


def fields4(spec, res, out, st=2, comp=1):
    D = np.load(f"{FD}/{spec}__DIC.npz"); pts = D["pts"]; Em = D["E"][st, comp]
    Ep = np.load(f"{FD}/{spec}__{PRI}__cal_n1.npz")["E"][st, comp]; Eb = np.load(f"{FD}/{spec}__{BASE}__cal_n1.npz")["E"][st, comp]
    info = res["info"]
    lo, hi = (0.0, float(np.percentile(Em, 99.7))) if comp == 1 else (float(np.percentile(Em, 0.3)), float(np.percentile(Em, 99.7)))
    if comp != 1:
        m = max(abs(lo), abs(hi)); lo, hi = -m, m
    dv = 0.2 * max(abs(lo), abs(hi))
    w = pts[:, 0].max() - pts[:, 0].min() + 4; hgt = info["yhi"] - info["ylo"] + 8
    W = 4.6; ph = (W / 2) * hgt / w * 0.92
    fh = 2 * (ph + 0.78) + 0.45
    fig, axs = plt.subplots(2, 2, figsize=(W, fh))
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 0.75 / fh, bottom=0.01, wspace=0.04, hspace=0.22)
    kw = dict(s=0.9, marker="s", linewidths=0, rasterized=True)
    cm_main = SEQ if comp == 1 else DIV
    panels = [(Em, cm_main, (lo, hi), "measured by the cameras"), (Ep, cm_main, (lo, hi), "predicted: full-field model"),
              (Ep - Em, DIV, (-dv, dv), "error: full-field model"), (Eb - Em, DIV, (-dv, dv), "error: tension-test model")]
    for ax, (val, cm, lim, ttl) in zip(axs.ravel(), panels):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=val, cmap=cm, vmin=lim[0], vmax=lim[1], **kw)
        _outline(ax, spec, info["ylo"], info["yhi"]); ax.set_title(ttl, fontsize=9.5, color=INK); ax.set_anchor("N")
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.015, fraction=0.045, aspect=16)
        cb.ax.tick_params(labelsize=8.5); cb.outline.set_visible(False); cb.ax.locator_params(nbins=4)
    cn = ["Sideways strain", "Stretch along the pull", "Shear strain"][comp]
    fig.suptitle(f"{cn} at {ST_LABEL[st]} of the way to peak load", fontsize=10.5, color=INK, y=1 - 0.12 / fh)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.06); plt.close(fig)


def fig_scores(out):
    """Held-out errors against tolerance, full-field model vs tension-test model."""
    rows = [("force curve error", "force_nrmse", 0.05), ("force at 90 % of the way", "force_last_err", 0.05),
            ("strain map error at 30 %", "strain_relL2_st13", 0.15), ("strain map error at 60 %", "strain_relL2_st27", 0.15),
            ("strain map error at 90 %", "strain_relL2_st40", 0.15), ("hot-spot strain error", "hot_err", 0.15)]
    specs = ["O15-3", "XT10", "XR5"]
    fig, axs = plt.subplots(3, 1, figsize=(4.6, 7.6), sharex=True)
    for ax, s in zip(axs, specs):
        style(ax); ax.grid(axis="y", visible=False)
        for i, (lab, key, tol) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot([0, 1], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=1)
            for k, col, mk, dy in ((BASE, C_BASE, "D", -0.18), (PRI, C_FULL, "o", 0.18)):
                v = abs(ho[s]["models"][k][key]) / tol
                ax.plot([v], [y + dy], marker=mk, color=col, ms=7, mec=SURFACE, mew=1, zorder=3)
        ax.axvline(1.0, color=INK, lw=1.3); ax.set_xlim(0, 1.08); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "half the limit", "limit"], fontsize=9)
        ax.set_title(NAMES[s], fontsize=10, color=INK, loc="left")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    axs[0].plot([], [], "o", color=C_FULL, ms=7, label="full-field model"); axs[0].plot([], [], "D", color=C_BASE, ms=7, label="tension-test model")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=9.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Error as a fraction of the allowed limit\n(left of the black line = pass)", fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.01, 1, 0.985)); fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_hardening(out, a95):
    fig, ax = plt.subplots(figsize=(4.6, 4.3)); style(ax)
    a = np.linspace(0, 0.9, 300)
    for i, n in enumerate(["BT3", "CT3", "CT4"]):
        e, s_eng, ep, s = baseline.curve(n)
        ax.plot(ep, s, color=INK, lw=4.5, alpha=0.16, label="tension tests (measured)" if i == 0 else None)
    ax.plot(a, femu.sigy(fz["models"][BASE]["p"], a), color=C_BASE, lw=2, label="fitted to tension tests only")
    ax.plot(a, femu.sigy(fz["models"][PRI]["p"], a), color=C_FULL, lw=2, label="full-field fit, direction-dependent")
    ax.plot(a, femu.sigy(fz["models"]["M2"]["p"], a), color=C_FULL, lw=1.6, ls=(0, (4, 2)), label="full-field fit, isotropic")
    b = J("baseline.json")
    for x, t in ((b["ep_max"], "tension\ntest ends"), (a95, "95 % of plate\nmaterial is\nbelow this")):
        ax.axvline(x, color=C_MUTED, lw=1, ls=":"); ax.text(x + 0.012, 1690, t, fontsize=8, color=INK2, va="top")
    ax.set_xlabel("plastic strain (true)"); ax.set_ylabel("yield stress, MPa"); ax.set_xlim(0, 0.9); ax.set_ylim(0, 1700)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Hardening curve: three calibrations", fontsize=10.5, color=INK, loc="left")
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_ident(out):
    f = J(f"fits/{PRI}.json")["best"]["ident"]
    raw = dict(zip(f["params"], f["dlog_1pct_profile"])); cv = dict(zip(f["curve_coords"]["names"], f["curve_coords"]["dlog_1pct_profile"]))
    lab_raw = {"s0": "initial yield s0", "Q1": "Voce size Q", "b1": "Voce rate b", "H": "linear slope H", "a": "Hill a", "h": "Hill h", "c": "Hill c"}
    fig, axs = plt.subplots(2, 1, figsize=(4.6, 5.0), sharex=True, gridspec_kw={'height_ratios': [7, 4]})
    for ax, dct, ttl, lab in ((axs[0], raw, "formula constants", lab_raw),
                              (axs[1], {k: v for k, v in cv.items() if k.startswith("sigy")}, "yield stress at a given strain", None)):
        style(ax); ax.grid(axis="y", visible=False)
        ks = list(dct)[::-1]
        ax.barh(range(len(ks)), [100 * dct[k] for k in ks], color=C_FULL, height=0.55)
        for i, k in enumerate(ks):
            ax.text(100 * dct[k] + 0.6, i, f"±{100 * dct[k]:.1f} %", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(ks))); ax.set_yticklabels([lab[k] if lab else k.replace("sigy(", "at strain ").replace(")", "") for k in ks], fontsize=9)
        ax.set_title(ttl, fontsize=10, color=INK, loc="left"); ax.set_xlim(0, 37)
    axs[1].set_xlabel("how far it can move before the fit gets 1 % worse", fontsize=9)
    fig.suptitle("What the data pin down", fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_specimens(out):
    items = [("O1", "O1", "calibration"), ("O14", "O14", "calibration"), ("O5", "O5", "check"),
             ("O15", "O15", "held out"), ("X", "X", "held out\n(2 directions)")]
    fig, axs = plt.subplots(1, 5, figsize=(4.6, 3.3), gridspec_kw={"width_ratios": [60, 60, 60, 60, 71]})
    for ax, (k, name, role) in zip(axs, items):
        bo, bh = data.load_geometry(k)
        held = role.startswith("held")
        ax.fill(bo[:, 0], bo[:, 1], color=INK if held else C_MUTED, alpha=0.30 if held else 0.14, lw=0)
        ax.plot(bo[:, 0], bo[:, 1], color=INK, lw=1.3)
        for h in bh:
            ax.fill(h[:, 0], h[:, 1], color=SURFACE, lw=0); ax.plot(h[:, 0], h[:, 1], color=INK, lw=1.3)
        ax.set_xlim(-3, 74); ax.set_ylim(-4, 248); ax.set_aspect("equal", anchor="S"); ax.axis("off")
        ax.set_title(f"{name}\n{role}", fontsize=8, color=INK, fontweight="bold" if held else "normal")
    fig.suptitle("Five shapes cut from one 1.5 mm stainless sheet,\neach gripped at the ends and pulled", fontsize=10, color=INK, y=1.08)
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_ccx(out):
    f = [x for x in os.listdir(R) if x.startswith("ccx_check_") and x.endswith(".json") and "devtest" not in x]
    if not f: return
    c = J(f[0]); spec = c["spec"]; info = ho[spec]["info"]
    x = np.array(info["ext_st"])[1:]
    fig, ax = plt.subplots(figsize=(4.6, 3.6)); style(ax)
    ax.plot(x, np.array(info["F_meas"])[1:] / 1e3, color=INK, lw=1.2, alpha=0.7, label="measured")
    ax.plot(x, np.array(c["F_jax"]) / 1e3, color=C_FULL, lw=2, label="this project's solver (JAX)")
    n = len(c["F_ccx"]); ax.plot(x[:n][::2], np.array(c["F_ccx"])[::2] / 1e3, "o", color=C_BASE, ms=5, mec=SURFACE, mew=0.8, label="CalculiX, same model")
    ax.set_xlabel("grip extension, mm"); ax.set_ylabel("force, kN"); ax.set_ylim(0, None); ax.set_xlim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Independent solver check, unseen plate O15", fontsize=10.5, color=INK, loc="left")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_warmup(out):
    w = J("warmup.json"); t = np.array(w["truth"]["truth"]); p = np.array(w["fits"]["A"]["p"]); p0 = np.zeros(9); p0[[0, 1, 2, 3]] = w["fits"]["A"]["start"][:4]
    a = np.linspace(0, w["truth"]["alpha_max"], 200)
    fig, ax = plt.subplots(figsize=(4.6, 3.6)); style(ax)
    ax.plot(a, femu.sigy(p0, a), color=C_MUTED, lw=1.5, ls=(0, (4, 2)), label="starting guess")
    ax.plot(a, femu.sigy(t, a), color=INK, lw=4, alpha=0.3, label="hidden truth")
    ax.plot(a, femu.sigy(p, a), color=C_FULL, lw=1.8, label="recovered from noisy synthetic data")
    ax.set_xlabel("plastic strain (true)"); ax.set_ylabel("yield stress, MPa"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Warm-up: finding a hidden answer", fontsize=10.5, color=INK, loc="left")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    a95 = J("alpha_seen.json")["alpha95_calibration"] if os.path.exists(f"{R}/alpha_seen.json") else 0.3
    fig_specimens(f"{R}/fig_specimens.png")
    fig_scores(f"{R}/fig_scores.png"); fig_hardening(f"{R}/fig_hardening.png", a95); fig_ident(f"{R}/fig_identifiability.png")
    fig_warmup(f"{R}/fig_warmup.png"); fig_ccx(f"{R}/fig_ccx.png")
    mods = [(PRI, "full-field model", C_FULL, "-"), (BASE, "tension-test model", C_BASE, "-")]
    for s in ["O15-3", "XT10", "XR5"]:
        fig_force(s, ho[s], mods, f"{R}/fig_force_{s}.png", title="Pulling force against extension")
        fields4(s, ho[s], f"{R}/fig_fields_{s}_Eyy.png", comp=1); fields4(s, ho[s], f"{R}/fig_fields_{s}_Exx.png", comp=0)
    fig_force("O5-4", dev["O5-4"], mods, f"{R}/fig_force_O5-4.png", title="Pulling force against extension")
    fields4("O5-4", dev["O5-4"], f"{R}/fig_fields_O5-4_Eyy.png", comp=1)
