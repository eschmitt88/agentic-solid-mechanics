"""Rung-1 figures. Run after analyze_pf.py."""
import json, os, numpy as np
import matplotlib.pyplot as plt
from analyze_pf import load, metrics, grid, field_img, shear_path, C, ORDER, SEQ

OUT = "results"
miehe_t = np.loadtxt("refs_pdf/phasefieldx_miehe_solution_tension.csv")
miehe_s = np.loadtxt("refs_pdf/phasefieldx_miehe_solution_shear.csv")
REF = {  # (peak kN, u at peak mm, label)
    "t015": [(0.70, 5.9e-3, "Molnár 2017"), (0.701, 5.55e-3, "Miehe 2010 (digitized)"), (0.80, 5.5e-3, "Zhou 2019")],
    "t0075": [(0.73, 5.9e-3, "Molnár 2017"), (0.74, 5.2e-3, "Zhou 2019"), (0.68, 5.0e-3, "Zhang 2017"), (0.73, 5.0e-3, "Zhang 2018")],
    "s0075": [(0.60, 0.0110, "Zhou 2019"), (0.595, 0.0108, "Zhang 2017")],
    "s015": [(0.54, 0.0100, "Zhou 2019")],
}
MK = ["s", "D", "^", "v"]


def refmarks(ax, key, scale):
    for (F, u, lab), mk in zip(REF[key], MK):
        ax.plot(u * scale, F, mk, ms=8, mfc="none", mec=C["ink"], mew=1.3, label=f"published peak: {lab}", zorder=5)


def curves(ax, tags, labels, scale, colors=None):
    for i, (t, lab) in enumerate(zip(tags, labels)):
        r = load(t)
        if r is None:
            continue
        ax.plot(r["u"] * scale, r["F"], color=(colors or ORDER)[i], label=lab)


def fig_curves():
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    sc = 1e3
    ax = axs[0, 0]
    curves(ax, ["t_l015_h2", "t_l015_h4", "t_l015_h8", "t_l015_h16"], ["this work, h = l/2", "h = l/4", "h = l/8", "h = l/16"], sc)
    ax.plot(miehe_t[:, 0] * sc, miehe_t[:, 1], "--", color=C["mute"], lw=1.3, label="Miehe 2010 curve (digitized, phasefieldx)")
    refmarks(ax, "t015", sc); ax.set_title("Tension, l = 0.015 mm"); ax.set_xlim(4.0, 6.6); ax.set_ylim(0, 0.9)
    ax = axs[0, 1]
    curves(ax, ["t_l0075_h2", "t_l0075_h4", "t_l0075_h8", "t_l0075_h16"], ["this work, h = l/2", "h = l/4", "h = l/8", "h = l/16"], sc)
    refmarks(ax, "t0075", sc); ax.set_title("Tension, l = 0.0075 mm"); ax.set_xlim(4.0, 6.6); ax.set_ylim(0, 0.9)
    ax = axs[1, 0]
    curves(ax, ["s_l015_h2", "s_l015_h4", "s_l015_h8"], ["this work, h = l/2", "h = l/4", "h = l/8"], sc)
    ax.plot(miehe_s[:, 0] * sc, miehe_s[:, 1], "--", color=C["mute"], lw=1.3, label="Miehe 2010 curve (digitized; l not stated)")
    refmarks(ax, "s015", sc); ax.set_title("Shear, l = 0.015 mm"); ax.set_xlim(0, 20); ax.set_ylim(0, 0.7)
    ax = axs[1, 1]
    curves(ax, ["s_l0075_h2", "s_l0075_h4", "s_l0075_h8"], ["this work, h = l/2", "h = l/4", "h = l/8"], sc)
    refmarks(ax, "s0075", sc); ax.set_title("Shear, l = 0.0075 mm"); ax.set_xlim(0, 20); ax.set_ylim(0, 0.7)
    for k, ax in enumerate(axs.ravel()):
        ax.set_xlabel("top-edge displacement [10⁻³ mm]"); ax.set_ylabel("reaction force [kN]")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left" if k < 2 else "lower center")
    fig.suptitle("Single-edge-notched plate: load–displacement, mesh series vs published results", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pf_load_displacement.png"); plt.close(fig)


def fig_convergence(M):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ax = axs[0]
    series = [("t_l015", "tension l=0.015", "t015"), ("t_l0075", "tension l=0.0075", "t0075"), ("s_l015", "shear l=0.015", "s015"), ("s_l0075", "shear l=0.0075", "s0075")]
    for i, (pre, lab, key) in enumerate(series):
        pts = [(M[f"{pre}_h{n}"]["h"] / M[f"{pre}_h{n}"]["l"], M[f"{pre}_h{n}"]["peak_F"]) for n in (2, 4, 8, 16) if f"{pre}_h{n}" in M]
        if pts:
            x, y = zip(*pts); ax.plot(x, y, "o-", color=ORDER[i], label=lab, ms=5)
            ref = np.mean([r[0] for r in REF[key]])
            ax.plot([0.0], [ref], "*", color=ORDER[i], ms=11, mec=C["ink"], mew=0.6)
    ax.set_xlabel("element size / length scale  h/l"); ax.set_ylabel("peak load [kN]"); ax.set_xlim(-0.03, 0.55)
    ax.set_title("Mesh convergence (★ = published mean)"); ax.legend(fontsize=8, frameon=False)
    ax = axs[1]
    tags = [("t_l015_h4_molnar_steps", "Δu = 1e-4 then 1e-5"), ("t_l015_h4_du1e-5", "Δu = 1e-5 throughout"), ("t_l015_h4", "Δu = 1e-5 then 1e-6 (Miehe)"),
            ("t_l015_h4_du2.5e-7", "Δu = 1e-5 then 2.5e-7"), ("t_l015_h4_stag_tol1e-5", "iterated to convergence")]
    for i, (t, lab) in enumerate(tags):
        r = load(t)
        if r is not None:
            ax.plot(r["u"] * 1e3, r["F"], color=ORDER[i], label=lab, ls="-" if "iter" not in lab else "--")
    ax.set_xlim(5.0, 6.6); ax.set_ylim(0, 0.8); ax.set_xlabel("displacement [10⁻³ mm]"); ax.set_ylabel("force [kN]")
    ax.set_title("Tension: single-pass staggered vs load step"); ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax = axs[2]
    tags = [("s_l015_h4_du4e-5", "Δu = 4e-5"), ("s_l015_h4", "Δu = 1e-5 (Miehe)"), ("s_l015_h4_du2.5e-6", "Δu = 2.5e-6"), ("s_l015_h4_stag100", "Δu = 1e-5, iterated (≤100 passes)")]
    for i, (t, lab) in enumerate(tags):
        r = load(t)
        if r is not None:
            ax.plot(r["u"] * 1e3, r["F"], color=ORDER[i], label=lab, ls="-" if "iter" not in lab else "--")
    ax.set_xlim(6, 20); ax.set_ylim(0, 0.65); ax.set_xlabel("displacement [10⁻³ mm]"); ax.set_ylabel("force [kN]")
    ax.set_title("Shear: single-pass staggered vs load step"); ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pf_convergence.png"); plt.close(fig)


def fig_paths(M):
    tt = next((t for t in ["t_l0075_h16", "t_l0075_h8", "t_l0075_h4"] if t in M), None)
    ss = next((t for t in ["s_l0075_h8", "s_l0075_h4", "s_l0075_h2"] if t in M), None)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
    r = load(tt); field_img(axs[0], r, title=f"Tension, l=0.0075, h=l/{round(r['meta']['l']/r['meta']['h'])}: final crack")
    r = load(ss)
    keys = sorted(k for k in r if k.startswith("d_at_"))
    pick = [k for k in keys if float(k[5:]) in (0.011, 0.013)]
    for ax, k in zip(axs[1:3], pick):
        field_img(ax, r, r[k], title=f"Shear, l=0.0075: u = {float(k[5:])*1e3:.0f}e-3 mm")
    field_img(axs[3], r, title=f"Shear, final (u = {r['u'][-1]*1e3:.0f}e-3 mm)")
    m = M[ss]
    if m.get("crack_angle_deg"):
        a = np.radians(m["crack_angle_deg"]); axs[3].plot([0.5, 0.5 + 0.22 * np.cos(a)], [0.5, 0.5 - 0.22 * np.sin(a)], color=C["orange"], lw=1.4)
        axs[3].text(0.56, 0.47, f"{m['crack_angle_deg']:.0f}°", color=C["ink"], fontsize=9)
        a = np.radians(61); axs[3].plot([0.5, 0.5 + 0.22 * np.cos(a)], [0.5, 0.5 - 0.22 * np.sin(a)], color=C["ink"], lw=1.0, ls=":")
    fig.suptitle("Phase field d (dark = broken). White line = initial notch. Orange = fitted initial crack direction, dotted = published 61°", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pf_crack_paths.png"); plt.close(fig)


def fig_traps(M):
    tags = [("s_l015_h4", "reference set-up\n(spectral split, irreversible)"), ("s_l015_h4_nosplit", "no energy split"),
            ("s_l015_h4_amor", "volumetric–deviatoric split"), ("s_l015_h4_noirrev", "no irreversibility"),
            ("s_l015_h4_freesides", "side edges left free")]
    fig = plt.figure(figsize=(14, 6.6))
    for i, (t, lab) in enumerate(tags):
        r = load(t)
        if r is None:
            continue
        ax = fig.add_subplot(2, 5, i + 1); field_img(ax, r, title=lab)
    ax = fig.add_subplot(2, 1, 2)
    for i, (t, lab) in enumerate(tags):
        r = load(t)
        if r is not None:
            ax.plot(r["u"] * 1e3, r["F"], color=ORDER[i], label=lab.replace("\n", " "))
    ax.plot(miehe_s[:, 0] * 1e3, miehe_s[:, 1], "--", color=C["mute"], lw=1.3, label="Miehe 2010 (digitized)")
    ax.set_xlabel("displacement [10⁻³ mm]"); ax.set_ylabel("force [kN]"); ax.set_xlim(0, 20); ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Shear test, l = 0.015 mm, h = l/4: what each modelling trap does to the crack and the curve", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pf_traps.png"); plt.close(fig)


if __name__ == "__main__":
    M = json.load(open("results/pf_run_metrics.json"))
    fig_curves(); fig_convergence(M); fig_paths(M); fig_traps(M)
    print("figures written")
