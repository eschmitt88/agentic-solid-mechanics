"""Render model-vs-data stress-stretch hysteresis loops (run in solidmech env)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"
d = np.load(R / "predictions.npz", allow_pickle=True)
lam, sig, names = d["lam"], d["sig"], list(d["names"])
idx = {n: i for i, n in enumerate(names)}
RATEC = {"rate1": "#1f6feb", "rate2": "#cf222e", "rate3": "#2da44e"}
RATEL = {"rate1": "0.01/s", "rate2": "0.03/s", "rate3": "0.05/s"}


def loop(ax, name, pred_key, label_data=True, held=False):
    i = idx[name]
    rk = name.rsplit("_", 1)[1]
    c = RATEC[rk]
    ax.plot(lam[i], sig[i], "o", ms=2.2, color=c, alpha=0.55,
            label=(f"data {RATEL[rk]}" if label_data else None))
    ax.plot(lam[i], d[pred_key][i], "-", lw=1.8, color=c,
            label=(f"model {RATEL[rk]}" + (" (held-out)" if held else "")))


def fig_fit_all():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in ("rate1", "rate2", "rate3"):
        loop(ax[0], f"Fig3_2_{r}", "pred_fit_all")
    ax[0].set_title("Rate-dependence at one amplitude (λ→4)")
    for a in ("Fig3_1_rate3", "Fig3_2_rate3", "Fig4_1_rate3", "Fig4_2_rate3"):
        loop(ax[1], a, "pred_fit_all", label_data=False)
    ax[1].set_title("Amplitude range at one rate (0.05/s)")
    for a in ax:
        a.set_xlabel("stretch λ"); a.set_ylabel("nominal stress"); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.suptitle("Calibrated to all 11 curves — Ogden + Bergström–Boyce (7 parameters)", fontsize=11)
    fig.tight_layout(); fig.savefig(R / "fit_all.png", dpi=130); plt.close(fig)


def fig_study_A():
    k = "pred_study_A_amplitude_extrapolation"
    AC = {"Fig3_1": "#1f6feb", "Fig3_2": "#8250df", "Fig4_1": "#bc4c00", "Fig4_2": "#cf222e"}
    AL = {"Fig3_1": "λ→2.25", "Fig3_2": "λ→4.0", "Fig4_1": "λ→6.25", "Fig4_2": "λ→8.9"}

    def aloop(ax, name, held=False):
        i = idx[name]; amp = name.rsplit("_", 1)[0]; c = AC[amp]
        ax.plot(lam[i], sig[i], "o", ms=2.2, color=c, alpha=0.5)
        ax.plot(lam[i], d[k][i], "-", lw=1.9, color=c,
                label=f"{AL[amp]}" + (" — predicted" if held else " — train"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a in ("Fig3_1_rate3", "Fig3_2_rate3"):
        aloop(ax[0], a)
    ax[0].set_title("Calibration range (λ ≤ 4, at 0.05/s)")
    for a in ("Fig4_1_rate3", "Fig4_2_rate3"):
        aloop(ax[1], a, held=True)
    ax[1].set_title("PREDICTED — amplitudes never seen in calibration")
    for a in ax:
        a.set_xlabel("stretch λ"); a.set_ylabel("nominal stress"); a.legend(fontsize=8); a.grid(alpha=0.3)
        a.set_xlim(0.8, 9.3)
    fig.suptitle("Validation A — amplitude extrapolation: trained on λ≤4 (dots=data, line=model), predicts λ up to 8.9",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(R / "study_A.png", dpi=130); plt.close(fig)


def fig_study_B():
    k = "pred_study_B_heldout_rate"
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, amp in enumerate(("Fig3_1", "Fig3_2", "Fig4_1")):
        # faint trained rates
        for r in ("rate1", "rate3"):
            i = idx[f"{amp}_{r}"]
            ax[j].plot(lam[i], sig[i], "o", ms=1.6, color="#bbb", alpha=0.5)
            ax[j].plot(lam[i], d[k][i], "-", lw=1.0, color="#bbb")
        loop(ax[j], f"{amp}_rate2", k, held=True)
        ax[j].set_title(f"{amp} (λ→{ {'Fig3_1':2.25,'Fig3_2':4.0,'Fig4_1':6.25}[amp] })")
        ax[j].set_xlabel("stretch λ"); ax[j].set_ylabel("nominal stress")
        ax[j].legend(fontsize=7); ax[j].grid(alpha=0.3)
    fig.suptitle("Validation B — held-out rate: trained on 0.01 & 0.05/s (grey), predicts 0.03/s (red)", fontsize=11)
    fig.tight_layout(); fig.savefig(R / "study_B.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    fig_fit_all(); fig_study_A(); fig_study_B()
    print("wrote fit_all.png, study_A.png, study_B.png")
