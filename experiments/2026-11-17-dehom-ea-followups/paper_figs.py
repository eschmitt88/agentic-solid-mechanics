"""Paper Figures 2, 4, 5, 6 and the H7 grade (lattice vs SIMP). Inputs: results/decomposition.json,
results/heldout/*.npz, results/simp/multistart.json, ../2026-11-17-dehom-dcbeam/results/heldout/*.npz.
Dataviz reference palette (validated), single axis per panel, direct labels, tables in results/paper_tables.json."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea  # noqa
import heldout as H  # noqa

FIG = HERE.parent.parent / "docs" / "paper" / "figs"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"FULL": "#2a78d6", "PCA": "#eb6834", "gen0": "#52514e", "SIMP": "#6250d6"}
BEAM = HERE.parent / "2026-11-17-dehom-dcbeam"
rows = json.load(open(HERE / "results/decomposition.json"))
tables = {}


def style(ax, title, xl, yl):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xlabel(xl, color=INK2, fontsize=9.5); ax.set_ylabel(yl, color=INK2, fontsize=9.5)


def avgF(f, obj):
    F, conn = H.averaged_objectives(np.load(f), obj)
    return F


def front(F):
    P = F[ea.nondominated_sort(F) == 0]
    return P[np.argsort(P[:, 0])]


# ---------------------------------------------------------------- Fig 2: nominal vs held-out fronts
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax, (lab, E, obj, run, yl) in zip(axs, [
        ("Girder, stiffness", HERE, "stiff", "stiff_FULL_s0", "compliance (log)"),
        ("Girder, buckling", HERE, "buck", "buck_FULL_s0", "first buckling load factor"),
        ("Paper's beam", BEAM, "umax", "umax_FULL_s0", "max displacement (log)")]):
    d = np.load(E / f"results/heldout/{run}__archive.npz")
    sgn = -1 if obj == "buck" else 1
    Pn = d["Fnom"][np.argsort(d["Fnom"][:, 0])]; Ph = front(avgF(E / f"results/heldout/{run}__archive.npz", obj))
    Pg = front(avgF(E / f"results/heldout/{obj}__gen0all.npz", obj))
    ax.axvspan(Pg[0, 0], Pg[-1, 0], color=GRID, alpha=0.6, lw=0)
    ax.step(Pg[:, 0], sgn * Pg[:, 1], where="post", color=C["gen0"], lw=2, label="initial population, held-out")
    ax.step(Pn[:, 0], sgn * Pn[:, 1], where="post", color=C["FULL"], lw=1.2, ls="--", label="evolved, as the search saw it")
    ax.step(Ph[:, 0], sgn * Ph[:, 1], where="post", color=C["FULL"], lw=2, label="evolved, held-out")
    if obj != "buck":
        ax.set_yscale("log")
    style(ax, lab, "volume fraction", yl)
axs[0].legend(frameon=False, fontsize=8.5)
fig.text(0.01, 0.005, "Grey band: volume range sampled by the initial population. Seed 0 of the paper's method.", fontsize=8, color=INK2)
fig.patch.set_facecolor(SURF); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIG / "fig2_fronts.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: paired d, nominal vs held-out
blocks = [("Girder S, gen 300", "stiff", 300, [0, 1, 2]), ("Girder S, gen 100", "stiff", 100, [0, 1, 2, 3, 4]),
          ("Girder B, gen 100", "buck", 100, [0, 1, 2, 3, 4]), ("Beam, gen 100", "umax", 100, [0, 1, 2])]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
t4 = {}
for i, (lab, obj, gen, seeds) in enumerate(blocks):
    for j, (m, col, name) in enumerate([("nominal", INK2, "nominal HV"), ("ho_all", C["PCA"], "held-out HV (fair baseline)")]):
        get = lambda arm, s: next((r for r in rows if r["run"] == f"{obj}_{arm}_s{s}" and r["gen"] == gen), None)
        dd = np.array([100 * (get("PCA", s)[m] - get("FULL", s)[m]) for s in seeds if get("PCA", s) and get("FULL", s)])
        if len(dd) < 2:
            continue
        from scipy import stats
        mu = dd.mean(); hw = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
        y = i + (0.15 if j == 0 else -0.15)
        ax.errorbar(mu, y, xerr=hw, fmt="o", color=col, ms=8, mec=SURF, mew=2, capsize=0, lw=2, label=name if i == 0 else None)
        ax.plot(dd, [y] * len(dd), "|", color=col, ms=10, alpha=0.6)
        t4[f"{lab}|{m}"] = dict(n=len(dd), mean=float(mu), ci=[float(mu - hw), float(mu + hw)])
ax.axvline(0, color=INK2, lw=0.8); ax.axvspan(-5, 5, color=GRID, alpha=0.5, lw=0)
ax.set_yticks(range(len(blocks))); ax.set_yticklabels([b[0] for b in blocks])
style(ax, "Linear PCA crossover minus the paper's VAE crossover", "difference in HV gain (percentage points)", "")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.text(0.01, 0.005, "Dots: mean over paired seeds, bars: 95 % t-interval, ticks: seeds. Grey band: the pre-registered ±5 pp threshold.", fontsize=8, color=INK2)
fig.patch.set_facecolor(SURF); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "fig4_operators.png", dpi=170); plt.close(fig)
tables["fig4_paired_d"] = t4

# ---------------------------------------------------------------- Fig 5: per-design realisation optimism
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
t5 = {}
for ax, obj, key, lab in zip(axs, ("stiff", "buck"), ("compliance", "blf"), ("Girder, stiffness", "Girder, buckling")):
    data, names = [], []
    for src, f in [("initial\n(nominal front)", f"{obj}__gen0.npz"), ("paper's\nmethod", f"{obj}_FULL_s0__archive.npz"),
                   ("PCA\ncrossover", f"{obj}_PCA_s0__archive.npz")]:
        # the front each source would report (selected on its own realisation) — no re-selection here
        d = np.load(HERE / "results/heldout" / f)
        conn = d["connected"].all(1)
        avg = np.exp(np.log(np.maximum(d[key], 1e-12)).mean(1))[conn]
        nom = d["Fnom"][conn, 1] * (-1 if obj == "buck" else 1)
        r = avg / nom if obj == "stiff" else nom / avg
        data.append(np.log10(r)); names.append(src)
        t5[f"{obj}|{src.replace(chr(10), ' ')}"] = dict(median=float(np.median(r)), p90=float(np.percentile(r, 90)),
                                                        n=int(len(r)), disconnected_frac=float(np.mean(~conn)))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, lw=2), flierprops=dict(marker="o", ms=3, mfc=INK2, mec="none"))
    for patch, col in zip(bp["boxes"], (C["gen0"], C["FULL"], C["PCA"])):
        patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(names)
    ax.axhline(0, color=INK2, lw=0.8)
    yt = [1, 1.25, 1.5, 2, 3, 5, 10]
    ax.set_yticks(np.log10(yt)); ax.set_yticklabels([f"×{v:g}" for v in yt])
    style(ax, lab, "", "how much better the design looked\nthan its realisation average")
fig.text(0.01, 0.005, "Front designs re-evaluated on 10 held-out phasor realisations; seed 0. Above ×1: the search's own realisation flattered the design.", fontsize=8, color=INK2)
fig.patch.set_facecolor(SURF); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "fig5_realisation.png", dpi=170); plt.close(fig)
tables["fig5_optimism"] = t5

# ---------------------------------------------------------------- Fig 6 + H7: lattice vs SIMP at matched volume
simp = json.load(open(HERE / "results/simp/multistart.json"))["rows"]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
h7 = {}
for ax, obj, key, lab in zip(axs, ("stiff", "buck"), ("compliance", "blf"), ("Girder, compliance (lower is better)", "Girder, BLF (higher is better)")):
    lat = np.concatenate([avgF(f, obj) for f in sorted((HERE / "results/heldout").glob(f"{obj}_*__archive.npz"))] +
                         [avgF(HERE / f"results/heldout/{obj}__gen0all.npz", obj)])
    vfs = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_s = [(max if obj == "buck" else min)(r[key] for r in simp if r["vf_target"] == v and r["connected"]) for v in vfs]
    best_l = []
    for v in vfs:
        m = np.abs(lat[:, 0] - v) <= 0.01
        best_l.append(np.nan if not m.any() else (-lat[m, 1].min() if obj == "buck" else lat[m, 1].min()))
    ax.plot(vfs, best_s, "o-", color=C["SIMP"], lw=2, ms=8, mec=SURF, mew=2, label="SIMP (multi-start best)")
    ax.plot(vfs, best_l, "o-", color=C["FULL"], lw=2, ms=8, mec=SURF, mew=2, label="best lattice, held-out (any arm)")
    if obj == "stiff":
        ax.set_yscale("log")
    style(ax, lab, "volume fraction", key)
    wins = [bool((l < s) if obj == "stiff" else (l > s)) for l, s, v in zip(best_l, best_s, vfs) if v in (0.30, 0.35, 0.40, 0.45) and np.isfinite(l)]
    h7[obj] = dict(vf=vfs, simp=best_s, lattice=[float(x) for x in best_l], lattice_wins_at_030_045=int(sum(wins)))
axs[0].legend(frameon=False, fontsize=8.5)
fig.patch.set_facecolor(SURF); fig.tight_layout(); fig.savefig(FIG / "fig6_simp.png", dpi=170); plt.close(fig)
tables["H7"] = h7
json.dump(tables, open(HERE / "results/paper_tables.json", "w"), indent=1)
print(json.dumps(h7, indent=1)); print(json.dumps(t5, indent=1))
