"""Shared plot styling (palette from the dataviz reference: blue = full-field model, orange = uniaxial
baseline, aqua = null model; measured data in ink)."""
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; C_MUTED = "#8a897f"; GRID = "#e4e3dd"
C_FULL = "#2a78d6"; C_BASE = "#eb6834"; C_NULL = "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f3f6fb", "#b9d3f2", "#6aa5e6", "#2a78d6", "#12498f", "#0a2a57"])
DIV = LinearSegmentedColormap.from_list("div", ["#12498f", "#2a78d6", "#a9c8ee", "#eeeeec", "#f4b79d", "#eb6834", "#9c3a12"])
matplotlib.rcParams.update({"font.size": 9, "axes.edgecolor": C_MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                            "ytick.color": INK2, "font.family": "DejaVu Sans", "axes.titleweight": "bold"})

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
