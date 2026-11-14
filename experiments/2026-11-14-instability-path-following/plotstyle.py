"""Shared matplotlib style: light surface, fixed categorical order, thin marks,
recessive grid.  Figures are sized to read on a phone (wide enough for one
column, large fonts)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {  # fixed categorical order (validated default palette, light mode)
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "violet": "#6250d6", "ref": "#0b0b0b", "mute": "#8a8984", "bad": "#e34948",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["violet"]]


def setup():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.7,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e",
        "text.color": "#0b0b0b", "axes.titlecolor": "#0b0b0b",
        "lines.linewidth": 2.0, "lines.markersize": 7,
        "legend.frameon": False, "legend.fontsize": 10.5,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })
    return plt
