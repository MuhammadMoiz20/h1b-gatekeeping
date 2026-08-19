"""Shared matplotlib style and figure helpers.

One style is defined here and imported by every notebook so all 14 figures read
as one system rather than as fourteen separate defaults.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import FIGURES

# Categorical slots, fixed order, from the validated reference palette.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = "#2a78d6"                       # single hue for sequential encodings
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
SURFACE = "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11.5, "axes.titleweight": "600", "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "out", "ytick.direction": "out",
    "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
    "legend.frameon": False, "lines.linewidth": 2,
})


def finish(ax, title, sub=None, xlab=None, ylab=None, grid="y"):
    """Apply title, optional subtitle, axis labels and grid to an axes."""
    ax.set_title(title, color=INK, pad=26 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2,
                va="bottom")
    ax.set_xlabel(xlab or "")
    ax.set_ylabel(ylab or "")
    ax.grid(axis=grid, alpha=0.9)
    ax.set_axisbelow(True)


def save(name):
    """Write the current figure to output/figures and render it inline."""
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.show()
    plt.close()
