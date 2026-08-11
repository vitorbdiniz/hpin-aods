"""Stopgap generation of the two figures introduced in this revision, so the
manuscript compiles before the R/ggplot pipeline is available.

`make_figures.R` produces the same two files in the project's ggplot style and
overwrites these; this script exists only so the build is never broken by a
missing figure.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
TAB = os.path.join(HERE, "..", "results", "tables")
FIGS = os.path.join(HERE, "..", "results", "figs")
PAPER = os.path.join(HERE, "..", "..", "paper")

HPIN_C, PIN_C = "#2ca02c", "#1f77b4"
WINDOWS = [60, 90, 120, 150, 180]


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color="grey", alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def pls_difference():
    h = pd.read_csv(os.path.join(TAB, "PLS_HPIN.csv"), index_col=0)
    p = pd.read_csv(os.path.join(TAB, "PLS_PIN.csv"), index_col=0)
    diff = (h - p).dropna(how="all")
    data = [diff[c].dropna().values for c in diff.columns]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bp = ax.boxplot(data, tick_labels=[str(w) for w in WINDOWS], patch_artist=True,
                    widths=0.55, flierprops=dict(marker=".", markersize=2,
                                                 alpha=0.25, markeredgecolor="grey"))
    for box in bp["boxes"]:
        box.set(facecolor=HPIN_C, alpha=0.75, linewidth=0.6)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.2)
    ax.axhline(0, linestyle="--", color="grey", linewidth=1)
    lo, hi = np.percentile(np.concatenate(data), [2, 98])
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Window size")
    ax.set_ylabel("PLS difference (HPIN - PIN)")
    style(ax)
    fig.tight_layout()
    return fig, "PLS_diff_boxplot.pdf"


def persistence_grid():
    g = pd.read_csv(os.path.join(TAB, "persistence_grid.csv"))
    g = g[g["window"] == 180]
    labels = {"clean": "Correctly specified", "overdispersed": "Overdispersed emissions"}

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2), sharex=True, sharey=True)
    for ax, dgp in zip(axes, ["clean", "overdispersed"]):
        s = g[g["dgp"] == dgp].sort_values("spectral_gap_true")
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, zorder=1)
        ax.plot(s["spectral_gap_true"], s["gap_hpin"], "-o", color=HPIN_C,
                markersize=5, linewidth=1.4, label="HPIN", zorder=3)
        ax.plot(s["spectral_gap_true"], np.ones(len(s)), "-^", color=PIN_C,
                markersize=5, linewidth=1.4, label="PIN", zorder=2)
        ax.set_title(labels[dgp], fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("True spectral gap  (1 - rho)")
        style(ax)
    axes[0].set_ylabel("Estimated spectral gap")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig, "persistence_grid.pdf"


def main():
    os.makedirs(FIGS, exist_ok=True)
    for builder in (pls_difference, persistence_grid):
        fig, name = builder()
        fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
        if os.path.isdir(PAPER):
            fig.savefig(os.path.join(PAPER, name), bbox_inches="tight")
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main()
