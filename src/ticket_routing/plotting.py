"""Matplotlib helpers for README figures (shared light-surface style)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def confusion_matrix_plot(cm: np.ndarray, labels: list[str], title: str, path):
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6), facecolor=SURFACE)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK_2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("Predicted department", color=INK_2)
    ax.set_ylabel("True department", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coverage_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], path):
    """Accuracy of auto-routed tickets vs share of tickets auto-routed."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style(ax)
    for (name, (cov, acc)), color in zip(curves.items(), SERIES):
        ax.plot(cov * 100, acc * 100, color=color, linewidth=2, label=name)
    ax.set_xlabel("Tickets routed automatically (%)", color=INK_2)
    ax.set_ylabel("Accuracy on auto-routed tickets (%)", color=INK_2)
    ax.set_title("Confidence threshold: automate the easy tickets, send the rest to a human",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
