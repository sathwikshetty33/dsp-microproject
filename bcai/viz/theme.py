"""Chart theme: a validated categorical palette and recessive chart furniture.

The palette below passed the six-check validator on the light surface
(#fcfcfb): lightness band, chroma floor, colour-vision-deficiency separation,
normal-vision separation and contrast.  Two slots sit under 3:1 contrast, so
every chart that uses them ships visible value labels -- the relief rule.
Slots are assigned in fixed order and never cycled.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e6e5e1"

#: fixed categorical order -- slot 1 first, never rotated
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")

#: single-hue ramp for magnitude (light -> dark)
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95", "#104281", "#0d366b")

STATUS = {"good": "#008300", "warning": "#eda100", "critical": "#e34948"}


def apply_theme() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": TEXT_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.markersize": 7,
        "figure.dpi": 130,
    })


def rounded_bars(ax, xs, heights, width, color, *, rounding=0.012,
                 baseline=0.0, label=None, gap=0.012):
    """Vertical bars with 4px-equivalent rounded tops and a surface gap.

    Matplotlib's own bars are square-ended, so the caps are drawn as
    FancyBboxPatch rectangles anchored to the baseline.
    """
    patches = []
    inner = max(width - gap, width * 0.6)
    for i, (x, h) in enumerate(zip(xs, heights)):
        if h is None:
            continue
        h = float(h)
        pad = min(rounding, abs(h) / 2 if h else rounding)
        patch = FancyBboxPatch(
            (x - inner / 2 + pad, baseline + pad),
            inner - 2 * pad, max(abs(h) - 2 * pad, 1e-6),
            boxstyle=f"round,pad={pad:.5f},rounding_size={pad:.5f}",
            linewidth=1.4, edgecolor=SURFACE, facecolor=color,
            mutation_aspect=1.0,
            label=label if i == 0 else None,
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def value_labels(ax, xs, heights, *, fmt="{:.2f}", dy=0.012, color=None,
                 fontsize=8.5, at=None) -> None:
    """Direct labels — required relief for the low-contrast palette slots.

    ``at`` overrides the y position the label is anchored to, so a label can
    clear an error bar instead of landing on top of it, while still printing
    the value from ``heights``.
    """
    anchors = heights if at is None else at
    for x, h, a in zip(xs, heights, anchors):
        if h is None:
            continue
        ax.text(x, float(a) + dy, fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color=color or TEXT_SECONDARY)


def header(ax, title=None, subtitle=None, *, legend=False, legend_ncols=1,
           legend_rows=1) -> None:
    """Stack legend, subtitle and title upward from the top of the axes.

    Everything lives above the plot area in one coordinate system, so a long
    subtitle or a wide legend can never land on top of the data or on the
    title -- which is what happens when the legend is placed inside.
    """
    y = 1.015
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, y), ncols=legend_ncols,
                  frameon=False, handlelength=1.1, handleheight=0.9,
                  columnspacing=1.6, borderpad=0.0)
        y += 0.052 * legend_rows + 0.014
    if subtitle:
        ax.text(0.0, y, subtitle, transform=ax.transAxes, fontsize=9.3,
                color=TEXT_SECONDARY, va="bottom", linespacing=1.4)
        y += 0.047 * (subtitle.count("\n") + 1) + 0.012
    if title:
        ax.text(0.0, y, title, transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=TEXT_PRIMARY, va="bottom")


def finish(ax, *, title=None, subtitle=None, ylabel=None, xlabel=None,
           ylim=None, legend=True, legend_ncols=1, legend_rows=1) -> None:
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    header(ax, title, subtitle, legend=legend, legend_ncols=legend_ncols,
           legend_rows=legend_rows)


def save(fig, path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def place_labels(ax, xs, ys, texts, *, fontsize=8.6, color=None,
                 marker_pad=9.0) -> None:
    """Annotate points without letting labels collide.

    Offsets are chosen greedily against the *rendered* text extents rather than
    guessed, so the result holds whatever the data turns out to be.  Candidate
    positions alternate above and below at growing distances, and each point's
    own marker is treated as an obstacle.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    color = color or TEXT_SECONDARY

    # Markers are obstacles for every label, including their own.
    boxes = []
    for x, y in zip(xs, ys):
        px, py = ax.transData.transform((x, y))
        boxes.append((px - marker_pad, py - marker_pad,
                      px + marker_pad, py + marker_pad))

    def overlaps(b):
        return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3])
                   for o in boxes)

    order = sorted(range(len(xs)), key=lambda i: -ys[i])
    right_candidates = [(11, 9), (11, -19), (-11, 9), (-11, -19),
                        (11, 26), (11, -36), (-11, 26), (-11, -36),
                        (11, 43), (11, -53)]
    # Near the right edge, try the left side first so labels stay in frame.
    left_candidates = [(-dx, dy) for dx, dy in right_candidates]
    x_mid = sum(ax.get_xlim()) / 2

    for i in order:
        chosen = None
        candidates = left_candidates if xs[i] > x_mid else right_candidates
        for dx, dy in candidates:
            ann = ax.annotate(texts[i], (xs[i], ys[i]),
                              textcoords="offset points", xytext=(dx, dy),
                              ha="left" if dx > 0 else "right",
                              fontsize=fontsize, color=color,
                              bbox=dict(facecolor=SURFACE, edgecolor="none",
                                        pad=1.2))
            bb = ann.get_window_extent(renderer)
            box = (bb.x0 - 2, bb.y0 - 1, bb.x1 + 2, bb.y1 + 1)
            if not overlaps(box):
                boxes.append(box)
                chosen = ann
                break
            ann.remove()
        if chosen is None:      # everything collided: accept the first slot
            ann = ax.annotate(texts[i], (xs[i], ys[i]),
                              textcoords="offset points", xytext=(11, 9),
                              ha="left", fontsize=fontsize, color=color,
                              bbox=dict(facecolor=SURFACE, edgecolor="none",
                                        pad=1.2))
            boxes.append(tuple(ann.get_window_extent(renderer).extents))
