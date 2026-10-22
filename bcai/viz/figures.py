"""Figures for the assessment.

Design rules followed throughout: one measure per axis (never a second
y-scale), categorical hues assigned in fixed slot order and never cycled,
a legend whenever two or more series are present, direct value labels on every
bar (required relief for the two palette slots below 3:1 contrast), a recessive
grid, and a surface-coloured gap between adjacent marks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .theme import (
    GRID,
    header,
    SEQUENTIAL,
    SERIES,
    STATUS,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_theme,
    finish,
    rounded_bars,
    place_labels,
    save,
    value_labels,
)


def _levels(results: pd.DataFrame) -> list[str]:
    return [f"L{int(v)}" for v in results["level"]]


def _grouped_bars(ax, results, specs, *, ylabel, title, subtitle,
                  fmt="{:.2f}", ylim=(0, 1.08)):
    """specs: list of (column, label, colour)."""
    n = len(specs)
    x = np.arange(len(results))
    slot = 0.82 / n
    for i, (col, label, colour) in enumerate(specs):
        offset = (i - (n - 1) / 2) * slot
        heights = results[col].to_numpy(dtype=float)
        rounded_bars(ax, x + offset, heights, slot, colour, label=label)
        value_labels(ax, x + offset, heights, fmt=fmt, fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(_levels(results))
    ax.set_xlim(-0.6, len(results) - 0.4)
    finish(ax, title=title, subtitle=subtitle, ylabel=ylabel,
           xlabel="Privacy posture", ylim=ylim, legend_ncols=len(specs))


def fig_tradeoff(results: pd.DataFrame, path: Path) -> None:
    """The headline: what each mitigation buys and what it costs."""
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    x = results["audit_capability_index"].to_numpy(float)
    y = results["privacy_exposure_index"].to_numpy(float)
    xerr = results.get("audit_capability_index_std")
    yerr = results.get("privacy_exposure_index_std")

    ax.plot(x, y, color=SERIES[0], linewidth=2.0, zorder=2, alpha=0.55)
    if xerr is not None and yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=GRID, elinewidth=1.4, capsize=0, zorder=1)
    ax.scatter(x, y, s=90, color=SERIES[0], edgecolor=SURFACE,
               linewidth=2.0, zorder=3)

    place_labels(ax, x, y,
                 [f"L{int(l)}  {n}" for l, n in
                  zip(results["level"], results["posture"])])

    ax.set_xlim(0.0, 1.10)
    ax.set_ylim(0, 0.95)
    finish(ax,
           title="Every privacy gain is paid for in audit capability",
           subtitle="Down is more private; right is more auditable.\n"
                    "No posture reaches the bottom-right corner.",
           ylabel="Privacy exposure index  (lower = more private)",
           xlabel="Audit capability index  (higher = more verifiable)",
           legend=False)
    ax.grid(axis="x", visible=True)
    save(fig, path)


def fig_attack_ladder(results: pd.DataFrame, path: Path) -> None:
    """Mixing only looks strong until a second heuristic is layered on."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _grouped_bars(
        ax, results,
        [("naive_analyst_precision", "Naive analyst (H1 everywhere)", SERIES[0]),
         ("competent_analyst_value", "Competent (excludes CoinJoins)", SERIES[1]),
         ("payment_deanonymisation", "Advanced (+ change arithmetic)", SERIES[2])],
        ylabel="Share of transaction value correctly attributed",
        title="Mixing defeats one heuristic, not an analyst",
        subtitle="Change outputs let an adversary solve CoinJoins by arithmetic, "
                 "recovering most of the lost coverage.")
    save(fig, path)


def fig_reidentification(results: pd.DataFrame, path: Path) -> None:
    """k-anonymity cannot touch the one field the chain must publish."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _grouped_bars(
        ax, results,
        [("reid_declared_only", "Declared fields only", SERIES[0]),
         ("reid_with_timestamps", "Declared fields + block timestamps", SERIES[1])],
        ylabel="Share of participants uniquely re-identified",
        title="A blockchain cannot k-anonymise its own clock",
        subtitle="Generalisation suppresses the declared quasi-identifiers; "
                 "the timestamp channel survives untouched.")
    save(fig, path)


def fig_membership(results: pd.DataFrame, path: Path) -> None:
    """Publishing per-client updates is what makes the model memorisable."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(results))
    available = results["mia_available"].astype(bool).to_numpy()
    specs = [("mia_auc", "Attack on published per-client update", SERIES[0]),
             ("mia_global_auc", "Attack on the global model only", SERIES[1])]
    slot = 0.82 / len(specs)
    for i, (col, label, colour) in enumerate(specs):
        offset = (i - (len(specs) - 1) / 2) * slot
        h = results[col].to_numpy(float)
        # Where the attack has no target at all, draw nothing rather than a
        # bar sitting at chance -- an absent bar is the honest encoding.
        mask = available if col == "mia_auc" else np.ones_like(available)
        xs = (x + offset)[mask]
        hs = h[mask]
        rounded_bars(ax, xs, hs, slot, colour, label=label)
        value_labels(ax, xs, hs, fmt="{:.3f}", fontsize=7.6, dy=0.006)

    ax.axhline(0.5, color=TEXT_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax.text(-0.52, 0.502, "chance", fontsize=8.2,
            color=TEXT_MUTED, ha="left", va="bottom")
    for i in np.flatnonzero(~available):
        ax.text(i - slot / 2, 0.458, "no target\non chain", fontsize=7.6,
                color=STATUS["good"], ha="center", va="bottom", linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(_levels(results))
    ax.set_xlim(-0.6, len(results) - 0.4)
    finish(ax,
           title="Membership inference feeds on the audit trail",
           subtitle="Anyone can rebuild a client's local model from public "
                    "state:\nw_client = w_global + published update.",
           ylabel="Attack AUC", xlabel="Privacy posture",
           ylim=(0.45, 0.78), legend_ncols=2)
    save(fig, path)


def fig_audit_heatmap(results: pd.DataFrame, path: Path) -> None:
    """Which specific audit powers each mitigation removes."""
    cols = [
        ("freerider_detection_auc", "Free-rider detection"),
        ("poisoning_detection_auc", "Poisoning detection"),
        ("contribution_fidelity", "Contribution fidelity"),
        ("payment_traceability", "Payment traceability"),
        ("metadata_completeness", "Metadata completeness"),
        ("supply_conservation", "Supply conservation"),
    ]
    matrix = np.array([results[c].to_numpy(float) for c, _ in cols])
    # Detection columns are AUCs: rescale so chance reads as zero capability.
    for i, (c, _) in enumerate(cols):
        if c.endswith("_auc"):
            matrix[i] = np.clip(2 * (matrix[i] - 0.5), 0, 1)

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(results)), _levels(results))
    ax.set_yticks(range(len(cols)), [label for _, label in cols])
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.4,
                    color="#ffffff" if v > 0.55 else TEXT_PRIMARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Privacy posture")
    header(ax,
           "Supply conservation is the only audit power left intact",
           "Capability retained: 0 = lost entirely, 1 = full.\n"
           "Detection rows are AUCs rescaled so chance reads as 0.")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=0)
    save(fig, path)


def fig_unicity(outcomes, path: Path) -> None:
    """How few attributes it takes to single someone out."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    # Postures 3-5 apply the same metadata mitigation and produce the same
    # curve, so plotting all three would just overprint one line.
    shown = [outcomes[0], outcomes[3]]
    for i, o in enumerate(shown):
        curve = o.reid.unicity
        if not curve:
            continue
        ks = sorted(int(k) for k in curve)
        vs = [curve[k] for k in ks]
        ax.plot(ks, vs, color=SERIES[i], marker="o", markeredgecolor=SURFACE,
                markeredgewidth=1.6, label=f"L{o.posture.level}  {o.posture.name}")
        ax.annotate(f"{vs[-1]:.2f}", (ks[-1], vs[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=8.4, color=SERIES[i])
    ax.set_xticks(range(1, 6))
    finish(ax,
           title="Unicity: a handful of innocuous attributes is enough",
           subtitle="Probability that a participant is unique given p known "
                    "quasi-identifiers.\nPostures 3-5 share one curve: they "
                    "apply the same metadata mitigation.",
           ylabel="P(participant is unique)",
           xlabel="Number of quasi-identifiers known to the adversary",
           ylim=(0, 1.08), legend_ncols=1, legend_rows=2)
    save(fig, path)


def fig_utility(results: pd.DataFrame, path: Path) -> None:
    """The third axis nobody gets to ignore: does the model still work?"""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(results))
    acc = results["fl_accuracy"].to_numpy(float)
    ref = float(results["fl_clean_reference"].mean())

    std = (results["fl_accuracy_std"].to_numpy(float)
           if "fl_accuracy_std" in results else np.zeros_like(acc))
    rounded_bars(ax, x, acc, 0.62, SERIES[0], label="Federated model accuracy")
    ax.errorbar(x, acc, yerr=std, fmt="none", ecolor=TEXT_MUTED,
                elinewidth=1.2, capsize=0)
    # Anchor the labels above the error bars, not on top of them.
    value_labels(ax, x, acc, fmt="{:.3f}", fontsize=8.2, dy=0.005,
                 at=acc + np.nan_to_num(std))
    ax.axhline(ref, color=SERIES[1], linewidth=2.0, linestyle=(0, (5, 3)),
               label=f"Honest, noise-free federation ({ref:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(_levels(results))
    ax.set_xlim(-0.6, len(results) - 0.4)
    finish(ax,
           title="Differential privacy is the only rung that costs real accuracy",
           subtitle="Mixing and k-anonymity touch the ledger, not the model; "
                    "noise touches the model.",
           ylabel="Held-out accuracy", xlabel="Privacy posture",
           ylim=(0.45, max(0.72, ref + 0.05)), legend_ncols=2)
    save(fig, path)


def render_all(results: pd.DataFrame, outcomes, figure_dir: Path) -> list[Path]:
    apply_theme()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("fig1_privacy_transparency_tradeoff.png", lambda p: fig_tradeoff(results, p)),
        ("fig2_attack_ladder.png", lambda p: fig_attack_ladder(results, p)),
        ("fig3_reidentification.png", lambda p: fig_reidentification(results, p)),
        ("fig4_membership_inference.png", lambda p: fig_membership(results, p)),
        ("fig5_audit_capability.png", lambda p: fig_audit_heatmap(results, p)),
        ("fig6_unicity.png", lambda p: fig_unicity(outcomes, p)),
        ("fig7_model_utility.png", lambda p: fig_utility(results, p)),
    ]
    written = []
    for name, fn in jobs:
        path = figure_dir / name
        fn(path)
        written.append(path)
        print(f"  wrote {path.name}")
    return written
