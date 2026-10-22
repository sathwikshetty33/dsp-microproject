"""Privacy metrics: k-anonymity, unicity, anonymity-set entropy, attack scores."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ..mitigations.kanon import equivalence_class_sizes


@dataclass
class KAnonReport:
    n_records: int
    min_k: float
    median_k: float
    mean_k: float
    unique_fraction: float        # rows in a class of size 1
    below_5_fraction: float
    anonymity_entropy_bits: float # mean log2(k): the adversary's residual doubt
    suppression_rate: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def k_anonymity_report(df: pd.DataFrame, qis: list[str]) -> KAnonReport:
    if df.empty:
        return KAnonReport(0, 0, 0, 0, 0, 0, 0, 0)
    sizes = equivalence_class_sizes(df, qis).astype(float)
    suppressed = df["suppressed"] if "suppressed" in df.columns else pd.Series(False, index=df.index)
    return KAnonReport(
        n_records=int(len(df)),
        min_k=float(sizes.min()),
        median_k=float(sizes.median()),
        mean_k=float(sizes.mean()),
        unique_fraction=float((sizes <= 1).mean()),
        below_5_fraction=float((sizes < 5).mean()),
        anonymity_entropy_bits=float(np.log2(sizes).mean()),
        suppression_rate=float(pd.Series(suppressed).fillna(False).mean()),
    )


def unicity_curve(df: pd.DataFrame, qis: list[str], rng: np.random.Generator,
                  n_trials: int = 400) -> dict[int, float]:
    """P(a record is unique | the adversary knows p random quasi-identifiers).

    The shape of this curve is the practical refutation of "we only publish
    coarse metadata": a handful of innocuous attributes is usually enough.
    """
    present = [q for q in qis if q in df.columns]
    out: dict[int, float] = {}
    if df.empty or not present:
        return {p: 0.0 for p in range(1, len(qis) + 1)}
    for p in range(1, len(present) + 1):
        hits = 0
        for _ in range(n_trials):
            cols = list(rng.choice(present, size=p, replace=False))
            sizes = equivalence_class_sizes(df, cols)
            row = int(rng.integers(0, len(df)))
            hits += int(sizes.iloc[row] <= 1)
        out[p] = hits / n_trials
    return out


def attack_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC with a defined value even when one class is missing."""
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.01) -> float:
    """TPR at a low false-positive rate -- the metric that matters when the
    adversary only needs a few confident hits, not an accurate average."""
    if len(np.unique(labels)) < 2:
        return target_fpr
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def advantage(auc: float) -> float:
    """Membership advantage, 2*(AUC - 0.5): 0 = no leakage, 1 = total."""
    return float(max(0.0, 2.0 * (auc - 0.5)))
