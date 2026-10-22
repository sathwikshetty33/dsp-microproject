"""Synthetic, non-IID tabular data for the federated task.

The task stands in for a clinical risk model: 14 raw features, one of which is
a *sensitive attribute* that is correlated with the label.  Clients hold
region-shifted, label-skewed shards, which is what makes per-client updates
informative enough to leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SENSITIVE_IDX = 0  # feature 0 is the sensitive attribute (e.g. a protected group)


@dataclass
class Shard:
    """One client's data.

    ``X_out``/``y_out`` are drawn from *the same distribution* as this client's
    training rows but were never trained on.  A membership-inference attack
    must be scored against these, not against a generic population holdout --
    otherwise the AUC measures distribution shift rather than memorisation.
    """

    client_id: int
    X: np.ndarray
    y: np.ndarray
    X_out: np.ndarray
    y_out: np.ndarray
    region: str
    skew: float
    label_rate: float

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


@dataclass
class Dataset:
    shards: list[Shard]
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    w_true: np.ndarray

    @property
    def X_members(self) -> np.ndarray:
        return np.vstack([s.X for s in self.shards])

    @property
    def y_members(self) -> np.ndarray:
        return np.concatenate([s.y for s in self.shards])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_dataset(
    n_clients: int,
    samples_per_client: int,
    n_features: int,
    holdout_size: int,
    regions: tuple[str, ...],
    rng: np.random.Generator,
    signal_strength: float = 0.45,
) -> Dataset:
    """Draw a shared ground-truth concept, then region-shifted client shards.

    ``signal_strength`` sets how deterministic the label is.  It is kept low on
    purpose: a clinical risk label is genuinely noisy (Bayes accuracy here is
    around 0.72), and that irreducible noise is what a model has to *memorise*
    rather than learn -- which is precisely what membership inference detects.
    """
    w_true = rng.normal(0, signal_strength, size=n_features)
    w_true[SENSITIVE_IDX] *= 1.6  # the sensitive attribute genuinely matters
    bias = float(rng.normal(0, 0.15))

    # Per-region covariate shift: this is what makes shards non-IID.
    region_shift = {r: rng.normal(0, 0.55, size=n_features) for r in regions}

    def draw(n: int, region: str, skew: float) -> tuple[np.ndarray, np.ndarray]:
        X = rng.normal(0, 1.0, size=(n, n_features)) + region_shift[region]
        X[:, SENSITIVE_IDX] = rng.binomial(1, 0.35 + 0.25 * skew, size=n).astype(float)
        logits = X @ w_true + bias + 0.6 * (skew - 0.5)
        p = _sigmoid(logits)
        y = rng.binomial(1, p).astype(float)
        return X, y

    shards: list[Shard] = []
    for cid in range(n_clients):
        region = regions[cid % len(regions)]
        skew = float(rng.beta(2.0, 2.0))
        # Site sizes vary by nearly an order of magnitude, as in real federations.
        n = int(np.clip(rng.lognormal(np.log(samples_per_client), 0.45), 60, samples_per_client * 3))
        X, y = draw(n, region, skew)
        X_out, y_out = draw(n, region, skew)   # same distribution, never trained on
        shards.append(
            Shard(client_id=cid, X=X, y=y, X_out=X_out, y_out=y_out,
                  region=region, skew=skew, label_rate=float(y.mean()))
        )

    Xh_parts, yh_parts = [], []
    per_region = max(1, holdout_size // len(regions))
    for region in regions:
        X, y = draw(per_region, region, 0.5)
        Xh_parts.append(X)
        yh_parts.append(y)
    return Dataset(shards, np.vstack(Xh_parts), np.concatenate(yh_parts), w_true)
