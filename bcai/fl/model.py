"""Logistic regression over random Fourier features.

Random features give the model enough capacity to *memorise* individual
training rows.  That is not an accident: memorisation is the precondition for
membership inference, and a model with no capacity to memorise would make the
privacy analysis vacuous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureMap:
    """Fixed random Fourier feature expansion, shared by every client."""

    W: np.ndarray   # (n_features, n_components)
    b: np.ndarray   # (n_components,)

    @property
    def dim(self) -> int:
        return int(self.W.shape[1] + 1)  # + intercept

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = np.sqrt(2.0 / self.W.shape[1]) * np.cos(X @ self.W + self.b)
        return np.hstack([Z, np.ones((Z.shape[0], 1))])


def make_feature_map(n_features: int, n_components: int, gamma: float,
                     rng: np.random.Generator) -> FeatureMap:
    W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(n_features, n_components))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    return FeatureMap(W=W, b=b)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def per_example_loss(Z: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Binary cross-entropy for each row -- the membership-inference signal."""
    p = np.clip(sigmoid(Z @ w), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_loss(Z: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 0.0) -> float:
    return float(per_example_loss(Z, y, w).mean() + 0.5 * l2 * float(w @ w))


def gradient(Z: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 0.0) -> np.ndarray:
    p = sigmoid(Z @ w)
    return Z.T @ (p - y) / Z.shape[0] + l2 * w


def accuracy(Z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(((sigmoid(Z @ w) >= 0.5).astype(float) == y).mean())


def local_train(Z: np.ndarray, y: np.ndarray, w0: np.ndarray, steps: int,
                lr: float, l2: float = 0.0) -> np.ndarray:
    """Plain full-batch gradient descent from the current global weights."""
    w = w0.copy()
    for _ in range(steps):
        w -= lr * gradient(Z, y, w, l2)
    return w


def clip_update(delta: np.ndarray, clip_norm: float) -> tuple[np.ndarray, float]:
    """L2-clip a client update; returns the clipped update and its true norm."""
    norm = float(np.linalg.norm(delta))
    if norm > clip_norm and norm > 0:
        return delta * (clip_norm / norm), norm
    return delta, norm
