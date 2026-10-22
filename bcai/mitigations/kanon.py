"""Generalisation and suppression -- the k-anonymity mitigation.

Applied to whatever quasi-identifiers a contract publishes.  Generalisation
coarsens a value (a region becomes a continent, an hour becomes a six-hour
block); suppression removes a row's quasi-identifiers entirely when its
equivalence class is still smaller than k.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

REGION_TO_CONTINENT = {
    "EU-N": "EU", "EU-S": "EU", "NA-E": "NA", "NA-W": "NA",
    "APAC": "APAC", "LATAM": "LATAM",
}
ORG_SIZE_COARSE = {"micro": "small", "small": "small", "medium": "large", "large": "large"}
VOLUME_COARSE = {"1-2": "low", "3-5": "low", "6-10": "mid", "11-20": "mid", "20+": "high"}

#: column -> coarsening function, used when a posture enables generalisation
DEFAULT_SPEC: dict[str, Callable[[pd.Series], pd.Series]] = {
    "region": lambda s: s.map(lambda v: REGION_TO_CONTINENT.get(v, "OTHER")),
    "org_size": lambda s: s.map(lambda v: ORG_SIZE_COARSE.get(v, v)),
    "volume_band": lambda s: s.map(lambda v: VOLUME_COARSE.get(v, v)),
    "top_model": lambda s: s.map(lambda v: "family:" + str(v).split("-")[0]),
}


def equivalence_class_sizes(df: pd.DataFrame, qis: list[str]) -> pd.Series:
    """Size of each row's quasi-identifier equivalence class (its k)."""
    present = [q for q in qis if q in df.columns]
    if not present or df.empty:
        return pd.Series(np.ones(len(df)), index=df.index)
    # Suppressed cells become a shared wildcard, so suppressed rows collapse
    # into one large equivalence class -- which is exactly what suppression is
    # supposed to achieve.
    keyed = df[present].fillna("*").astype(str).agg("|".join, axis=1)
    return keyed.map(keyed.value_counts())


def generalise(df: pd.DataFrame, qis: list[str],
               spec: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    spec = DEFAULT_SPEC if spec is None else spec
    for col, fn in spec.items():
        if col in out.columns:
            out[col] = fn(out[col])
    return out


def suppress_below_k(df: pd.DataFrame, qis: list[str], k: int) -> pd.DataFrame:
    """Blank the quasi-identifiers of any row in a class smaller than k."""
    out = df.copy()
    sizes = equivalence_class_sizes(out, qis)
    mask = sizes < k
    present = [q for q in qis if q in out.columns]
    for col in present:
        out[col] = out[col].astype(object)
    out.loc[mask, present] = np.nan
    out["suppressed"] = mask.to_numpy()
    return out


def apply_k_anonymity(df: pd.DataFrame, qis: list[str], k: int,
                      spec: dict | None = None) -> pd.DataFrame:
    return suppress_below_k(generalise(df, qis, spec), qis, k)


def adaptive_k_anonymity(df: pd.DataFrame, qis: list[str], k: int,
                         max_suppression: float = 0.2,
                         spec: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Full-domain generalisation with a suppression budget.

    Publishing four quasi-identifiers about 83 organisations and demanding
    k=5 is impossible -- every equivalence class is a singleton and naive
    suppression blanks the entire table.  That is the curse of dimensionality
    for k-anonymity (Aggarwal, 2005), and a real deployment resolves it by
    *dropping* attributes until the remaining ones can carry k.

    Attributes are dropped highest-cardinality first, because those are the
    ones fragmenting the classes.  Returns the published table and the list of
    quasi-identifiers that survived -- the survivors are what the chain can
    still legitimately disclose, and the drops are a transparency cost.
    """
    generalised = generalise(df, qis, spec)
    remaining = [q for q in qis if q in generalised.columns]

    while remaining:
        sizes = equivalence_class_sizes(generalised, remaining)
        if float((sizes < k).mean()) <= max_suppression:
            break
        cardinality = {q: generalised[q].nunique() for q in remaining}
        drop = max(cardinality, key=lambda q: (cardinality[q], q))
        remaining = [q for q in remaining if q != drop]

    dropped = [q for q in qis if q not in remaining]
    out = generalised.copy()
    for col in dropped:
        if col in out.columns:
            out[col] = np.nan            # no longer published at all
    out = suppress_below_k(out, remaining, k) if remaining else out.assign(
        suppressed=True
    )
    return out, remaining
