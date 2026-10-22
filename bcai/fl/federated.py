"""FedAvg over a public ledger, with free-riders, poisoners and DP.

Each round every client publishes a record on chain.  What that record
*contains* is the dial this whole study turns:

* posture 0-2  -- exact sample count, update L2 norm, local loss, plain QIs;
* posture 3    -- quasi-identifiers generalised before publication;
* posture 4    -- counts under a Laplace mechanism, updates clipped + Gaussian;
* posture 5    -- only a commitment and a validity proof.

Everything an auditor can compute and everything an attacker can compute is
derived from exactly that published record, which is what makes the
transparency/privacy comparison fair.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import FLConfig, Posture
from ..synth.entities import EntityUniverse, Role
from ..synth.ledger import EPOCH
from .data import Dataset, make_dataset
from .model import (
    FeatureMap,
    accuracy,
    clip_update,
    local_train,
    make_feature_map,
    mean_loss,
)

ROUND_INTERVAL = 6 * 3_600  # one federated round every six hours

#: The honest, noise-free reference federation depends only on the data and the
#: hyper-parameters -- never on the posture -- so it is computed once and reused.
_CLEAN_REFERENCE_CACHE: dict[tuple, float] = {}


@dataclass
class MIABundle:
    """Everything a membership-inference adversary can actually use.

    ``local_models`` is reconstructed purely from public chain state --
    the global weights plus the per-client update the client published.  When a
    posture hides updates (secure aggregation / ZK) it is ``None``, and the
    attack simply cannot be mounted.
    """

    local_models: np.ndarray | None      # (n_clients, dim)
    global_model: np.ndarray             # (dim,)
    Z_in: list[np.ndarray]
    y_in: list[np.ndarray]
    Z_out: list[np.ndarray]
    y_out: list[np.ndarray]


@dataclass
class FLResult:
    posture_level: int
    onchain: pd.DataFrame        # what the chain actually publishes
    history: pd.DataFrame        # per-round global model quality
    client_truth: pd.DataFrame   # ground truth, never published
    updates: np.ndarray | None   # (rounds, clients, dim) submitted updates, or
                                 # None when secure aggregation hides them
    mia: MIABundle
    final_accuracy: float
    clean_accuracy_reference: float
    dp_epsilon: float | None


def _laplace(rng: np.random.Generator, scale: float, size=None) -> np.ndarray | float:
    return rng.laplace(0.0, scale, size=size)


def dp_epsilon(noise_multiplier: float, rounds: int, delta: float = 1e-5) -> float | None:
    """(eps, delta)-DP for `rounds` compositions of a Gaussian mechanism.

    Uses the standard RDP accountant for the Gaussian mechanism,
    eps_RDP(alpha) = alpha / (2 * sigma^2) per round, then converts to
    approximate DP and minimises over alpha.
    """
    if noise_multiplier <= 0:
        return None
    alphas = np.arange(1.05, 128.0, 0.05)
    rdp = rounds * alphas / (2.0 * noise_multiplier**2)
    eps = rdp + np.log1p(-1.0 / alphas) - (np.log(delta) + np.log(alphas)) / (alphas - 1.0)
    return float(np.min(eps))


def _commitment(round_id: int, client_id: int, delta: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{round_id}:{client_id}:".encode())
    h.update(np.ascontiguousarray(delta, dtype=np.float64).tobytes())
    return "0x" + h.hexdigest()[:32]


def _generalise(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """k-anonymity by generalisation + suppression of the published QIs."""
    out = df.copy()
    continent = {"EU-N": "EU", "EU-S": "EU", "NA-E": "NA", "NA-W": "NA",
                 "APAC": "APAC", "LATAM": "LATAM"}
    out["region"] = out["region"].map(lambda r: continent.get(r, "OTHER"))
    out["org_size"] = out["org_size"].map(
        {"micro": "small", "small": "small", "medium": "large", "large": "large"}
    )
    out["active_hour"] = (out["active_hour"] // 6) * 6          # 6-hour blocks
    out["samples_bucket"] = (out["published_samples"] // 500) * 500
    qis = ["region", "org_size", "active_hour", "samples_bucket"]
    sizes = out.groupby(qis, observed=True)[qis[0]].transform("size")
    out.loc[sizes < k, qis] = np.nan     # suppress groups below k
    out["suppressed"] = sizes < k
    return out


def run_federation(
    cfg: FLConfig,
    posture: Posture,
    universe: EntityUniverse,
    seed: int,
) -> FLResult:
    # Two independent streams.  `world_rng` fixes the experiment (data, feature
    # map, who cheats) identically for every posture; `mech_rng` draws only the
    # posture-dependent randomness (DP noise, publication jitter).  Without this
    # split the posture comparison would measure resampling noise, not PETs.
    world_rng = np.random.default_rng(seed + 1_000)
    mech_rng = np.random.default_rng(seed + 7_000 + posture.level * 31)
    rng = world_rng

    # ---- bind federated clients to real ledger entities ---------------------
    candidates = universe.by_role(Role.HOSPITAL) + universe.by_role(Role.CLINIC)
    if len(candidates) < cfg.n_clients:
        candidates = candidates + universe.by_role(Role.RESEARCH_LAB)
    picks = rng.choice(len(candidates), size=cfg.n_clients, replace=False)
    clients = [candidates[int(i)] for i in picks]
    regions = tuple(dict.fromkeys(e.region for e in clients))

    data = make_dataset(cfg.n_clients, cfg.samples_per_client, cfg.n_features,
                        cfg.holdout_size, regions, rng, cfg.signal_strength)
    fmap: FeatureMap = make_feature_map(cfg.n_features, cfg.n_components,
                                        cfg.rff_gamma, rng)

    # ---- assign adversarial roles ------------------------------------------
    ids = rng.permutation(cfg.n_clients)
    freeriders = set(int(i) for i in ids[: cfg.n_freeriders])
    poisoners = set(int(i) for i in ids[cfg.n_freeriders: cfg.n_freeriders + cfg.n_poisoners])

    Z_local, y_local = [], []
    for shard in data.shards:
        y = shard.y.copy()
        if shard.client_id in poisoners:
            flip = rng.random(y.shape[0]) < cfg.poison_flip_rate
            y[flip] = 1.0 - y[flip]
        Z_local.append(fmap(shard.X))
        y_local.append(y)

    Z_hold, y_hold = fmap(data.X_holdout), data.y_holdout
    dim = fmap.dim

    # ---- claimed sample counts (free-riders inflate) ------------------------
    true_counts = np.array([s.n for s in data.shards], dtype=float)
    claimed = true_counts.copy()
    for cid in freeriders:
        claimed[cid] = true_counts[cid] * rng.uniform(2.5, 4.0)

    # DP protection on the published counts also blinds the aggregator's
    # weighting, which is exactly the tension this experiment measures.
    if posture.dp_count_epsilon:
        sensitivity = cfg.dp_count_sensitivity
        noisy = claimed + _laplace(mech_rng, sensitivity / posture.dp_count_epsilon,
                                   size=claimed.shape)
        published_counts = np.maximum(1.0, noisy)
    else:
        published_counts = claimed.copy()

    noise_std = (posture.dp_noise_multiplier * cfg.clip_norm
                 / np.sqrt(cfg.n_clients)) if posture.dp_noise_multiplier > 0 else 0.0

    w = np.zeros(dim)
    updates = np.zeros((cfg.n_rounds, cfg.n_clients, dim))
    records: list[dict] = []
    history: list[dict] = []
    prev_global_delta = np.zeros(dim)

    for r in range(cfg.n_rounds):
        round_updates = np.zeros((cfg.n_clients, dim))
        true_norms = np.zeros(cfg.n_clients)
        local_losses = np.zeros(cfg.n_clients)

        for cid in range(cfg.n_clients):
            if cid in freeriders:
                # A lazy but not stupid free-rider: replay the last global
                # movement with a little noise, so the update looks alive.
                delta = prev_global_delta * world_rng.uniform(0.8, 1.15) + world_rng.normal(
                    0, 0.02, size=dim
                )
                local_losses[cid] = float(mean_loss(Z_local[cid], y_local[cid], w, cfg.l2))
            else:
                w_local = local_train(Z_local[cid], y_local[cid], w,
                                      cfg.local_steps, cfg.learning_rate, cfg.l2)
                delta = w_local - w
                local_losses[cid] = float(mean_loss(Z_local[cid], y_local[cid], w_local, cfg.l2))

            delta, raw_norm = clip_update(delta, cfg.clip_norm)
            true_norms[cid] = raw_norm
            if noise_std > 0:
                delta = delta + mech_rng.normal(0, noise_std, size=dim)
            round_updates[cid] = delta

        weights = published_counts / published_counts.sum()
        aggregate = (round_updates * weights[:, None]).sum(axis=0)
        w_new = w + aggregate
        prev_global_delta = w_new - w
        w = w_new
        updates[r] = round_updates
        last_round_updates = round_updates.copy()

        acc = accuracy(Z_hold, y_hold, w)
        history.append(
            dict(round=r, global_accuracy=acc,
                 global_loss=mean_loss(Z_hold, y_hold, w),
                 mean_update_norm=float(np.linalg.norm(round_updates, axis=1).mean()))
        )

        for cid in range(cfg.n_clients):
            e = clients[cid]
            hour = int(world_rng.choice(24, p=e.activity_weights()))
            ts = EPOCH + r * ROUND_INTERVAL + hour * 3_600 + int(mech_rng.integers(0, 600))
            rec = dict(
                round=r,
                client_id=cid,
                address=e.addresses[0],
                entity_id=e.eid,
                region=e.region,
                org_size=e.org_size,
                active_hour=hour,
                timestamp=ts,
                model_id="medfl-v1",
                published_samples=float(published_counts[cid]),
                commitment=_commitment(r, cid, round_updates[cid]),
            )
            if posture.hide_update_metadata:
                # Only a commitment and a proof bit reach the chain.
                rec["update_norm"] = np.nan
                rec["local_loss"] = np.nan
                rec["proof_valid"] = True
            else:
                rec["update_norm"] = float(np.linalg.norm(round_updates[cid]))
                rec["local_loss"] = float(local_losses[cid])
                rec["proof_valid"] = True
            records.append(rec)

    onchain = pd.DataFrame(records)
    if posture.generalise_metadata:
        onchain = _generalise(onchain, posture.suppression_k)
    else:
        onchain["samples_bucket"] = onchain["published_samples"].round()
        onchain["suppressed"] = False

    # ---- assemble the membership-inference attack surface -------------------
    # Non-members are drawn from each client's OWN distribution, so the attack
    # AUC measures memorisation rather than covariate shift.
    Z_out = [fmap(s.X_out) for s in data.shards]
    y_out = [s.y_out.copy() for s in data.shards]
    for cid in poisoners:                       # a poisoner's own labels are flipped
        flip = world_rng.random(y_out[cid].shape[0]) < cfg.poison_flip_rate
        y_out[cid][flip] = 1.0 - y_out[cid][flip]

    if posture.hide_update_metadata:
        local_models = None                     # nothing to reconstruct from
    else:
        local_models = w[None, :] + last_round_updates
    mia = MIABundle(local_models=local_models, global_model=w,
                    Z_in=Z_local, y_in=y_local, Z_out=Z_out, y_out=y_out)

    # Reference: what the same federation reaches with no adversaries and no DP.
    cache_key = (seed, cfg.n_clients, cfg.samples_per_client, cfg.n_features,
                 cfg.n_components, cfg.n_rounds, cfg.local_steps,
                 cfg.learning_rate, cfg.clip_norm, cfg.l2, cfg.rff_gamma)
    if cache_key not in _CLEAN_REFERENCE_CACHE:
        _CLEAN_REFERENCE_CACHE[cache_key] = _clean_reference(
            cfg, data, fmap, Z_hold, y_hold
        )
    clean_ref = _CLEAN_REFERENCE_CACHE[cache_key]

    client_truth = pd.DataFrame(
        dict(
            client_id=np.arange(cfg.n_clients),
            entity_id=[e.eid for e in clients],
            address=[e.addresses[0] for e in clients],
            region=[e.region for e in clients],
            org_size=[e.org_size for e in clients],
            true_samples=true_counts,
            claimed_samples=claimed,
            published_samples=published_counts,
            label_rate=[s.label_rate for s in data.shards],
            is_freerider=[c in freeriders for c in range(cfg.n_clients)],
            is_poisoner=[c in poisoners for c in range(cfg.n_clients)],
        )
    )

    return FLResult(
        posture_level=posture.level,
        onchain=onchain,
        history=pd.DataFrame(history),
        client_truth=client_truth,
        updates=None if posture.hide_update_metadata else updates,
        mia=mia,
        final_accuracy=float(accuracy(Z_hold, y_hold, w)),
        clean_accuracy_reference=clean_ref,
        dp_epsilon=dp_epsilon(posture.dp_noise_multiplier, cfg.n_rounds),
    )


def _clean_reference(cfg: FLConfig, data: Dataset, fmap: FeatureMap,
                     Z_hold: np.ndarray, y_hold: np.ndarray) -> float:
    """Accuracy of an honest, noise-free federation -- the utility ceiling."""
    Z = [fmap(s.X) for s in data.shards]
    y = [s.y for s in data.shards]
    counts = np.array([s.n for s in data.shards], dtype=float)
    weights = counts / counts.sum()
    w = np.zeros(fmap.dim)
    for _ in range(cfg.n_rounds):
        agg = np.zeros(fmap.dim)
        for cid in range(cfg.n_clients):
            wl = local_train(Z[cid], y[cid], w, cfg.local_steps, cfg.learning_rate, cfg.l2)
            delta, _ = clip_update(wl - w, cfg.clip_norm)
            agg += weights[cid] * delta
        w += agg
    return float(accuracy(Z_hold, y_hold, w))
