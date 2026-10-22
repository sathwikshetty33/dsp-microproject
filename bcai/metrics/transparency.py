"""What an auditor can still verify at each privacy posture.

Every metric here is computed from the *same published record* the attackers
read.  That is the whole argument of the project: the auditor and the adversary
are reading the same bytes, so a mitigation that blinds one blinds the other.

Five capabilities are scored, each in [0, 1]:

``freerider_detection``   catching a participant that claims rewards without
                          doing the work (AUC, rescaled so 0.5 -> 0).
``poisoning_detection``   catching a participant submitting corrupted updates.
``contribution_fidelity`` how well published contribution claims track reality.
``payment_traceability``  share of value flow an auditor can attribute.
``supply_conservation``   can every transaction be checked to balance?

The last one is included deliberately: it is the transparency property that
*survives* every mitigation, and it is worth being precise about which
guarantees a PET actually costs you.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import spearmanr

from ..attacks.clustering import ClusterReport
from ..fl.federated import FLResult
from ..metrics.privacy import attack_auc
from ..synth.ledger import Ledger


def _rescale_auc(auc: float) -> float:
    """Map an AUC onto [0, 1] where 0.5 (chance) becomes 0."""
    return float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))


@dataclass
class TransparencyReport:
    freerider_detection_auc: float
    poisoning_detection_auc: float
    contribution_fidelity: float
    payment_traceability: float
    supply_conservation: float
    metadata_completeness: float
    audit_capability_index: float
    updates_visible: bool

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def verify_supply_conservation(ledger: Ledger) -> float:
    """Fraction of transactions an auditor can check balance to the satoshi."""
    io = ledger.tx_io
    ins = io[io["side"] == "in"].groupby("txid")["value"].sum()
    outs = io[io["side"] == "out"].groupby("txid")["value"].sum()
    tx = ledger.transactions.set_index("txid")
    checkable = tx[tx["kind"] != "coinbase"]
    if checkable.empty:
        return 1.0
    lhs = ins.reindex(checkable.index).fillna(0.0)
    rhs = outs.reindex(checkable.index).fillna(0.0) + checkable["fee"]
    return float((np.abs(lhs - rhs) < 1e-3).mean())


def detect_freeriders(fl: FLResult) -> float:
    """Score each client on how closely it echoes the previous aggregate.

    A free-rider that replays the global movement is highly correlated with
    the aggregate and contributes no independent signal.  Honest clients,
    training on their own non-IID shard, are not.
    """
    if fl.updates is None:
        return 0.5                                  # no per-client data to audit
    U = fl.updates                                  # (rounds, clients, dim)
    n_rounds, n_clients, _ = U.shape
    scores = np.zeros(n_clients)
    for c in range(n_clients):
        sims = []
        for r in range(1, n_rounds):
            prev_agg = U[r - 1].mean(axis=0)
            u = U[r, c]
            denom = np.linalg.norm(prev_agg) * np.linalg.norm(u)
            if denom > 0:
                sims.append(float(u @ prev_agg / denom))
        scores[c] = float(np.mean(sims)) if sims else 0.0
    labels = fl.client_truth["is_freerider"].to_numpy().astype(int)
    return attack_auc(scores, labels)


def detect_poisoners(fl: FLResult) -> float:
    """Flag clients whose updates point away from the federation consensus."""
    if fl.updates is None:
        return 0.5
    U = fl.updates
    n_rounds, n_clients, _ = U.shape
    scores = np.zeros(n_clients)
    for c in range(n_clients):
        sims = []
        for r in range(n_rounds):
            others = np.delete(U[r], c, axis=0).mean(axis=0)
            u = U[r, c]
            denom = np.linalg.norm(others) * np.linalg.norm(u)
            if denom > 0:
                sims.append(float(u @ others / denom))
        scores[c] = -float(np.mean(sims)) if sims else 0.0   # low similarity = suspect
    labels = fl.client_truth["is_poisoner"].to_numpy().astype(int)
    return attack_auc(scores, labels)


def contribution_fidelity(fl: FLResult) -> float:
    """Rank correlation between what clients publish and what they actually hold."""
    truth = fl.client_truth
    rho = spearmanr(truth["published_samples"], truth["true_samples"]).statistic
    if np.isnan(rho):
        return 0.0
    return float(np.clip(rho, 0.0, 1.0))


def build_transparency_report(ledger: Ledger, fl: FLResult,
                              cluster: ClusterReport,
                              metadata_completeness: float) -> TransparencyReport:
    fr = detect_freeriders(fl)
    po = detect_poisoners(fl)
    fidelity = contribution_fidelity(fl)
    # An auditor tracing funds has exactly the adversary's clustering power.
    traceability = float(cluster.value_attributed * cluster.attribution_precision)
    conservation = verify_supply_conservation(ledger)

    components = [
        _rescale_auc(fr),
        _rescale_auc(po),
        fidelity,
        traceability,
        conservation,
        metadata_completeness,
    ]
    return TransparencyReport(
        freerider_detection_auc=fr,
        poisoning_detection_auc=po,
        contribution_fidelity=fidelity,
        payment_traceability=traceability,
        supply_conservation=conservation,
        metadata_completeness=metadata_completeness,
        audit_capability_index=float(np.mean(components)),
        updates_visible=fl.updates is not None,
    )
