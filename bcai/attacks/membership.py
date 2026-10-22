"""Membership inference against federated updates published on chain.

The attack is the standard loss-threshold test (Yeom et al.): a record that a
model was trained on gets a lower loss than a statistically identical record it
never saw.  What makes it a *blockchain* problem is the target.

A client's local model is not secret here.  The global weights are public
(everyone needs them to participate) and the client's update is published as
part of the audit trail, so any observer can reconstruct

    w_client = w_global + delta_client

and test arbitrary records against it.  There is no need to compromise the
hospital, subvert the aggregator, or even be a participant.  Reading the chain
is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

from ..fl.federated import MIABundle
from ..fl.model import per_example_loss
from ..metrics.privacy import advantage, attack_auc, tpr_at_fpr


@dataclass
class MembershipReport:
    attack_available: bool          # False once updates leave the chain
    n_clients_attacked: int
    mean_auc: float
    max_auc: float
    mean_advantage: float
    tpr_at_1pct_fpr: float
    global_model_auc: float         # the much weaker "aggregate only" baseline
    per_client_auc: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _loss_attack(w: np.ndarray, Z_in: np.ndarray, y_in: np.ndarray,
                 Z_out: np.ndarray, y_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower loss => predicted member.  Scores are negated losses."""
    l_in = per_example_loss(Z_in, y_in, w)
    l_out = per_example_loss(Z_out, y_out, w)
    scores = -np.concatenate([l_in, l_out])
    labels = np.concatenate([np.ones(len(l_in)), np.zeros(len(l_out))])
    return scores, labels


def run_membership_attack(mia: MIABundle) -> MembershipReport:
    n_clients = len(mia.Z_in)

    # Baseline: what an observer learns from the aggregate model alone.
    g_scores, g_labels = [], []
    for c in range(n_clients):
        s, l = _loss_attack(mia.global_model, mia.Z_in[c], mia.y_in[c],
                            mia.Z_out[c], mia.y_out[c])
        g_scores.append(s)
        g_labels.append(l)
    global_auc = attack_auc(np.concatenate(g_scores), np.concatenate(g_labels))

    if mia.local_models is None:
        # Secure aggregation / ZK: the per-client update never reaches the
        # chain, so this attack has no target at all.
        return MembershipReport(
            attack_available=False, n_clients_attacked=0, mean_auc=0.5,
            max_auc=0.5, mean_advantage=0.0, tpr_at_1pct_fpr=0.01,
            global_model_auc=global_auc, per_client_auc=[],
        )

    aucs, all_scores, all_labels = [], [], []
    for c in range(n_clients):
        s, l = _loss_attack(mia.local_models[c], mia.Z_in[c], mia.y_in[c],
                            mia.Z_out[c], mia.y_out[c])
        aucs.append(attack_auc(s, l))
        all_scores.append(s)
        all_labels.append(l)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return MembershipReport(
        attack_available=True,
        n_clients_attacked=n_clients,
        mean_auc=float(np.mean(aucs)),
        max_auc=float(np.max(aucs)),
        mean_advantage=float(np.mean([advantage(a) for a in aucs])),
        tpr_at_1pct_fpr=tpr_at_fpr(scores, labels, 0.01),
        global_model_auc=global_auc,
        per_client_auc=[float(a) for a in aucs],
    )
