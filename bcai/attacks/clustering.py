"""Address-clustering de-anonymisation of the payment layer.

Two heuristics that any chain analyst applies on day one:

H1 *common input ownership* -- all inputs of a transaction are assumed to be
   controlled by one party.  CoinJoin exists specifically to falsify this.
H2 *change detection* -- in a two-output payment where exactly one output is a
   round number, the other is change and belongs to the sender.

The interesting output is not the clustering itself but what it buys an
adversary who knows **one** address per entity (a KYC record, a donation
address, an invoice): how much of the chain does that one fact unlock?
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from ..synth.ledger import Ledger


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:       # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def labels(self) -> dict[str, str]:
        return {a: self.find(a) for a in self.parent}


def _is_round(value: float) -> bool:
    for step in (100.0, 25.0, 10.0, 5.0, 1.0):
        if abs(value / step - round(value / step)) < 1e-9:
            return True
    return False


@dataclass
class ClusterReport:
    n_addresses: int
    n_clusters: int
    collapse_ratio: float          # 1 - clusters/addresses; 0 = no linkage gained
    purity: float                  # weighted share of the dominant entity per cluster
    adjusted_rand: float
    adjusted_mutual_info: float
    false_merge_rate: float        # clusters spanning >1 real entity
    entity_recovery_rate: float    # entities with >=50% of addresses in one cluster
    largest_cluster_share: float
    value_attributed: float        # share of output value linked to a named entity
    attribution_precision: float   # of that, the share attributed *correctly*

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def detect_coinjoins(ledger: Ledger) -> set[str]:
    """Spot joint transactions from public structure alone.

    The tell is the equal-denomination output pattern: two or more inputs and
    two or more outputs carrying exactly the same value.  A competent analyst
    does this *before* clustering, because applying common-input-ownership to a
    CoinJoin is what poisons the whole graph.  No ground-truth label is used.
    """
    outs = ledger.tx_io[ledger.tx_io["side"] == "out"]
    repeated = (
        outs.assign(v=outs["value"].round(6))
        .groupby(["txid", "v"]).size()
        .groupby(level=0).max()
    )
    n_in = ledger.transactions.set_index("txid")["n_in"]
    candidates = repeated[repeated >= 2].index
    return {t for t in candidates if int(n_in.get(t, 0)) >= 2}


def cluster_addresses(ledger: Ledger, use_change_heuristic: bool = True,
                      skip_coinjoins: bool = True) -> dict[str, str]:
    """Run H1 (+H2) and return address -> cluster-id.

    ``skip_coinjoins`` withholds the common-input-ownership assumption from
    transactions that look like joint spends -- the difference between a naive
    and a competent analyst.
    """
    uf = UnionFind()
    for addr in ledger.address_owner["address"]:
        uf.add(addr)

    io = ledger.tx_io
    ins = io[io["side"] == "in"]
    outs = io[io["side"] == "out"]
    in_by_tx = ins.groupby("txid")["address"].apply(list)
    out_by_tx = {t: list(zip(g["address"], g["value"]))
                 for t, g in outs.groupby("txid")}

    joint = detect_coinjoins(ledger) if skip_coinjoins else set()

    # H1: common input ownership, withheld from suspected joint spends.
    for txid, addrs in in_by_tx.items():
        if txid in joint:
            continue
        first = addrs[0]
        for a in addrs[1:]:
            uf.union(first, a)

    if use_change_heuristic:
        # H2: the odd-valued output of a two-output payment is change.
        n_out = ledger.transactions.set_index("txid")["n_out"]
        for txid, addrs in in_by_tx.items():
            if txid in joint or int(n_out.get(txid, 0)) != 2:
                continue
            outputs = out_by_tx.get(txid, [])
            if len(outputs) != 2:
                continue
            rounds = [_is_round(v) for _, v in outputs]
            if sum(rounds) == 1:
                change_addr = outputs[0][0] if not rounds[0] else outputs[1][0]
                uf.union(addrs[0], change_addr)

    return uf.labels()


def evaluate_clustering(ledger: Ledger, labels: dict[str, str]) -> ClusterReport:
    """Score a clustering against the ground-truth address ownership."""
    owner = ledger.owner_map
    addrs = [a for a in labels if a in owner]
    truth = np.array([owner[a] for a in addrs])
    pred_ids = pd.factorize(pd.Series([labels[a] for a in addrs]))[0]

    df = pd.DataFrame(dict(address=addrs, cluster=pred_ids, entity=truth))
    sizes = df.groupby("cluster").size()
    dominant = df.groupby(["cluster", "entity"]).size().groupby(level=0).max()
    purity = float((dominant.sum()) / len(df))
    n_entities_per_cluster = df.groupby("cluster")["entity"].nunique()
    false_merge = float((n_entities_per_cluster > 1).mean())

    # An entity counts as "recovered" only if a majority of its addresses land
    # in one cluster AND that entity dominates that cluster.  Without the second
    # condition a single giant false merge would score as perfect recovery.
    per_entity = df.groupby(["entity", "cluster"]).size()
    entity_tot = df.groupby("entity").size()
    best_idx = per_entity.groupby(level=0).idxmax()
    best_share = per_entity.groupby(level=0).max() / entity_tot
    dominant_entity_of = (
        df.groupby(["cluster", "entity"]).size().groupby(level=0).idxmax()
        .map(lambda t: t[1])
    )
    recovered = [
        bool(best_share[e] >= 0.5 and dominant_entity_of.get(best_idx[e][1]) == e)
        for e in entity_tot.index
    ]
    recovery = float(np.mean(recovered))

    # --- the adversary's actual gain -------------------------------------
    # The adversary holds one known address per entity (a KYC record, an
    # invoice, a published donation address).  Each known address labels the
    # cluster it sits in.  When several entities claim the same cluster -- which
    # is exactly what a false merge produces -- the analyst attributes it to the
    # claimant contributing the most addresses.  Deterministic by construction,
    # so the metric does not depend on row order.
    # The known set: each entity's long-lived first address (the one it also
    # uses to sign federated-learning updates).  Clusters with no known address
    # in them stay unattributed -- that is the adversary's coverage limit.
    known_addrs = {e.addresses[0] for e in ledger.universe.entities if e.addresses}
    claims = (
        df[df["address"].isin(known_addrs)]
        .groupby(["cluster", "entity"]).size()
        .reset_index(name="n")
        .sort_values(["cluster", "n", "entity"], ascending=[True, False, True])
    )
    winner = claims.drop_duplicates("cluster").set_index("cluster")["entity"]
    df["attributed_entity"] = df["cluster"].map(winner)

    outs = ledger.tx_io[ledger.tx_io["side"] == "out"]
    value_by_addr = outs.groupby("address")["value"].sum()
    df["value"] = df["address"].map(value_by_addr).fillna(0.0)
    total_value = float(df["value"].sum()) or 1.0
    labelled = df[df["attributed_entity"].notna()]
    value_attributed = float(labelled["value"].sum() / total_value)
    correct = labelled[labelled["attributed_entity"] == labelled["entity"]]
    precision = float(correct["value"].sum() / labelled["value"].sum()) if len(labelled) else 0.0

    return ClusterReport(
        n_addresses=len(df),
        n_clusters=int(df["cluster"].nunique()),
        collapse_ratio=float(1.0 - df["cluster"].nunique() / len(df)),
        purity=purity,
        adjusted_rand=float(adjusted_rand_score(truth, pred_ids)),
        adjusted_mutual_info=float(adjusted_mutual_info_score(truth, pred_ids)),
        false_merge_rate=false_merge,
        entity_recovery_rate=recovery,
        largest_cluster_share=float(sizes.max() / len(df)),
        value_attributed=value_attributed,
        attribution_precision=precision,
    )


def run_clustering_attack(ledger: Ledger, skip_coinjoins: bool = True
                          ) -> tuple[ClusterReport, dict[str, str]]:
    labels = cluster_addresses(ledger, use_change_heuristic=True,
                               skip_coinjoins=skip_coinjoins)
    return evaluate_clustering(ledger, labels), labels
