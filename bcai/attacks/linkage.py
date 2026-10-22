"""Breaking CoinJoin by arithmetic, then re-running the clustering.

A joint transaction hides *which* equal-denomination output belongs to which
contributor.  It does not hide the change.  Each contributor's change satisfies

    change = (sum of that contributor's inputs) - denomination - fee_share

so an adversary can look for the subset of inputs that reproduces each change
output.  When exactly one subset matches, those inputs are provably co-owned
and provably tied to that change address -- no behavioural guesswork involved.

This is the reason real CoinJoin implementations push participants towards
uniform, change-free denominations.  The point of including it here is that it
makes the mixing result *compositional*: posture 2 looks strong against one
heuristic and much weaker once a second one is layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations

import pandas as pd

from ..synth.ledger import Ledger
from .clustering import UnionFind, detect_coinjoins

TOLERANCE = 5e-4
MAX_SUBSET = 3
MAX_INPUTS_FOR_SEARCH = 28


@dataclass
class LinkageReport:
    n_coinjoins: int
    n_attacked: int
    n_change_outputs: int
    n_uniquely_solved: int
    solve_rate: float              # change outputs pinned to exactly one input set
    inputs_relinked: int
    precision: float               # of solved links, how many were actually correct

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _solve_transaction(in_vals: list[float], out_vals: list[float],
                       fee: float) -> list[tuple[int, tuple[int, ...]]]:
    """Return (change_output_index, input_indices) for uniquely solvable change."""
    if not out_vals or len(in_vals) > MAX_INPUTS_FOR_SEARCH:
        return []
    counts = pd.Series(out_vals).round(6).value_counts()
    denom = float(counts.index[0])
    n_denom = int(counts.iloc[0])
    if n_denom < 2:
        return []
    fee_share = fee / n_denom
    change_idx = [i for i, v in enumerate(out_vals)
                  if abs(round(v, 6) - denom) > TOLERANCE]
    if not change_idx:
        return []

    # Pre-enumerate subset sums once per transaction.
    sums: dict[int, list[tuple[int, ...]]] = {}
    n = len(in_vals)
    for size in range(1, min(MAX_SUBSET, n) + 1):
        for combo in combinations(range(n), size):
            key = int(round(sum(in_vals[i] for i in combo) / TOLERANCE))
            sums.setdefault(key, []).append(combo)

    solved: list[tuple[int, tuple[int, ...]]] = []
    for oi in change_idx:
        target = out_vals[oi] + denom + fee_share
        key = int(round(target / TOLERANCE))
        matches: list[tuple[int, ...]] = []
        for k in (key - 1, key, key + 1):        # tolerate float rounding
            matches.extend(sums.get(k, []))
        uniq = {m for m in matches}
        if len(uniq) == 1:
            solved.append((oi, next(iter(uniq))))
    return solved


def run_coinjoin_linkage(ledger: Ledger, labels: dict[str, str]
                         ) -> tuple[LinkageReport, dict[str, str]]:
    """Attack every CoinJoin, then re-cluster with the recovered links."""
    joint_ids = detect_coinjoins(ledger)
    coinjoins = ledger.transactions[ledger.transactions["txid"].isin(joint_ids)]
    owner = ledger.owner_map

    # Rebuild the existing clustering inside a fresh union-find so the recovered
    # links are layered on top of it rather than replacing it.
    uf = UnionFind()
    groups: dict[str, list[str]] = {}
    for addr, cl in labels.items():
        groups.setdefault(cl, []).append(addr)
    for members in groups.values():
        first = members[0]
        uf.add(first)          # singleton clusters must survive the rebuild too
        for a in members[1:]:
            uf.union(first, a)

    io = ledger.tx_io
    io_cj = io[io["txid"].isin(set(coinjoins["txid"]))]
    ins = {t: g for t, g in io_cj[io_cj["side"] == "in"].groupby("txid")}
    outs = {t: g for t, g in io_cj[io_cj["side"] == "out"].groupby("txid")}
    fees = dict(zip(coinjoins["txid"], coinjoins["fee"]))

    n_attacked = n_change = n_solved = n_relinked = 0
    correct = 0
    for txid in coinjoins["txid"]:
        gi, go = ins.get(txid), outs.get(txid)
        if gi is None or go is None:
            continue
        in_addrs = list(gi["address"])
        in_vals = [float(v) for v in gi["value"]]
        out_addrs = list(go["address"])
        out_vals = [float(v) for v in go["value"]]
        n_attacked += 1
        denom_counts = pd.Series(out_vals).round(6).value_counts()
        n_change += sum(1 for v in out_vals
                        if abs(round(v, 6) - float(denom_counts.index[0])) > TOLERANCE)

        for oi, combo in _solve_transaction(in_vals, out_vals, float(fees[txid])):
            n_solved += 1
            change_addr = out_addrs[oi]
            truth = owner.get(change_addr)
            if all(owner.get(in_addrs[i]) == truth for i in combo):
                correct += 1
            for i in combo:
                uf.union(change_addr, in_addrs[i])
                n_relinked += 1

    report = LinkageReport(
        n_coinjoins=int(len(coinjoins)),
        n_attacked=n_attacked,
        n_change_outputs=n_change,
        n_uniquely_solved=n_solved,
        solve_rate=float(n_solved / n_change) if n_change else 0.0,
        inputs_relinked=n_relinked,
        precision=float(correct / n_solved) if n_solved else 0.0,
    )
    return report, uf.labels()
