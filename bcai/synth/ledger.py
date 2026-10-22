"""Synthetic UTXO payment ledger for the marketplace.

The ledger is deliberately *realistic in the ways that matter for privacy*:

* entities spend several of their own outputs in one transaction, which is what
  the common-input-ownership heuristic feeds on;
* payments tend to be round numbers while change does not, which is what the
  change-address heuristic feeds on;
* transactions are timestamped from each entity's local activity clock, which
  is what the behavioural-fingerprint attack feeds on;
* CoinJoin transactions pool inputs from several entities, deliberately
  poisoning the first heuristic -- this is the mitigation under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import LedgerConfig, Posture
from .entities import Entity, EntityUniverse, Role, build_universe

EPOCH = 1_767_225_600  # 2026-01-01T00:00:00Z, an arbitrary simulation start

#: who pays whom, and how often
ROLE_AFFINITY: dict[Role, list[tuple[Role, float]]] = {
    Role.PATIENT: [(Role.CLINIC, 0.45), (Role.HOSPITAL, 0.25), (Role.EXCHANGE, 0.2), (Role.MIXER, 0.1)],
    Role.CLINIC: [(Role.AGGREGATOR, 0.4), (Role.DATA_BROKER, 0.3), (Role.EXCHANGE, 0.2), (Role.HOSPITAL, 0.1)],
    Role.HOSPITAL: [(Role.AGGREGATOR, 0.45), (Role.DATA_BROKER, 0.25), (Role.EXCHANGE, 0.2), (Role.CLINIC, 0.1)],
    Role.RESEARCH_LAB: [(Role.AGGREGATOR, 0.5), (Role.DATA_BROKER, 0.3), (Role.EXCHANGE, 0.2)],
    Role.DATA_BROKER: [(Role.EXCHANGE, 0.4), (Role.PATIENT, 0.35), (Role.MIXER, 0.25)],
    Role.AGGREGATOR: [(Role.HOSPITAL, 0.4), (Role.CLINIC, 0.4), (Role.RESEARCH_LAB, 0.2)],
    Role.EXCHANGE: [(Role.PATIENT, 0.4), (Role.RESEARCH_LAB, 0.2), (Role.CLINIC, 0.2), (Role.HOSPITAL, 0.2)],
    Role.MIXER: [(Role.PATIENT, 0.4), (Role.DATA_BROKER, 0.3), (Role.CLINIC, 0.3)],
}

PURPOSE_BY_TARGET: dict[Role, str] = {
    Role.AGGREGATOR: "inference_fee",
    Role.DATA_BROKER: "dataset_purchase",
    Role.HOSPITAL: "fl_reward",
    Role.CLINIC: "fl_reward",
    Role.RESEARCH_LAB: "grant_disbursement",
    Role.EXCHANGE: "fiat_offramp",
    Role.PATIENT: "data_royalty",
    Role.MIXER: "mix_deposit",
}


@dataclass
class Ledger:
    """Flat, auditable representation of the generated chain."""

    transactions: pd.DataFrame   # txid, block, timestamp, kind, purpose, fee, n_in, n_out
    tx_io: pd.DataFrame          # txid, side, index, address, value
    address_owner: pd.DataFrame  # address, entity_id, role, region, org_size
    universe: EntityUniverse

    @property
    def owner_map(self) -> dict[str, int]:
        return dict(zip(self.address_owner["address"], self.address_owner["entity_id"]))

    def inputs_by_tx(self) -> dict[str, list[str]]:
        ins = self.tx_io[self.tx_io["side"] == "in"]
        return ins.groupby("txid")["address"].apply(list).to_dict()

    def summary(self) -> dict[str, float]:
        return {
            "n_transactions": int(len(self.transactions)),
            "n_addresses": int(self.address_owner["address"].nunique()),
            "n_entities": int(self.address_owner["entity_id"].nunique()),
            "n_coinjoin": int((self.transactions["kind"] == "coinjoin").sum()),
            "addresses_per_entity": float(
                self.address_owner.groupby("entity_id").size().mean()
            ),
            "total_value": float(
                self.tx_io.loc[self.tx_io["side"] == "out", "value"].sum()
            ),
        }


class _Builder:
    """Mutable state for one generation run."""

    def __init__(self, cfg: LedgerConfig, posture: Posture,
                 rng: np.random.Generator, world_rng: np.random.Generator):
        self.cfg = cfg
        self.posture = posture
        self.rng = rng
        # The population is drawn from a posture-independent stream, so the same
        # organisations -- same regions, sizes, clocks and habits -- face every
        # posture.  Otherwise a posture comparison would partly measure a
        # different world rather than a different mitigation.
        self.universe = build_universe(cfg.n_entities, world_rng)
        self.tx_rows: list[dict] = []
        self.io_rows: list[dict] = []
        self.counter = 0
        self._seed_utxos()

    # ------------------------------------------------------------------ setup
    def _seed_utxos(self) -> None:
        """Exchanges are the mints; everybody else is funded out of them."""
        for e in self.universe.entities:
            n_utxo = int(self.rng.integers(2, 6))
            base = self.cfg.genesis_balance * (3.0 if e.role is Role.EXCHANGE else 1.0)
            for _ in range(n_utxo):
                addr = self.universe.receiving_address(
                    e, self.rng, self.posture.fresh_address_rate
                )
                e.utxos.append((addr, float(base / n_utxo * self.rng.uniform(0.6, 1.4))))

    # ------------------------------------------------------------- primitives
    def _txid(self) -> str:
        self.counter += 1
        return f"tx{self.counter:07d}"

    def _timestamp(self, entity: Entity) -> int:
        day = int(self.rng.integers(0, self.cfg.n_days))
        hour = int(self.rng.choice(24, p=entity.activity_weights()))
        return EPOCH + day * 86_400 + hour * 3_600 + int(self.rng.integers(0, 3_600))

    def _pick_sender(self, weights: np.ndarray) -> Entity:
        idx = int(self.rng.choice(len(self.universe.entities), p=weights))
        return self.universe.entities[idx]

    def _pick_target(self, sender: Entity) -> Entity | None:
        options = ROLE_AFFINITY[sender.role]
        roles, probs = zip(*options)
        probs = np.array(probs, dtype=float)
        probs /= probs.sum()
        role = roles[int(self.rng.choice(len(roles), p=probs))]
        pool = self.universe.by_role(role)
        pool = [e for e in pool if e.eid != sender.eid]
        if not pool:
            return None
        return pool[int(self.rng.integers(0, len(pool)))]

    def _payment_value(self, sender: Entity) -> float:
        raw = float(self.rng.lognormal(sender.value_mu, sender.value_sigma))
        raw = float(np.clip(raw, 0.5, 20_000.0))
        if self.rng.random() < self.cfg.round_payment_prob:
            # Humans and billing systems like round numbers -- and that is
            # precisely what makes the *other* output identifiable as change.
            step = float(self.rng.choice([1.0, 5.0, 10.0, 25.0, 100.0]))
            raw = max(step, round(raw / step) * step)
        else:
            raw = round(raw, 4)
        return raw

    def _select_inputs(self, sender: Entity, need: float) -> list[tuple[str, float]] | None:
        """Greedy coin selection; may deliberately over-select to create
        multi-input transactions, the raw material for clustering."""
        if not sender.utxos:
            return None
        order = self.rng.permutation(len(sender.utxos))
        chosen: list[int] = []
        total = 0.0
        want_multi = self.rng.random() < self.cfg.multi_input_prob
        for i in order:
            chosen.append(int(i))
            total += sender.utxos[int(i)][1]
            enough = total >= need
            if enough and (not want_multi or len(chosen) >= min(self.cfg.max_inputs, len(order))):
                break
            if enough and want_multi and len(chosen) >= int(self.rng.integers(2, self.cfg.max_inputs + 1)):
                break
        if total < need:
            return None
        picked = [sender.utxos[i] for i in chosen]
        for i in sorted(chosen, reverse=True):
            sender.utxos.pop(i)
        return picked

    def _record(self, txid: str, block: int, ts: int, kind: str, purpose: str,
                fee: float, ins: list[tuple[str, float]], outs: list[tuple[str, float]]) -> None:
        self.tx_rows.append(
            dict(txid=txid, block=block, timestamp=ts, kind=kind, purpose=purpose,
                 fee=round(fee, 6), n_in=len(ins), n_out=len(outs))
        )
        for i, (addr, val) in enumerate(ins):
            self.io_rows.append(dict(txid=txid, side="in", index=i, address=addr, value=round(val, 6)))
        for i, (addr, val) in enumerate(outs):
            self.io_rows.append(dict(txid=txid, side="out", index=i, address=addr, value=round(val, 6)))

    def _fee(self, n_in: int) -> float:
        return round(0.01 + 0.004 * n_in, 6)

    def _topup(self, entity: Entity) -> None:
        """Refill an out-of-funds entity by an on-chain withdrawal from an
        exchange, so the ledger conserves value rather than minting it."""
        amount = float(self.cfg.genesis_balance * self.rng.uniform(0.2, 0.8))
        exchanges = [e for e in self.universe.by_role(Role.EXCHANGE) if e.eid != entity.eid]
        self.rng.shuffle(exchanges)
        for ex in exchanges:
            ins = self._select_inputs(ex, amount + 0.1)
            if ins is None:
                continue
            fee = self._fee(len(ins))
            addr = self.universe.receiving_address(
                entity, self.rng, self.posture.fresh_address_rate
            )
            outs = [(addr, round(amount, 6))]
            entity.utxos.append((addr, round(amount, 6)))
            change = sum(v for _, v in ins) - amount - fee
            if change > 1e-6:
                caddr = self.universe.receiving_address(
                    ex, self.rng, self.posture.fresh_address_rate
                )
                outs.append((caddr, round(change, 6)))
                ex.utxos.append((caddr, round(change, 6)))
            ts = self._timestamp(ex)
            self.rng.shuffle(outs)
            self._record(self._txid(), (ts - EPOCH) // self.cfg.seconds_per_block, ts,
                         "payment", "exchange_withdrawal", fee, ins, outs)
            return
        # Every exchange is dry: mint explicitly and label it, so an auditor can
        # still account for total supply.
        addr = self.universe.receiving_address(entity, self.rng, self.posture.fresh_address_rate)
        entity.utxos.append((addr, round(amount, 6)))
        ts = self._timestamp(entity)
        self._record(self._txid(), (ts - EPOCH) // self.cfg.seconds_per_block, ts,
                     "coinbase", "issuance", 0.0, [], [(addr, round(amount, 6))])

    # ---------------------------------------------------------------- payment
    def _simple_payment(self, sender: Entity) -> bool:
        target = self._pick_target(sender)
        if target is None:
            return False
        value = self._payment_value(sender)
        fee_guess = self._fee(2)
        ins = self._select_inputs(sender, value + fee_guess)
        if ins is None:
            self._topup(sender)
            ins = self._select_inputs(sender, value + fee_guess)
            if ins is None:
                return False

        fee = self._fee(len(ins))
        total_in = sum(v for _, v in ins)
        change = total_in - value - fee
        ts = self._timestamp(sender)
        txid = self._txid()

        pay_addr = self.universe.receiving_address(target, self.rng, self.posture.fresh_address_rate)
        outs = [(pay_addr, value)]
        if change > 1e-6:
            change_addr = self.universe.receiving_address(
                sender, self.rng, self.posture.fresh_address_rate
            )
            outs.append((change_addr, round(change, 6)))
            sender.utxos.append((change_addr, round(change, 6)))
        target.utxos.append((pay_addr, value))

        # Shuffle so the change output is not always in a fixed position.
        self.rng.shuffle(outs)
        self._record(txid, (ts - EPOCH) // self.cfg.seconds_per_block, ts, "payment",
                     PURPOSE_BY_TARGET[target.role], fee, ins, outs)
        return True

    # --------------------------------------------------------------- coinjoin
    def _coinjoin(self, initiator: Entity) -> bool:
        """Equal-output joint transaction: several entities contribute inputs,
        so common-input-ownership produces a *false* merge."""
        k = self.posture.coinjoin_anonymity_set
        pool = [initiator]
        candidates = [e for e in self.universe.entities
                      if e.eid != initiator.eid and len(e.utxos) >= 1]
        if len(candidates) < k - 1:
            return False
        picks = self.rng.choice(len(candidates), size=k - 1, replace=False)
        pool.extend(candidates[int(i)] for i in picks)

        denom = float(self.rng.choice([5.0, 10.0, 25.0, 50.0]))
        ins: list[tuple[str, float]] = []
        contributors: list[Entity] = []
        contributed: list[float] = []
        for e in pool:
            picked = self._select_inputs(e, denom + 0.05)
            if picked is None:
                self._topup(e)
                picked = self._select_inputs(e, denom + 0.05)
            if picked is None:
                continue
            ins.extend(picked)
            contributors.append(e)
            contributed.append(sum(v for _, v in picked))
        if len(contributors) < 2:
            return False

        fee = self._fee(len(ins))
        outs: list[tuple[str, float]] = []
        for e in contributors:
            # Uniform-denomination output: indistinguishable across participants.
            addr = self.universe.new_address(e)
            outs.append((addr, denom))
            e.utxos.append((addr, denom))
        fee_share = fee / len(contributors)
        for e, spent in zip(contributors, contributed):
            change = spent - denom - fee_share
            if change > 1e-6:
                addr = self.universe.new_address(e)
                outs.append((addr, round(change, 6)))
                e.utxos.append((addr, round(change, 6)))

        ts = self._timestamp(initiator)
        self.rng.shuffle(outs)
        self._record(self._txid(), (ts - EPOCH) // self.cfg.seconds_per_block, ts,
                     "coinjoin", "mix", fee, ins, outs)
        return True

    # ------------------------------------------------------------------- main
    def run(self) -> Ledger:
        rates = np.array([e.rate for e in self.universe.entities], dtype=float)
        weights = rates / rates.sum()

        attempts = 0
        max_attempts = self.cfg.n_transactions * 4
        while len(self.tx_rows) < self.cfg.n_transactions and attempts < max_attempts:
            attempts += 1
            sender = self._pick_sender(weights)
            if self.rng.random() < self.posture.coinjoin_rate:
                if not self._coinjoin(sender):
                    self._simple_payment(sender)
            else:
                self._simple_payment(sender)

        transactions = pd.DataFrame(self.tx_rows).sort_values("timestamp").reset_index(drop=True)
        tx_io = pd.DataFrame(self.io_rows)
        owner = pd.DataFrame(
            [
                dict(address=a, entity_id=eid,
                     role=self.universe.by_id[eid].role.value,
                     region=self.universe.by_id[eid].region,
                     org_size=self.universe.by_id[eid].org_size)
                for a, eid in self.universe.address_owner.items()
            ]
        )
        return Ledger(transactions, tx_io, owner, self.universe)


def generate_ledger(cfg: LedgerConfig, posture: Posture, seed: int) -> Ledger:
    """Build one synthetic ledger under a given privacy posture.

    ``seed`` fixes the world (who exists, how they behave); the posture only
    changes how that world's activity is recorded on chain.
    """
    world_rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + posture.level * 9_973)
    return _Builder(cfg, posture, rng, world_rng).run()
