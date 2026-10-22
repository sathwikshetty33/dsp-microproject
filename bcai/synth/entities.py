"""Synthetic participants in a decentralised medical-AI marketplace.

Nothing here is derived from real chain data.  Each entity gets a behavioural
profile (activity clock, payment sizes, address-reuse habits) that later shows
up as a *fingerprint* the de-anonymisation attacks can exploit.  That is the
point: the behaviour, not the identity, is what leaks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Role(str, Enum):
    EXCHANGE = "exchange"
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    RESEARCH_LAB = "research_lab"
    DATA_BROKER = "data_broker"
    AGGREGATOR = "aggregator"
    PATIENT = "patient"
    MIXER = "mixer"


#: role -> (population share, payments/day, log-normal value mu/sigma, timezone offset spread)
ROLE_PROFILE: dict[Role, dict[str, float]] = {
    Role.EXCHANGE: dict(share=0.03, rate=14.0, mu=3.4, sigma=1.1, work_start=0, work_len=24),
    Role.HOSPITAL: dict(share=0.10, rate=2.2, mu=2.9, sigma=0.7, work_start=7, work_len=12),
    Role.CLINIC: dict(share=0.14, rate=1.4, mu=2.2, sigma=0.6, work_start=8, work_len=10),
    Role.RESEARCH_LAB: dict(share=0.16, rate=3.1, mu=2.5, sigma=0.9, work_start=9, work_len=11),
    Role.DATA_BROKER: dict(share=0.08, rate=4.0, mu=3.1, sigma=1.0, work_start=6, work_len=14),
    Role.AGGREGATOR: dict(share=0.05, rate=5.5, mu=1.8, sigma=0.5, work_start=0, work_len=24),
    Role.PATIENT: dict(share=0.42, rate=0.35, mu=1.2, sigma=0.8, work_start=8, work_len=14),
    Role.MIXER: dict(share=0.02, rate=6.0, mu=2.6, sigma=0.9, work_start=0, work_len=24),
}

REGIONS = ("EU-N", "EU-S", "NA-E", "NA-W", "APAC", "LATAM")
ORG_SIZE_BUCKETS = ("micro", "small", "medium", "large")


@dataclass
class Entity:
    """A real-world organisation or person controlling one or more addresses."""

    eid: int
    role: Role
    region: str
    org_size: str
    tz_offset: int
    rate: float                 # expected payments per day
    value_mu: float             # log-normal location of payment size
    value_sigma: float
    work_start: int
    work_len: int
    reuse_bias: float           # 0 = perfect hygiene, 1 = always reuse one address
    addresses: list[str] = field(default_factory=list)
    utxos: list[tuple[str, float]] = field(default_factory=list)

    def activity_weights(self) -> np.ndarray:
        """24-bin probability of transacting in each local hour."""
        hours = np.arange(24)
        local = (hours + self.tz_offset) % 24
        inside = ((local - self.work_start) % 24) < self.work_len
        w = np.where(inside, 1.0, 0.08)
        # A smooth mid-shift peak makes profiles distinguishable but not trivial.
        peak = (self.work_start + self.work_len / 2) % 24
        w *= 1.0 + 0.8 * np.exp(-0.5 * (((local - peak + 12) % 24 - 12) / 3.0) ** 2)
        return w / w.sum()


def _address(eid: int, index: int, salt: str = "bcai") -> str:
    """Deterministic pseudonymous address string (a stand-in for a pubkey hash)."""
    digest = hashlib.sha256(f"{salt}:{eid}:{index}".encode()).hexdigest()
    return "0x" + digest[:40]


class EntityUniverse:
    """The full population, plus the ground-truth address -> entity mapping."""

    def __init__(self, entities: list[Entity]):
        self.entities = entities
        self.by_id = {e.eid: e for e in entities}
        self.address_owner: dict[str, int] = {}

    def new_address(self, entity: Entity) -> str:
        addr = _address(entity.eid, len(entity.addresses))
        entity.addresses.append(addr)
        self.address_owner[addr] = entity.eid
        return addr

    def receiving_address(self, entity: Entity, rng: np.random.Generator,
                          fresh_rate: float) -> str:
        """Pick a receiving address, honouring both posture and personal habit."""
        effective_fresh = fresh_rate * (1.0 - entity.reuse_bias)
        if entity.addresses and rng.random() > effective_fresh:
            # Reuse: heavier entities concentrate on a small set of hot addresses.
            hot = min(len(entity.addresses), 3)
            return entity.addresses[int(rng.integers(0, hot))]
        return self.new_address(entity)

    def by_role(self, role: Role) -> list[Entity]:
        return [e for e in self.entities if e.role is role]

    def __len__(self) -> int:
        return len(self.entities)


def build_universe(n_entities: int, rng: np.random.Generator) -> EntityUniverse:
    """Sample a population of entities with role-consistent behaviour."""
    roles = list(ROLE_PROFILE)
    shares = np.array([ROLE_PROFILE[r]["share"] for r in roles], dtype=float)
    shares /= shares.sum()
    counts = np.maximum(1, np.round(shares * n_entities).astype(int))

    entities: list[Entity] = []
    eid = 0
    for role, count in zip(roles, counts):
        p = ROLE_PROFILE[role]
        for _ in range(count):
            # Privacy hygiene is a personal trait; services reuse far more.
            if role in (Role.EXCHANGE, Role.AGGREGATOR, Role.MIXER):
                reuse_bias = float(rng.uniform(0.75, 0.95))
            elif role is Role.PATIENT:
                reuse_bias = float(rng.beta(2.0, 3.0))
            else:
                reuse_bias = float(rng.beta(2.5, 2.5))
            entities.append(
                Entity(
                    eid=eid,
                    role=role,
                    region=str(rng.choice(REGIONS)),
                    org_size=str(rng.choice(ORG_SIZE_BUCKETS, p=[0.35, 0.3, 0.22, 0.13])),
                    tz_offset=int(rng.integers(-8, 10)),
                    rate=float(p["rate"] * rng.lognormal(0.0, 0.35)),
                    value_mu=float(p["mu"] + rng.normal(0, 0.2)),
                    value_sigma=float(max(0.2, p["sigma"] + rng.normal(0, 0.1))),
                    work_start=int(p["work_start"]),
                    work_len=int(p["work_len"]),
                    reuse_bias=reuse_bias,
                )
            )
            eid += 1

    universe = EntityUniverse(entities)
    for e in entities:
        universe.new_address(e)  # every entity starts with one address
    return universe
