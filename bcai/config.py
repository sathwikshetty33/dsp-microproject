"""Experiment configuration and the privacy-posture ladder.

The whole study is organised around a single ordered dial, ``PostureLevel``.
Level 0 is a fully transparent public ledger; each subsequent level adds one
privacy-enhancing technology (PET).  Every attack and every audit metric is
recomputed at every level, which is what produces the trade-off curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"


@dataclass(frozen=True)
class Posture:
    """One rung of the privacy ladder.

    Attributes
    ----------
    level:
        Ordinal index, 0 (fully transparent) .. 5 (zero-knowledge).
    name / blurb:
        Human labels used in the report and figures.
    fresh_address_rate:
        Probability an entity generates a new receiving address per payment.
        Drives how well common-input-ownership clustering works.
    coinjoin_rate:
        Fraction of payments routed through a CoinJoin-style joint transaction,
        which deliberately breaks the common-input-ownership assumption.
    coinjoin_anonymity_set:
        Number of distinct entities per joint transaction.
    generalise_metadata:
        Apply k-anonymity generalisation (bucketing/suppression) to the
        quasi-identifiers published with each federated-learning update.
    suppression_k:
        Target k for generalisation; groups smaller than k are suppressed.
    dp_noise_multiplier:
        Gaussian noise multiplier applied to clipped client updates (DP-SGD).
    dp_count_epsilon:
        Epsilon of the Laplace mechanism protecting published sample counts.
        ``None`` means counts are published exactly.
    hide_update_metadata:
        Publish only a commitment plus a validity proof; per-client norms and
        losses never hit the chain.  Models the ZK / trusted-execution case.
    """

    level: int
    name: str
    blurb: str
    fresh_address_rate: float
    coinjoin_rate: float
    coinjoin_anonymity_set: int
    generalise_metadata: bool
    suppression_k: int
    dp_noise_multiplier: float
    dp_count_epsilon: float | None
    hide_update_metadata: bool

    @property
    def short(self) -> str:
        return f"L{self.level}"


POSTURES: tuple[Posture, ...] = (
    Posture(
        level=0,
        name="Transparent baseline",
        blurb="Plain public ledger. Addresses reused, all FL metadata in the clear.",
        fresh_address_rate=0.05,
        coinjoin_rate=0.0,
        coinjoin_anonymity_set=1,
        generalise_metadata=False,
        suppression_k=1,
        dp_noise_multiplier=0.0,
        dp_count_epsilon=None,
        hide_update_metadata=False,
    ),
    Posture(
        level=1,
        name="Address hygiene",
        blurb="One-time addresses and change rotation; metadata still public.",
        fresh_address_rate=0.85,
        coinjoin_rate=0.0,
        coinjoin_anonymity_set=1,
        generalise_metadata=False,
        suppression_k=1,
        dp_noise_multiplier=0.0,
        dp_count_epsilon=None,
        hide_update_metadata=False,
    ),
    Posture(
        level=2,
        name="Hygiene + mixing",
        blurb="Adds CoinJoin-style joint transactions that break input ownership.",
        fresh_address_rate=0.85,
        coinjoin_rate=0.55,
        coinjoin_anonymity_set=6,
        generalise_metadata=False,
        suppression_k=1,
        dp_noise_multiplier=0.0,
        dp_count_epsilon=None,
        hide_update_metadata=False,
    ),
    Posture(
        level=3,
        name="+ k-anonymous metadata",
        blurb="FL quasi-identifiers generalised and rare groups suppressed (k=5).",
        fresh_address_rate=0.85,
        coinjoin_rate=0.55,
        coinjoin_anonymity_set=6,
        generalise_metadata=True,
        suppression_k=5,
        dp_noise_multiplier=0.0,
        dp_count_epsilon=None,
        hide_update_metadata=False,
    ),
    Posture(
        level=4,
        name="+ differential privacy",
        blurb="Clipped client updates with Gaussian noise; counts under Laplace.",
        fresh_address_rate=0.85,
        coinjoin_rate=0.55,
        coinjoin_anonymity_set=6,
        generalise_metadata=True,
        suppression_k=5,
        dp_noise_multiplier=1.1,
        dp_count_epsilon=0.5,
        hide_update_metadata=False,
    ),
    Posture(
        level=5,
        name="+ zero-knowledge",
        blurb="Only commitments and validity proofs on chain; no per-client stats.",
        fresh_address_rate=0.95,
        coinjoin_rate=0.75,
        coinjoin_anonymity_set=10,
        generalise_metadata=True,
        suppression_k=5,
        dp_noise_multiplier=1.1,
        dp_count_epsilon=0.5,
        hide_update_metadata=True,
    ),
)

POSTURE_BY_LEVEL = {p.level: p for p in POSTURES}


@dataclass(frozen=True)
class LedgerConfig:
    """Size and shape of the synthetic payment layer."""

    n_entities: int = 180
    n_transactions: int = 6000
    n_days: int = 90
    seconds_per_block: int = 12
    genesis_balance: float = 5_000.0
    multi_input_prob: float = 0.45
    max_inputs: int = 5
    round_payment_prob: float = 0.55


@dataclass(frozen=True)
class FLConfig:
    """Size and shape of the on-chain federated-learning workload."""

    n_clients: int = 16
    samples_per_client: int = 120
    n_features: int = 14
    n_components: int = 768      # random Fourier features -> capacity to memorise
    rff_gamma: float = 0.12
    signal_strength: float = 0.45  # low on purpose: Bayes accuracy ~0.72
    l2: float = 0.0
    n_rounds: int = 30
    local_steps: int = 400
    learning_rate: float = 2.5
    clip_norm: float = 20.0
    dp_count_sensitivity: float = 25.0   # protect a cohort of 25 records, not one
    n_freeriders: int = 3
    n_poisoners: int = 2
    poison_flip_rate: float = 0.45
    holdout_size: int = 2_000


@dataclass(frozen=True)
class Config:
    seed: int = 20_260_926
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    postures: tuple[Posture, ...] = POSTURES

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["postures"] = [asdict(p) for p in self.postures]
        return d

    def ensure_dirs(self) -> None:
        for path in (OUTPUT_DIR, DATA_DIR, FIGURE_DIR, REPORT_DIR):
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config()
