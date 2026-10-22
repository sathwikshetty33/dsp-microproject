"""On-chain records emitted by the marketplace's AI contracts.

Besides the federated-learning updates, a deployed system publishes inference
requests and dataset listings.  These carry no payload -- inputs are committed
to, not revealed -- and are therefore often described as privacy-preserving.
They still carry a requester address, a timestamp, a model id, a fee and a
coarse request tag, which is more than enough quasi-identifying structure to
re-identify the requester.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..config import Posture
from ..synth.entities import EntityUniverse, Role
from ..synth.ledger import EPOCH

MODELS = ("medfl-v1", "medfl-v2", "onco-risk", "sepsis-early", "derm-triage")
REQUEST_TAGS = ("screening", "triage", "second-opinion", "research-cohort", "audit")


def generate_ai_records(universe: EntityUniverse, posture: Posture, n_days: int,
                        seed: int) -> pd.DataFrame:
    """One row per inference request or dataset listing."""
    world_rng = np.random.default_rng(seed + 3_000)
    mech_rng = np.random.default_rng(seed + 9_000 + posture.level * 17)

    consumers = (universe.by_role(Role.RESEARCH_LAB) + universe.by_role(Role.CLINIC)
                 + universe.by_role(Role.HOSPITAL) + universe.by_role(Role.DATA_BROKER))
    rows: list[dict] = []
    for e in consumers:
        n_req = int(world_rng.poisson(max(1.0, e.rate * 2.5)))
        prefs = world_rng.dirichlet(np.ones(len(MODELS)) * 0.8)
        for _ in range(n_req):
            hour = int(world_rng.choice(24, p=e.activity_weights()))
            day = int(world_rng.integers(0, n_days))
            ts = EPOCH + day * 86_400 + hour * 3_600 + int(mech_rng.integers(0, 3_600))
            model = MODELS[int(world_rng.choice(len(MODELS), p=prefs))]
            payload = f"{e.eid}:{ts}:{model}".encode()
            rows.append(
                dict(
                    record_id="ai" + hashlib.sha256(payload).hexdigest()[:14],
                    requester_address=(
                        e.addresses[0] if world_rng.random() > posture.fresh_address_rate
                        else universe.new_address(e)
                    ),
                    entity_id=e.eid,
                    role=e.role.value,
                    region=e.region,
                    org_size=e.org_size,
                    model_id=model,
                    request_tag=str(world_rng.choice(REQUEST_TAGS)),
                    active_hour=hour,
                    timestamp=ts,
                    fee=float(np.round(world_rng.lognormal(e.value_mu - 1.0, 0.4), 4)),
                    input_commitment="0x" + hashlib.sha256(payload + b"c").hexdigest()[:32],
                )
            )
    return pd.DataFrame(rows)


def participant_profiles(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse the record stream into one quasi-identifier profile per entity.

    This is exactly what an adversary builds before matching against an
    external directory (a hospital registry, a grant database, LinkedIn).
    """
    if records.empty:
        return pd.DataFrame()
    grp = records.groupby("entity_id")
    profiles = pd.DataFrame({
        "role": grp["role"].first(),
        "region": grp["region"].first(),
        "org_size": grp["org_size"].first(),
        "n_requests": grp.size(),
        "modal_hour": grp["active_hour"].agg(lambda s: int(s.mode().iloc[0])),
        "top_model": grp["model_id"].agg(lambda s: s.mode().iloc[0]),
        "median_fee": grp["fee"].median(),
    }).reset_index()
    profiles["activity_band"] = (profiles["modal_hour"] // 4) * 4
    profiles["volume_band"] = pd.cut(
        profiles["n_requests"], bins=[-1, 2, 5, 10, 20, 10_000],
        labels=["1-2", "3-5", "6-10", "11-20", "20+"]
    ).astype(str)
    return profiles
