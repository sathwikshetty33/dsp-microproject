"""Re-identifying marketplace participants from published AI-layer metadata.

Two adversaries are simulated against the same published table:

* **Declared-field adversary** -- knows an organisation's region, size, request
  volume and preferred model from an external directory.  This is the adversary
  k-anonymity is designed to stop, and it does.
* **Timestamp adversary** -- additionally uses the block timestamps, which are
  *not* suppressible.  A chain cannot coarsen its own clock without breaking
  consensus, so this channel survives every metadata mitigation in the ladder.

The gap between the two is the central limitation this project set out to
evaluate: k-anonymity protects the fields a contract chooses to publish, not
the fields the ledger publishes on its behalf.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..config import Posture
from ..metrics.privacy import KAnonReport, k_anonymity_report, unicity_curve
from ..mitigations.kanon import adaptive_k_anonymity

#: fields a contract chooses to publish -- these can be generalised/suppressed
DECLARED_QIS = ["region", "org_size", "volume_band", "top_model"]
#: derived from block timestamps -- these cannot be suppressed without
#: breaking the chain's own ordering guarantees
INHERENT_QIS = ["modal_hour"]


@dataclass
class ReidReport:
    n_participants: int
    published_qis: list[str]
    dropped_qis: list[str]
    reid_declared_only: float        # unique on declared QIs alone
    reid_with_timestamps: float      # unique once block times are added
    timestamp_uplift: float          # what the unsuppressible channel adds
    k_declared: KAnonReport
    k_combined: KAnonReport
    unicity: dict[int, float]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["k_declared"] = self.k_declared.as_dict()
        d["k_combined"] = self.k_combined.as_dict()
        d["unicity"] = {str(k): v for k, v in self.unicity.items()}
        return d


def publish_profiles(profiles: pd.DataFrame, posture: Posture
                     ) -> tuple[pd.DataFrame, list[str]]:
    """Apply whatever metadata mitigation the posture mandates.

    Returns the published table and the quasi-identifiers that survived
    generalisation -- the dropped ones are fields the chain can no longer
    disclose, which the transparency metrics charge against this posture.
    """
    if profiles.empty:
        return profiles, []
    if posture.generalise_metadata:
        return adaptive_k_anonymity(profiles, DECLARED_QIS, posture.suppression_k)
    out = profiles.copy()
    out["suppressed"] = False
    return out, list(DECLARED_QIS)


def run_reidentification(profiles: pd.DataFrame, posture: Posture,
                         rng: np.random.Generator) -> ReidReport:
    published, surviving = publish_profiles(profiles, posture)
    dropped = [q for q in DECLARED_QIS if q not in surviving]
    if published.empty:
        empty = k_anonymity_report(published, [])
        return ReidReport(0, [], [], 0.0, 0.0, 0.0, empty, empty, {})

    k_declared = k_anonymity_report(published, DECLARED_QIS)
    combined_qis = DECLARED_QIS + INHERENT_QIS
    k_combined = k_anonymity_report(published, combined_qis)

    return ReidReport(
        n_participants=int(len(published)),
        published_qis=surviving,
        dropped_qis=dropped,
        reid_declared_only=k_declared.unique_fraction,
        reid_with_timestamps=k_combined.unique_fraction,
        timestamp_uplift=float(k_combined.unique_fraction - k_declared.unique_fraction),
        k_declared=k_declared,
        k_combined=k_combined,
        unicity=unicity_curve(published, combined_qis, rng),
    )
