"""End-to-end experiment: run every attack and every audit at every posture."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .attacks.clustering import (
    ClusterReport,
    cluster_addresses,
    evaluate_clustering,
    run_clustering_attack,
)
from .attacks.linkage import LinkageReport, run_coinjoin_linkage
from .attacks.membership import MembershipReport, run_membership_attack
from .attacks.reidentify import DECLARED_QIS, ReidReport, run_reidentification
from .config import DATA_DIR, Config, Posture
from .fl.federated import FLResult, run_federation
from .metrics.transparency import TransparencyReport, build_transparency_report
from .synth.ai_records import generate_ai_records, participant_profiles
from .synth.ledger import Ledger, generate_ledger


@dataclass
class PostureOutcome:
    """Everything measured at one rung of the ladder."""

    posture: Posture
    naive_cluster: ClusterReport
    competent_cluster: ClusterReport
    advanced_cluster: ClusterReport
    linkage: LinkageReport
    reid: ReidReport
    membership: MembershipReport
    transparency: TransparencyReport
    fl_accuracy: float
    fl_clean_reference: float
    dp_epsilon: float | None
    ledger_summary: dict
    runtime_seconds: float

    # ---- the two headline indices ---------------------------------------
    @property
    def payment_deanonymisation(self) -> float:
        """Share of transaction volume an adversary attributes *correctly*."""
        c = self.advanced_cluster
        return float(c.value_attributed * c.attribution_precision)

    @property
    def privacy_exposure_index(self) -> float:
        """Mean of the three attack surfaces, each in [0, 1]."""
        return float(np.mean([
            self.payment_deanonymisation,
            self.reid.reid_with_timestamps,
            self.membership.mean_advantage,
        ]))

    @property
    def audit_capability_index(self) -> float:
        return self.transparency.audit_capability_index

    @property
    def utility_retention(self) -> float:
        """Model accuracy relative to an honest, noise-free federation."""
        if self.fl_clean_reference <= 0:
            return 0.0
        return float(self.fl_accuracy / self.fl_clean_reference)

    def to_row(self) -> dict:
        return {
            "level": self.posture.level,
            "posture": self.posture.name,
            "blurb": self.posture.blurb,
            # privacy side
            "payment_deanonymisation": self.payment_deanonymisation,
            "naive_analyst_precision": self.naive_cluster.attribution_precision,
            "competent_analyst_value": self.competent_cluster.value_attributed,
            "advanced_analyst_value": self.advanced_cluster.value_attributed,
            "entity_recovery_rate": self.advanced_cluster.entity_recovery_rate,
            "coinjoin_solve_rate": self.linkage.solve_rate,
            "coinjoin_link_precision": self.linkage.precision,
            "reid_declared_only": self.reid.reid_declared_only,
            "reid_with_timestamps": self.reid.reid_with_timestamps,
            "timestamp_uplift": self.reid.timestamp_uplift,
            "median_k": self.reid.k_combined.median_k,
            "anonymity_entropy_bits": self.reid.k_combined.anonymity_entropy_bits,
            "mia_available": self.membership.attack_available,
            "mia_auc": self.membership.mean_auc,
            "mia_advantage": self.membership.mean_advantage,
            "mia_tpr_at_1pct": self.membership.tpr_at_1pct_fpr,
            "mia_global_auc": self.membership.global_model_auc,
            "privacy_exposure_index": self.privacy_exposure_index,
            # transparency side
            "freerider_detection_auc": self.transparency.freerider_detection_auc,
            "poisoning_detection_auc": self.transparency.poisoning_detection_auc,
            "contribution_fidelity": self.transparency.contribution_fidelity,
            "payment_traceability": self.transparency.payment_traceability,
            "supply_conservation": self.transparency.supply_conservation,
            "metadata_completeness": self.transparency.metadata_completeness,
            "audit_capability_index": self.audit_capability_index,
            # utility side
            "fl_accuracy": self.fl_accuracy,
            "fl_clean_reference": self.fl_clean_reference,
            "utility_retention": self.utility_retention,
            "dp_epsilon": self.dp_epsilon,
            "runtime_seconds": self.runtime_seconds,
        }


def _metadata_completeness(reid: ReidReport, posture: Posture) -> float:
    """How much of the intended metadata the chain still actually publishes."""
    if reid.n_participants == 0:
        declared = 0.0
    else:
        kept = len(reid.published_qis) / len(DECLARED_QIS)
        declared = kept * (1.0 - reid.k_declared.suppression_rate)
    update_fields = 0.0 if posture.hide_update_metadata else 1.0
    return float(np.mean([declared, update_fields]))


def run_posture(cfg: Config, posture: Posture, verbose: bool = True,
                seed: int | None = None, persist: bool = True) -> PostureOutcome:
    seed = cfg.seed if seed is None else seed
    t0 = time.time()
    if verbose:
        print(f"  [L{posture.level}] {posture.name}: ledger ...", flush=True)
    ledger: Ledger = generate_ledger(cfg.ledger, posture, seed)

    if verbose:
        print(f"  [L{posture.level}] clustering + linkage ...", flush=True)
    naive = evaluate_clustering(
        ledger, cluster_addresses(ledger, True, skip_coinjoins=False)
    )
    competent, labels = run_clustering_attack(ledger, skip_coinjoins=True)
    linkage, relinked = run_coinjoin_linkage(ledger, labels)
    advanced = evaluate_clustering(ledger, relinked)

    if verbose:
        print(f"  [L{posture.level}] AI records + re-identification ...", flush=True)
    records = generate_ai_records(ledger.universe, posture, cfg.ledger.n_days, seed)
    profiles = participant_profiles(records)
    rng = np.random.default_rng(seed + 500 + posture.level)
    reid = run_reidentification(profiles, posture, rng)

    if verbose:
        print(f"  [L{posture.level}] federated training ...", flush=True)
    fl: FLResult = run_federation(cfg.fl, posture, ledger.universe, seed)
    membership = run_membership_attack(fl.mia)

    transparency = build_transparency_report(
        ledger, fl, advanced, _metadata_completeness(reid, posture)
    )

    # Persist the raw synthetic records so the analysis is reproducible.
    if persist:
        _persist(ledger, records, fl, posture)

    return PostureOutcome(
        posture=posture,
        naive_cluster=naive,
        competent_cluster=competent,
        advanced_cluster=advanced,
        linkage=linkage,
        reid=reid,
        membership=membership,
        transparency=transparency,
        fl_accuracy=fl.final_accuracy,
        fl_clean_reference=fl.clean_accuracy_reference,
        dp_epsilon=fl.dp_epsilon,
        ledger_summary=ledger.summary(),
        runtime_seconds=time.time() - t0,
    )


def _persist(ledger: Ledger, records: pd.DataFrame, fl: FLResult,
             posture: Posture) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"L{posture.level}"
    ledger.transactions.to_csv(DATA_DIR / f"{tag}_transactions.csv", index=False)
    ledger.tx_io.to_csv(DATA_DIR / f"{tag}_tx_io.csv", index=False)
    ledger.address_owner.to_csv(DATA_DIR / f"{tag}_address_owner.csv", index=False)
    records.to_csv(DATA_DIR / f"{tag}_ai_records.csv", index=False)
    fl.onchain.to_csv(DATA_DIR / f"{tag}_fl_onchain.csv", index=False)
    fl.client_truth.to_csv(DATA_DIR / f"{tag}_fl_client_truth.csv", index=False)
    fl.history.to_csv(DATA_DIR / f"{tag}_fl_history.csv", index=False)


#: columns that are never averaged across repeated trials
_NUMERIC_EXCLUDE = {"level", "posture", "blurb", "mia_available", "trial", "seed"}


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-trial rows into one row per posture, with mean and std."""
    numeric = [c for c in raw.columns
               if c not in _NUMERIC_EXCLUDE and pd.api.types.is_numeric_dtype(raw[c])]
    grouped = raw.groupby("level")
    results = grouped[numeric].mean().reset_index()
    stds = grouped[numeric].std().reset_index()
    for c in numeric:
        results[f"{c}_std"] = stds[c]
    meta = raw.drop_duplicates("level").set_index("level")
    for col in ("posture", "blurb", "mia_available"):
        results[col] = results["level"].map(meta[col])
    results["n_trials"] = int(raw["trial"].nunique()) if "trial" in raw else 1
    return results


def run_experiment(cfg: Config, verbose: bool = True, n_repeats: int = 5
                   ) -> tuple[pd.DataFrame, list[PostureOutcome]]:
    """Run every posture ``n_repeats`` times on independent worlds.

    Several metrics -- poisoner-detection AUC over two poisoners, Spearman
    fidelity over sixteen clients -- are coarse on a single draw.  Repeating
    the whole experiment on fresh worlds and reporting mean and standard
    deviation is what makes the posture ordering trustworthy rather than an
    artefact of one seed.
    """
    cfg.ensure_dirs()
    rows: list[dict] = []
    representative: list[PostureOutcome] = []

    for rep in range(n_repeats):
        seed = cfg.seed + rep * 104_729
        if verbose:
            print(f"\n--- trial {rep + 1}/{n_repeats} (seed {seed}) ---", flush=True)
        for posture in cfg.postures:
            outcome = run_posture(cfg, posture, verbose=verbose, seed=seed,
                                  persist=(rep == 0))
            row = outcome.to_row()
            row["trial"] = rep
            row["seed"] = seed
            rows.append(row)
            if rep == 0:
                representative.append(outcome)

    raw = pd.DataFrame(rows)
    raw.to_csv(DATA_DIR / "results_raw.csv", index=False)

    results = aggregate_results(raw)
    results.to_csv(DATA_DIR / "results.csv", index=False)

    detail = {
        f"L{o.posture.level}": {
            "posture": asdict(o.posture),
            "naive_cluster": o.naive_cluster.as_dict(),
            "competent_cluster": o.competent_cluster.as_dict(),
            "advanced_cluster": o.advanced_cluster.as_dict(),
            "linkage": o.linkage.as_dict(),
            "reidentification": o.reid.as_dict(),
            "membership": o.membership.as_dict(),
            "transparency": o.transparency.as_dict(),
            "ledger": o.ledger_summary,
        }
        for o in representative
    }
    (DATA_DIR / "detail.json").write_text(json.dumps(detail, indent=2, default=str))
    return results, representative
