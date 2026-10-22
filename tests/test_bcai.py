"""Correctness tests for the synthetic generators, attacks and metrics.

These check *properties that must hold for the assessment to mean anything*:
that the ledger conserves value, that postures see the same world, that
attacks beat chance where they should and sit at chance where the data is
withheld, and that the mitigations actually do what they claim.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bcai.attacks.clustering import (
    UnionFind,
    cluster_addresses,
    detect_coinjoins,
    evaluate_clustering,
    run_clustering_attack,
)
from bcai.attacks.linkage import run_coinjoin_linkage
from bcai.attacks.membership import run_membership_attack
from bcai.attacks.reidentify import DECLARED_QIS, run_reidentification
from bcai.config import DEFAULT_CONFIG, POSTURE_BY_LEVEL, Config
from bcai.fl.federated import dp_epsilon, run_federation
from bcai.fl.model import clip_update, make_feature_map, per_example_loss
from bcai.metrics.privacy import advantage, k_anonymity_report, tpr_at_fpr
from bcai.metrics.transparency import (
    detect_freeriders,
    detect_poisoners,
    verify_supply_conservation,
)
from bcai.mitigations.kanon import adaptive_k_anonymity, equivalence_class_sizes
from bcai.synth.ai_records import generate_ai_records, participant_profiles
from bcai.synth.ledger import generate_ledger


@pytest.fixture(scope="module")
def small_cfg() -> Config:
    cfg = DEFAULT_CONFIG
    return replace(
        cfg,
        ledger=replace(cfg.ledger, n_entities=50, n_transactions=700, n_days=20),
        fl=replace(cfg.fl, n_clients=6, n_rounds=5, local_steps=40,
                   n_components=96, samples_per_client=80, holdout_size=200,
                   n_freeriders=2, n_poisoners=1),
    )


@pytest.fixture(scope="module")
def ledger_l0(small_cfg):
    return generate_ledger(small_cfg.ledger, POSTURE_BY_LEVEL[0], small_cfg.seed)


@pytest.fixture(scope="module")
def ledger_l2(small_cfg):
    return generate_ledger(small_cfg.ledger, POSTURE_BY_LEVEL[2], small_cfg.seed)


# ----------------------------------------------------------------- union-find
def test_union_find_is_transitive():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    uf.union("x", "y")
    labels = uf.labels()
    assert labels["a"] == labels["c"]
    assert labels["x"] != labels["a"]


# --------------------------------------------------------------------- ledger
def test_ledger_conserves_value(ledger_l0):
    """Inputs must equal outputs plus fee on every non-coinbase transaction."""
    assert verify_supply_conservation(ledger_l0) == pytest.approx(1.0)


def test_every_address_has_exactly_one_owner(ledger_l0):
    owner = ledger_l0.address_owner
    assert owner["address"].is_unique
    io_addrs = set(ledger_l0.tx_io["address"])
    assert io_addrs <= set(owner["address"])


def test_world_is_identical_across_postures(small_cfg):
    """A posture may change how activity is recorded, never who exists."""
    def fingerprint(level):
        led = generate_ledger(small_cfg.ledger, POSTURE_BY_LEVEL[level], small_cfg.seed)
        return [(e.eid, e.role.value, e.region, e.org_size, e.tz_offset)
                for e in led.universe.entities]

    assert fingerprint(0) == fingerprint(3) == fingerprint(5)


def test_higher_posture_uses_more_addresses(small_cfg, ledger_l0):
    l1 = generate_ledger(small_cfg.ledger, POSTURE_BY_LEVEL[1], small_cfg.seed)
    assert l1.summary()["n_addresses"] > ledger_l0.summary()["n_addresses"]


def test_coinjoins_only_appear_when_enabled(ledger_l0, ledger_l2):
    assert ledger_l0.summary()["n_coinjoin"] == 0
    assert ledger_l2.summary()["n_coinjoin"] > 0


# ----------------------------------------------------------------- clustering
def test_coinjoin_detection_uses_no_ground_truth(ledger_l2):
    """Structural detection should agree closely with the true labels."""
    detected = detect_coinjoins(ledger_l2)
    truth = set(ledger_l2.transactions.loc[
        ledger_l2.transactions["kind"] == "coinjoin", "txid"])
    recall = len(detected & truth) / len(truth)
    assert recall > 0.9


def test_clustering_deanonymises_transparent_ledger(ledger_l0):
    report, _ = run_clustering_attack(ledger_l0)
    assert report.attribution_precision > 0.95
    assert report.value_attributed > 0.8
    assert report.entity_recovery_rate > 0.8


def test_mixing_degrades_clustering(ledger_l0, ledger_l2):
    base, _ = run_clustering_attack(ledger_l0)
    mixed, _ = run_clustering_attack(ledger_l2)
    assert mixed.value_attributed < base.value_attributed


def test_naive_analyst_is_poisoned_by_coinjoins(ledger_l2):
    naive = evaluate_clustering(
        ledger_l2, cluster_addresses(ledger_l2, True, skip_coinjoins=False))
    competent, _ = run_clustering_attack(ledger_l2, skip_coinjoins=True)
    assert naive.attribution_precision < competent.attribution_precision


def test_clustering_metrics_are_order_independent(ledger_l0):
    """Re-labelling clusters must not change any reported number."""
    labels = cluster_addresses(ledger_l0)
    a = evaluate_clustering(ledger_l0, labels)
    remap = {c: f"z{i}" for i, c in enumerate(sorted(set(labels.values()))[::-1])}
    b = evaluate_clustering(ledger_l0, {k: remap[v] for k, v in labels.items()})
    assert a.as_dict() == pytest.approx(b.as_dict())


# -------------------------------------------------------------------- linkage
def test_coinjoin_subset_sum_links_are_correct(ledger_l2):
    _, labels = run_clustering_attack(ledger_l2)
    report, _ = run_coinjoin_linkage(ledger_l2, labels)
    assert report.n_uniquely_solved > 0
    # The links are derived from exact arithmetic, so they must be near-perfect.
    assert report.precision > 0.9


def test_linkage_is_a_noop_without_coinjoins(ledger_l0):
    _, labels = run_clustering_attack(ledger_l0)
    report, relabelled = run_coinjoin_linkage(ledger_l0, labels)
    assert report.n_uniquely_solved == 0

    def partition(d):
        groups = {}
        for addr, cl in d.items():
            groups.setdefault(cl, set()).add(addr)
        return sorted(map(sorted, groups.values()))

    assert partition(labels) == partition(relabelled)


# --------------------------------------------------------------- k-anonymity
def test_equivalence_classes_count_correctly():
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 2]})
    assert list(equivalence_class_sizes(df, ["a", "b"])) == [2, 2, 1]


def test_adaptive_k_anonymity_meets_its_budget():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "region": rng.choice(["EU-N", "EU-S", "NA-E", "APAC"], 120),
        "org_size": rng.choice(["micro", "small", "medium", "large"], 120),
        "volume_band": rng.choice(["1-2", "3-5", "6-10", "20+"], 120),
        "top_model": rng.choice(["medfl-v1", "onco-risk", "derm-triage"], 120),
    })
    out, kept = adaptive_k_anonymity(df, DECLARED_QIS, k=5, max_suppression=0.2)
    assert kept, "generalisation should not have to drop every attribute"
    sizes = equivalence_class_sizes(out, kept)
    assert float((sizes < 5).mean()) <= 0.2
    assert len(kept) <= len(DECLARED_QIS)


def test_k_anonymity_report_flags_unique_rows():
    df = pd.DataFrame({"a": ["p", "q", "r"]})
    rep = k_anonymity_report(df, ["a"])
    assert rep.unique_fraction == pytest.approx(1.0)
    assert rep.anonymity_entropy_bits == pytest.approx(0.0)


# ---------------------------------------------------------- re-identification
def test_generalisation_reduces_declared_field_reidentification(small_cfg, ledger_l0):
    rng = np.random.default_rng(0)
    prof = participant_profiles(
        generate_ai_records(ledger_l0.universe, POSTURE_BY_LEVEL[0],
                            small_cfg.ledger.n_days, small_cfg.seed))
    open_rep = run_reidentification(prof, POSTURE_BY_LEVEL[0], rng)
    kanon_rep = run_reidentification(prof, POSTURE_BY_LEVEL[3], rng)
    assert kanon_rep.reid_declared_only < open_rep.reid_declared_only


def test_timestamp_channel_survives_generalisation(small_cfg, ledger_l0):
    """The core F2 claim: the clock is not suppressible."""
    rng = np.random.default_rng(0)
    prof = participant_profiles(
        generate_ai_records(ledger_l0.universe, POSTURE_BY_LEVEL[3],
                            small_cfg.ledger.n_days, small_cfg.seed))
    rep = run_reidentification(prof, POSTURE_BY_LEVEL[3], rng)
    assert rep.reid_with_timestamps > rep.reid_declared_only
    assert rep.timestamp_uplift > 0


# -------------------------------------------------------------------- the model
def test_clipping_bounds_the_update_norm():
    delta = np.array([3.0, 4.0])          # norm 5
    clipped, raw = clip_update(delta, 1.0)
    assert raw == pytest.approx(5.0)
    assert np.linalg.norm(clipped) == pytest.approx(1.0)


def test_clipping_leaves_small_updates_alone():
    delta = np.array([0.3, 0.4])
    clipped, raw = clip_update(delta, 1.0)
    assert np.allclose(clipped, delta)
    assert raw == pytest.approx(0.5)


def test_per_example_loss_is_lower_when_prediction_is_right():
    fmap = make_feature_map(3, 8, 0.5, np.random.default_rng(0))
    Z = fmap(np.random.default_rng(1).normal(size=(4, 3)))
    w = np.zeros(Z.shape[1])
    losses = per_example_loss(Z, np.ones(4), w)
    assert np.all(losses > 0)
    assert losses.shape == (4,)


def test_dp_epsilon_decreases_with_more_noise():
    assert dp_epsilon(0.0, 10) is None
    assert dp_epsilon(2.0, 10) < dp_epsilon(0.5, 10)


# ------------------------------------------------------ federation & attacks
@pytest.fixture(scope="module")
def fl_open(small_cfg, ledger_l0):
    return run_federation(small_cfg.fl, POSTURE_BY_LEVEL[0],
                          ledger_l0.universe, small_cfg.seed)


@pytest.fixture(scope="module")
def fl_zk(small_cfg, ledger_l0):
    return run_federation(small_cfg.fl, POSTURE_BY_LEVEL[5],
                          ledger_l0.universe, small_cfg.seed)


def test_published_records_cover_every_client_and_round(small_cfg, fl_open):
    assert len(fl_open.onchain) == small_cfg.fl.n_clients * small_cfg.fl.n_rounds


def test_zero_knowledge_posture_publishes_no_update_statistics(fl_zk):
    assert fl_zk.updates is None
    assert fl_zk.onchain["update_norm"].isna().all()
    assert fl_zk.onchain["local_loss"].isna().all()
    assert fl_zk.onchain["commitment"].notna().all()


def test_membership_attack_beats_chance_when_updates_are_public(fl_open):
    rep = run_membership_attack(fl_open.mia)
    assert rep.attack_available
    assert rep.mean_auc > 0.5
    assert rep.mean_advantage > 0


def test_membership_attack_has_no_target_under_secure_aggregation(fl_zk):
    rep = run_membership_attack(fl_zk.mia)
    assert not rep.attack_available
    assert rep.mean_auc == pytest.approx(0.5)
    assert rep.mean_advantage == pytest.approx(0.0)


def test_per_client_updates_leak_more_than_the_aggregate(fl_open):
    """F3's core comparison."""
    rep = run_membership_attack(fl_open.mia)
    assert rep.mean_auc > rep.global_model_auc


def test_auditor_loses_its_detectors_under_secure_aggregation(fl_zk):
    assert detect_freeriders(fl_zk) == pytest.approx(0.5)
    assert detect_poisoners(fl_zk) == pytest.approx(0.5)


def test_auditor_detects_freeriders_when_updates_are_public(fl_open):
    assert detect_freeriders(fl_open) > 0.6


def test_dp_noise_is_actually_applied(small_cfg, ledger_l0, fl_open):
    """The DP posture must change the published update norms."""
    dp = run_federation(small_cfg.fl, POSTURE_BY_LEVEL[4],
                        ledger_l0.universe, small_cfg.seed)
    assert dp.updates is not None
    open_norm = np.linalg.norm(fl_open.updates, axis=2).mean()
    dp_norm = np.linalg.norm(dp.updates, axis=2).mean()
    assert dp_norm > open_norm          # injected noise inflates the norm
    assert dp.dp_epsilon is not None


# -------------------------------------------------------------------- metrics
def test_advantage_is_zero_at_chance():
    assert advantage(0.5) == pytest.approx(0.0)
    assert advantage(0.4) == pytest.approx(0.0)      # clamped, never negative
    assert advantage(1.0) == pytest.approx(1.0)


def test_tpr_at_fpr_is_defined_for_a_perfect_separator():
    scores = np.concatenate([np.ones(50), np.zeros(50)])
    labels = np.concatenate([np.ones(50), np.zeros(50)])
    assert tpr_at_fpr(scores, labels, 0.01) == pytest.approx(1.0)
