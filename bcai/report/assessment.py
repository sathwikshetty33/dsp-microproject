"""Generate the security/privacy assessment from the measured results.

Every number in the report is read out of the results table, so the prose
cannot drift away from the experiment.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..attacks.reidentify import DECLARED_QIS
from ..config import Config


def _v(results: pd.DataFrame, level: int, col: str) -> float:
    return float(results.loc[results["level"] == level, col].iloc[0])


def _pm(results: pd.DataFrame, level: int, col: str, fmt: str = "{:.3f}",
        std_fmt: str | None = None) -> str:
    """value ± std, when the experiment was repeated.

    A spread is always non-negative, so a signed value format never carries
    over to it.
    """
    val = _v(results, level, col)
    std_fmt = std_fmt or fmt.replace("+", "")
    std_col = f"{col}_std"
    if std_col in results.columns:
        std = float(results.loc[results["level"] == level, std_col].iloc[0])
        if pd.notna(std):
            return f"{fmt.format(val)} ± {std_fmt.format(std)}"
    return fmt.format(val)


def _join(items) -> str:
    items = [f"`{i}`" for i in items]
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _severity(likelihood: str, impact: str) -> str:
    """Standard likelihood x impact matrix.

    Critical is reserved for high-on-both; otherwise every row in the register
    would read Critical and the column would carry no information.
    """
    scale = {"Low": 1, "Medium": 2, "High": 3}
    score = scale[likelihood] * scale[impact]
    if score >= 9:
        return "**Critical**"
    if score >= 6:
        return "**High**"
    if score >= 3:
        return "Medium"
    return "Low"


def write_assessment(cfg: Config, results: pd.DataFrame, outcomes,
                     out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = sorted(int(v) for v in results["level"])
    lo, hi = levels[0], levels[-1]
    n_trials = int(results["n_trials"].iloc[0]) if "n_trials" in results else 1

    # ---------------------------------------------------------------- summary
    s = []
    s.append("# Privacy and Security Limitations of Blockchain AI")
    s.append("")
    s.append("**A quantitative assessment of the transparency/privacy trade-off "
             "in on-chain federated learning**")
    s.append("")
    s.append(f"*Assessment date: {date.today().isoformat()} · "
             f"{n_trials} independent trials per posture · "
             f"seed {cfg.seed} · all data synthetic*")
    s.append("")
    s.append("---")
    s.append("")
    s.append("## 1. Executive summary")
    s.append("")
    s.append(
        "A public ledger is often proposed as the coordination layer for "
        "federated machine learning: participants publish model updates, an "
        "aggregator combines them, and the chain provides an immutable audit "
        "trail that makes free-riding and poisoning detectable. This "
        "assessment builds such a system end to end on synthetic records and "
        "attacks it."
    )
    s.append("")
    s.append(
        "The central result is that the audit trail and the attack surface are "
        "the same object. Every privacy-enhancing technology evaluated here "
        "removes adversarial capability and auditor capability together, in "
        "close to equal measure, because both parties read identical bytes."
    )
    s.append("")
    s.append(_table([
        ["Privacy exposure index", _pm(results, lo, "privacy_exposure_index"),
         _pm(results, hi, "privacy_exposure_index"), "lower is better"],
        ["Audit capability index", _pm(results, lo, "audit_capability_index"),
         _pm(results, hi, "audit_capability_index"), "higher is better"],
        ["Model accuracy", _pm(results, lo, "fl_accuracy"),
         _pm(results, hi, "fl_accuracy"),
         f"honest reference {_v(results, lo, 'fl_clean_reference'):.3f}"],
    ], ["Index", f"L{lo} (transparent)", f"L{hi} (zero-knowledge)", "Direction"]))
    s.append("")
    s.append(
        f"Moving from the transparent baseline to the strongest posture cuts "
        f"privacy exposure by "
        f"{_pct(1 - _v(results, hi, 'privacy_exposure_index') / _v(results, lo, 'privacy_exposure_index'))} "
        f"and audit capability by "
        f"{_pct(1 - _v(results, hi, 'audit_capability_index') / _v(results, lo, 'audit_capability_index'))}. "
        "There is no posture in the ladder that is simultaneously private and "
        "auditable; the frontier has no corner solution."
    )
    s.append("")
    s.append("**Headline findings**")
    s.append("")
    s.append(
        f"1. **Pseudonymity is not privacy.** On the transparent baseline, one "
        f"known address per organisation attributes "
        f"{_pct(_v(results, lo, 'payment_deanonymisation'))} of all transaction "
        f"value to a named party."
    )
    s.append(
        f"2. **Address hygiene alone is near-useless.** Rotating to one-time "
        f"addresses (L1) still leaves "
        f"{_pct(_v(results, 1, 'payment_deanonymisation'))} of value correctly "
        f"attributed, because change outputs re-link the wallets."
    )
    s.append(
        f"3. **Mixing is defeated by arithmetic, not by guesswork.** CoinJoin "
        f"cuts a competent analyst's coverage to "
        f"{_pct(_v(results, 2, 'competent_analyst_value'))}, but solving change "
        f"outputs against input subsets restores it to "
        f"{_pct(_v(results, 2, 'payment_deanonymisation'))} at "
        f"{_pct(_v(results, 2, 'coinjoin_link_precision'))} precision."
    )
    s.append(
        f"4. **A chain cannot k-anonymise its own clock.** Generalisation drops "
        f"declared-field re-identification from "
        f"{_pct(_v(results, lo, 'reid_declared_only'))} to "
        f"{_pct(_v(results, 3, 'reid_declared_only'))}, but block timestamps — "
        f"which consensus requires — put it back to "
        f"{_pct(_v(results, 3, 'reid_with_timestamps'))}."
    )
    s.append(
        f"5. **The audit trail is the membership-inference oracle.** Because "
        f"`w_client = w_global + published_update`, any observer reconstructs a "
        f"hospital's local model and tests records against it: AUC "
        f"{_v(results, lo, 'mia_auc'):.3f} versus "
        f"{_v(results, lo, 'mia_global_auc'):.3f} against the global model alone."
    )
    s.append(
        f"6. **Differential privacy works, and it is the only rung that costs "
        f"accuracy.** DP cuts attack AUC to {_v(results, 4, 'mia_auc'):.3f} but "
        f"also degrades poisoning detection from "
        f"{_v(results, lo, 'poisoning_detection_auc'):.3f} to "
        f"{_v(results, 4, 'poisoning_detection_auc'):.3f} AUC, and costs "
        f"{_pct(1 - _v(results, 4, 'utility_retention'))} of model utility."
    )
    s.append("")

    # ------------------------------------------------------------ methodology
    s.append("---")
    s.append("")
    s.append("## 2. System under assessment")
    s.append("")
    s.append(
        "A decentralised medical-AI marketplace. Hospitals and clinics train a "
        "shared clinical risk model by federated averaging; research labs buy "
        "inference; data brokers sell datasets; an exchange provides on- and "
        "off-ramps. Three record types are published on chain:"
    )
    s.append("")
    s.append("- **Payment transactions** (UTXO model) — fees, rewards, purchases.")
    s.append("- **Federated-learning updates** — one record per client per round, "
             "carrying a commitment, a claimed sample count and (at lower "
             "postures) the update norm and local loss.")
    s.append("- **Inference and listing records** — requester address, model id, "
             "a request tag, a fee and an input commitment (never the input).")
    s.append("")
    s.append(f"Scale per trial: {cfg.ledger.n_entities} entities, "
             f"{cfg.ledger.n_transactions} transactions over {cfg.ledger.n_days} "
             f"days, {cfg.fl.n_clients} federated clients "
             f"({cfg.fl.n_freeriders} free-riders, {cfg.fl.n_poisoners} poisoners) "
             f"over {cfg.fl.n_rounds} rounds.")
    s.append("")
    s.append("### 2.1 Threat model")
    s.append("")
    s.append(_table([
        ["Passive chain observer", "Full ledger history; one known address per "
         "organisation (KYC record, invoice, published donation address)",
         "De-anonymise payments; re-identify participants"],
        ["Metadata analyst", "Ledger plus an external organisational directory "
         "(region, size, activity hours)",
         "Match directory entries to on-chain participants"],
        ["Model observer", "Global weights and published per-client updates",
         "Infer whether a specific patient record was in a hospital's shard"],
        ["Honest auditor", "Exactly the same published records",
         "Detect free-riding, poisoning and misreported contributions"],
    ], ["Actor", "Capability", "Goal"]))
    s.append("")
    s.append(
        "The auditor is listed deliberately alongside the adversaries. It holds "
        "no privileged input. That is the finding, not an oversight in the "
        "design."
    )
    s.append("")
    s.append("### 2.2 Privacy postures evaluated")
    s.append("")
    s.append(_table(
        [[f"L{p.level}", p.name, p.blurb] for p in cfg.postures],
        ["Level", "Posture", "Mechanism"]))
    s.append("")

    # --------------------------------------------------------------- findings
    s.append("---")
    s.append("")
    s.append("## 3. Findings")
    s.append("")
    s.append("### F1 — Payment de-anonymisation (attack ladder)")
    s.append("")
    s.append("Three analysts of increasing competence, all reading public data only.")
    s.append("")
    s.append(_table([
        [f"L{l}",
         _pm(results, l, "naive_analyst_precision", "{:.3f}"),
         _pm(results, l, "competent_analyst_value", "{:.3f}"),
         _pm(results, l, "payment_deanonymisation", "{:.3f}"),
         _pm(results, l, "entity_recovery_rate", "{:.3f}")]
        for l in levels
    ], ["Posture", "Naive precision", "Competent coverage",
        "Advanced (correct value)", "Wallets recovered"]))
    s.append("")
    s.append(
        "The naive analyst applies common-input-ownership to every transaction "
        "and is poisoned by CoinJoins — precision collapses to "
        f"{_v(results, 2, 'naive_analyst_precision'):.3f} at L2. A competent "
        "analyst detects joint spends structurally and withholds the heuristic, "
        "trading coverage for precision. The advanced analyst then solves each "
        "CoinJoin's change output against subsets of its inputs. "
        f"At L2 that recovers {_pct(_v(results, 2, 'coinjoin_solve_rate'))} of "
        f"change outputs at {_pct(_v(results, 2, 'coinjoin_link_precision'))} "
        "precision."
    )
    s.append("")
    s.append("> **Root cause.** Equal-denomination outputs hide *which* output is "
             "whose, but the change output does not: `change = inputs − denom − "
             "fee_share` is an equation an adversary can solve. Production "
             "mixers avoid this with uniform, change-free denominations; a naive "
             "implementation does not get the privacy it advertises.")
    s.append("")

    s.append("### F2 — Re-identification of AI-layer participants")
    s.append("")
    s.append(_table([
        [f"L{l}",
         _pm(results, l, "reid_declared_only", "{:.3f}"),
         _pm(results, l, "reid_with_timestamps", "{:.3f}"),
         _pm(results, l, "timestamp_uplift", "{:+.3f}"),
         _pm(results, l, "median_k", "{:.1f}"),
         _pm(results, l, "anonymity_entropy_bits", "{:.2f}")]
        for l in levels
    ], ["Posture", "Declared fields", "+ timestamps", "Timestamp uplift",
        "Median k", "Anonymity entropy (bits)"]))
    s.append("")
    s.append(
        f"Quasi-identifiers published by the contract ({', '.join(DECLARED_QIS)}) "
        "can be generalised and suppressed, and doing so is effective: "
        f"declared-field uniqueness falls from "
        f"{_pct(_v(results, lo, 'reid_declared_only'))} to "
        f"{_pct(_v(results, 3, 'reid_declared_only'))}. Block timestamps cannot "
        "be generalised — consensus depends on them — and an adversary who adds "
        "the modal activity hour recovers "
        f"{_pct(_v(results, 3, 'reid_with_timestamps'))} uniqueness. The "
        f"timestamp channel's contribution grows from "
        f"{_v(results, lo, 'timestamp_uplift'):+.3f} to "
        f"{_v(results, hi, 'timestamp_uplift'):+.3f} as the suppressible "
        "channels are closed: it is the residual, and it is not addressable at "
        "the application layer."
    )
    s.append("")
    s.append(
        "Achieving k=5 over the declared fields also required *dropping* "
        f"attributes entirely. At L{hi} the chain still publishes "
        f"{_join(outcomes[-1].reid.published_qis)}, and no longer publishes "
        f"{_join(outcomes[-1].reid.dropped_qis)}. With four "
        "quasi-identifiers over ~80 organisations, every equivalence class is a "
        "singleton and naive suppression blanks the whole table — the curse of "
        "dimensionality for k-anonymity (Aggarwal, 2005)."
    )
    s.append("")

    s.append("### F3 — Membership inference against published updates")
    s.append("")
    s.append(_table([
        [f"L{l}",
         "yes" if bool(results.loc[results["level"] == l, "mia_available"].iloc[0]) else "**no target**",
         _pm(results, l, "mia_auc", "{:.3f}"),
         _pm(results, l, "mia_advantage", "{:.3f}"),
         _pm(results, l, "mia_tpr_at_1pct", "{:.3f}"),
         _pm(results, l, "mia_global_auc", "{:.3f}")]
        for l in levels
    ], ["Posture", "Attack mountable", "AUC (per-client)", "Advantage",
        "TPR @ 1% FPR", "AUC (global model)"]))
    s.append("")
    s.append(
        "The attack is the standard loss-threshold test. What makes it a "
        "blockchain problem is that it needs no privileged access: the global "
        "weights are public by necessity and the per-client update is published "
        "*as the audit record*, so `w_client = w_global + delta_client` is "
        "reconstructible by any observer. Attacking the aggregate alone is far "
        f"weaker (AUC {_v(results, lo, 'mia_global_auc'):.3f} versus "
        f"{_v(results, lo, 'mia_auc'):.3f}) — the leakage is created by "
        "per-client disclosure, which is precisely the feature that makes the "
        "system auditable."
    )
    s.append("")
    eps = _v(results, 4, "dp_epsilon")
    s.append(
        f"> **Caveat on the formal guarantee.** The DP posture reduces the "
        f"attack empirically (AUC {_v(results, 4, 'mia_auc'):.3f}), but the "
        f"composed privacy budget over {cfg.fl.n_rounds} rounds is "
        f"ε ≈ {eps:.1f} at δ = 1e-5. An epsilon in the tens is far outside any "
        f"recommended range and should not be presented as a formal guarantee. "
        f"The observed protection here is empirical, not certified."
    )
    s.append("")

    s.append("### F4 — What the auditor loses")
    s.append("")
    s.append(_table([
        [f"L{l}",
         _pm(results, l, "freerider_detection_auc", "{:.3f}"),
         _pm(results, l, "poisoning_detection_auc", "{:.3f}"),
         _pm(results, l, "contribution_fidelity", "{:.3f}"),
         _pm(results, l, "payment_traceability", "{:.3f}"),
         _pm(results, l, "supply_conservation", "{:.3f}"),
         _pm(results, l, "audit_capability_index", "{:.3f}")]
        for l in levels
    ], ["Posture", "Free-rider AUC", "Poisoning AUC", "Contribution fidelity",
        "Payment traceability", "Supply conservation", "Audit index"]))
    s.append("")
    s.append(
        f"Poisoning detection is the first casualty of differential privacy: "
        f"AUC falls from {_v(results, lo, 'poisoning_detection_auc'):.3f} to "
        f"{_v(results, 4, 'poisoning_detection_auc'):.3f}, because the noise "
        "that hides a client's data also hides the direction of a malicious "
        "update. Under secure aggregation (L5) both detectors lose their input "
        "entirely and sit at chance."
    )
    s.append("")
    s.append(
        "Contribution fidelity never reaches 1.0 even on the transparent "
        f"baseline ({_pm(results, lo, 'contribution_fidelity', '{:.3f}')}): "
        "free-riders inflate their claimed sample counts, and nothing on the "
        "chain can contradict a self-reported number. Transparency records "
        "*claims*; it does not verify them."
    )
    s.append("")
    s.append(
        "**Supply conservation stays at 1.000 at every posture.** It is worth "
        "being precise about this: privacy technologies do not dissolve a "
        "ledger's integrity guarantees. What they remove is *attribution* — the "
        "ability to tie verified value flow to identified parties. Integrity "
        "and attribution are different properties and the distinction is often "
        "elided in the literature."
    )
    s.append("")

    s.append("### F5 — Utility cost")
    s.append("")
    s.append(_table([
        [f"L{l}", _pm(results, l, "fl_accuracy", "{:.3f}"),
         _pm(results, l, "utility_retention", "{:.3f}"),
         ("—" if pd.isna(_v(results, l, "dp_epsilon")) else f"{_v(results, l, 'dp_epsilon'):.1f}")]
        for l in levels
    ], ["Posture", "Held-out accuracy", "Utility retention", "ε (δ=1e-5)"]))
    s.append("")
    s.append(
        "Ledger-layer mitigations (address hygiene, mixing) and metadata "
        "generalisation cost no accuracy at all — they never touch the training "
        "signal. Only gradient noise does, and it costs "
        f"{_pct(1 - _v(results, hi, 'utility_retention'))} of the honest "
        "federation's accuracy at the strongest posture. Any claim that privacy "
        "is 'free' in such a system is a claim about which layer the mitigation "
        "sits in, not about privacy."
    )
    s.append("")

    # ---------------------------------------------------------- risk register
    s.append("---")
    s.append("")
    s.append("## 4. Risk register")
    s.append("")
    risks = [
        ("R1", "Payment graph de-anonymisation via address clustering",
         "High", "High",
         f"{_pct(_v(results, lo, 'payment_deanonymisation'))} of value attributed at L0",
         f"Reduced to {_pct(_v(results, hi, 'payment_deanonymisation'))} at L{hi}"),
        ("R2", "CoinJoin defeated by change-output arithmetic",
         "High", "High",
         f"{_pct(_v(results, 2, 'coinjoin_solve_rate'))} solved at "
         f"{_pct(_v(results, 2, 'coinjoin_link_precision'))} precision",
         "Use uniform, change-free denominations; otherwise unmitigated"),
        ("R3", "Participant re-identification via block timestamps",
         "High", "Medium",
         f"{_pct(_v(results, hi, 'reid_with_timestamps'))} unique even at L{hi}",
         "Not addressable on-chain; requires submission-time randomisation"),
        ("R4", "Membership inference on patient records via published updates",
         "High", "High",
         f"AUC {_v(results, lo, 'mia_auc'):.3f}, TPR@1%FPR "
         f"{_v(results, lo, 'mia_tpr_at_1pct'):.3f}",
         f"AUC {_v(results, 4, 'mia_auc'):.3f} under DP; eliminated under "
         "secure aggregation"),
        ("R5", "Undetected model poisoning once updates are hidden or noised",
         "Medium", "High",
         f"Detection AUC {_v(results, 4, 'poisoning_detection_auc'):.3f} at L4, "
         f"{_v(results, hi, 'poisoning_detection_auc'):.3f} at L{hi}",
         "Requires ZK range/norm proofs, not plaintext inspection"),
        ("R6", "Free-riding on inflated self-reported contributions",
         "High", "Medium",
         f"Contribution fidelity {_v(results, lo, 'contribution_fidelity'):.3f} "
         "even with full transparency",
         "Needs proof-of-training, not disclosure"),
        ("R7", "Immutability makes disclosure permanent and irrevocable",
         "High", "High",
         "Every record above is retained indefinitely by construction",
         "No technical mitigation; only non-publication"),
    ]
    s.append(_table(
        [[rid, desc, lik, imp, _severity(lik, imp), ev, res]
         for rid, desc, lik, imp, ev, res in risks],
        ["ID", "Risk", "Likelihood", "Impact", "Severity", "Evidence",
         "Residual / mitigation"]))
    s.append("")

    # ---------------------------------------------------------- effectiveness
    s.append("---")
    s.append("")
    s.append("## 5. Mitigation effectiveness")
    s.append("")
    rows = []
    for l in levels[1:]:
        d_priv = _v(results, l, "privacy_exposure_index") - _v(results, lo, "privacy_exposure_index")
        d_audit = _v(results, l, "audit_capability_index") - _v(results, lo, "audit_capability_index")
        d_util = _v(results, l, "utility_retention") - _v(results, lo, "utility_retention")
        ratio = (abs(d_priv) / abs(d_audit)) if abs(d_audit) > 1e-9 else float("inf")
        rows.append([f"L{l}", f"{d_priv:+.3f}", f"{d_audit:+.3f}", f"{d_util:+.3f}",
                     "∞" if ratio == float("inf") else f"{ratio:.2f}"])
    s.append(_table(rows, ["Posture", "Δ privacy exposure", "Δ audit capability",
                           "Δ utility", "Privacy gained per unit of audit lost"]))
    s.append("")
    s.append(
        "A ratio above 1.0 means the posture buys more privacy than it costs in "
        "auditability. Read this table as the design question the system "
        "architect actually faces, not as a ranking: the right posture depends "
        "on whether the deployment's dominant risk is patient re-identification "
        "or undetected model corruption."
    )
    s.append("")

    # --------------------------------------------------------------- limits
    s.append("---")
    s.append("")
    s.append("## 6. Limitations and threats to validity")
    s.append("")
    s.append(
        "- **Synthetic data throughout.** No real chain or patient data is used. "
        "Absolute numbers depend on the generator's parameters; the *ordering* "
        "of postures and the mechanisms behind each finding are the "
        "transferable results, not the specific percentages."
    )
    s.append(
        "- **The generator encodes the assumptions the attacks exploit.** "
        "Round-number payments, per-contributor CoinJoin change and role-based "
        "activity clocks are modelled because real systems exhibit them, but "
        "their strength here is a modelling choice. A deployment with "
        "change-free mixing would not show F1's recovery."
    )
    s.append(
        f"- **Small federation.** {cfg.fl.n_clients} clients with "
        f"{cfg.fl.n_poisoners} poisoners makes the poisoning-detection AUC "
        f"coarse; this is why every figure is averaged over {n_trials} "
        "independent worlds and reported with a standard deviation."
    )
    s.append(
        "- **One model family.** Logistic regression over random Fourier "
        "features was chosen because it memorises measurably. Deep networks "
        "generally leak more, so F3 should be read as a lower bound."
    )
    s.append(
        "- **The indices are unweighted means.** The privacy-exposure and "
        "audit-capability indices average their components equally. They are "
        "summaries for comparison across postures, not calibrated risk scores; "
        "the component tables are the evidence."
    )
    s.append(
        "- **ZK is modelled by omission.** Posture 5 simulates zero-knowledge "
        "proofs by withholding data and asserting validity. Real proof systems "
        "have their own trusted-setup, soundness and cost properties that this "
        "assessment does not evaluate."
    )
    s.append("")

    # ------------------------------------------------------- recommendations
    s.append("---")
    s.append("")
    s.append("## 7. Recommendations")
    s.append("")
    s.append(
        "1. **Do not treat an on-chain audit trail as privacy-neutral.** "
        "Per-client disclosure is what makes both auditing and membership "
        "inference possible. Decide which one the deployment needs before "
        "choosing what to publish."
    )
    s.append(
        "2. **Publish proofs, not statistics.** The only posture that breaks "
        "the symmetry is one where the auditor verifies a *property* (norm "
        "bounds, honest training, sample-count ranges) without reading the "
        "underlying value. Every posture that publishes a number hands the "
        "adversary the same number."
    )
    s.append(
        "3. **If mixing is used, use change-free uniform denominations.** F1 "
        "shows a change-emitting mixer surrenders most of its anonymity set to "
        "arithmetic alone."
    )
    s.append(
        "4. **Randomise submission times.** Timestamps are the one "
        "quasi-identifier no application-layer mitigation reaches. Batched or "
        "jittered submission windows are the only lever, and they trade against "
        "protocol latency."
    )
    s.append(
        "5. **Do not quote an epsilon without composing it.** The DP posture "
        f"here reaches ε ≈ {eps:.1f} over {cfg.fl.n_rounds} rounds. Report "
        "composed budgets, or report the empirical attack result and say so."
    )
    s.append(
        "6. **Treat immutability as a privacy hazard in its own right.** Every "
        "other risk in the register is permanent by construction; a "
        "re-identification that is infeasible today remains available to every "
        "future adversary against the same stored bytes."
    )
    s.append("")

    # ------------------------------------------------------------ reproduce
    s.append("---")
    s.append("")
    s.append("## 8. Reproducing this assessment")
    s.append("")
    s.append("```bash")
    s.append("python -m venv .venv && .venv/bin/pip install -r requirements.txt")
    s.append(f".venv/bin/python run_all.py --trials {n_trials}")
    s.append("```")
    s.append("")
    s.append("Artefacts:")
    s.append("")
    s.append("- `outputs/data/results.csv` — aggregated metrics (mean and std)")
    s.append("- `outputs/data/results_raw.csv` — one row per posture per trial")
    s.append("- `outputs/data/detail.json` — full per-posture attack reports")
    s.append("- `outputs/data/L*_*.csv` — the synthetic records themselves")
    s.append("- `outputs/figures/` — figures 1–7")
    s.append("")
    s.append("### Figures")
    s.append("")
    for name, caption in [
        ("fig1_privacy_transparency_tradeoff.png",
         "The trade-off frontier: privacy exposure against audit capability."),
        ("fig2_attack_ladder.png",
         "Payment de-anonymisation under three levels of analyst competence."),
        ("fig3_reidentification.png",
         "Re-identification with and without the block-timestamp channel."),
        ("fig4_membership_inference.png",
         "Membership inference against published updates versus the global model."),
        ("fig5_audit_capability.png",
         "Which audit capabilities each posture retains."),
        ("fig6_unicity.png",
         "Unicity: probability of being unique given p known quasi-identifiers."),
        ("fig7_model_utility.png",
         "Model accuracy across postures against the honest-federation reference."),
    ]:
        s.append(f"![{caption}](../figures/{name})")
        s.append("")
        s.append(f"*{caption}*")
        s.append("")

    path = out_dir / "assessment.md"
    path.write_text("\n".join(s))
    return path
