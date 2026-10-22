# Privacy and Security Limitations of Blockchain AI

A quantitative security and privacy assessment of **on-chain federated
learning**, built on fully synthetic blockchain records.

The system under study is a decentralised medical-AI marketplace: hospitals and
clinics train a shared clinical risk model by federated averaging, publishing
one update record per round to a public ledger so that free-riding and model
poisoning are auditable. That audit trail is the selling point. This project
measures what it costs.

> **The finding in one sentence.** The audit trail and the attack surface are
> the same object — every privacy-enhancing technology evaluated here removes
> adversarial capability and auditor capability together, because both parties
> read identical bytes.

All data is synthetic. No real chain data, no real patient data.

---

## What it does

The project builds a complete system and then attacks it, six times over, at
six escalating privacy postures:

| Level | Posture | Mechanism added |
|---|---|---|
| L0 | Transparent baseline | Plain public ledger, all metadata in the clear |
| L1 | Address hygiene | One-time addresses, change rotation |
| L2 | + mixing | CoinJoin-style joint transactions |
| L3 | + k-anonymity | Generalised & suppressed quasi-identifiers |
| L4 | + differential privacy | Clipped updates, Gaussian noise, Laplace counts |
| L5 | + zero-knowledge | Only commitments and validity proofs on chain |

### Attacks implemented

| Attack | Target | Method |
|---|---|---|
| **Address clustering** | Payment layer | Common-input-ownership + change-address heuristics over a union-find, scored against ground-truth ownership |
| **CoinJoin subset-sum** | Mixing | Solves `change = inputs − denomination − fee_share` against input subsets to recover provable co-ownership |
| **Re-identification** | AI-layer metadata | k-anonymity / unicity analysis over published quasi-identifiers, with and without the block-timestamp channel |
| **Membership inference** | Federated updates | Reconstructs `w_client = w_global + published_update` from public state and runs a loss-threshold test |

### Audit capabilities measured

Free-rider detection, poisoning detection, contribution fidelity, payment
traceability, metadata completeness, and supply conservation — each computed
from *exactly the same published records* the attacks read.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_all.py                 # full run, 5 trials per posture
.venv/bin/python run_all.py --quick         # fast smoke test
.venv/bin/python run_all.py --trials 10     # more repetitions
.venv/bin/python -m pytest tests/ -q        # 32 correctness tests
```

Outputs land in `outputs/`:

```
outputs/
├── report/assessment.md     # the deliverable: findings, risk register, recommendations
├── figures/fig1..fig7.png   # the charts
└── data/
    ├── results.csv          # aggregated metrics (mean + std across trials)
    ├── results_raw.csv      # one row per posture per trial
    ├── detail.json          # full per-posture attack reports
    └── L*_*.csv             # the synthetic records themselves
```

---

## Project structure

```
bcai/
├── config.py            # the privacy-posture ladder and all experiment sizing
├── pipeline.py          # runs every attack and audit at every posture
├── synth/
│   ├── entities.py      # population with behavioural profiles (clocks, habits)
│   ├── ledger.py        # UTXO ledger: payments, change, CoinJoins
│   └── ai_records.py    # inference requests and dataset listings
├── fl/
│   ├── data.py          # non-IID clinical shards + distribution-matched non-members
│   ├── model.py         # logistic regression over random Fourier features
│   └── federated.py     # FedAvg with free-riders, poisoners, DP, and on-chain records
├── attacks/
│   ├── clustering.py    # heuristic de-anonymisation + CoinJoin detection
│   ├── linkage.py       # subset-sum attack on mixing
│   ├── reidentify.py    # quasi-identifier re-identification
│   └── membership.py    # membership inference
├── metrics/
│   ├── privacy.py       # k-anonymity, unicity, AUC, advantage, TPR@FPR
│   └── transparency.py  # the auditor's six capabilities
├── mitigations/kanon.py # generalisation, suppression, adaptive k-anonymity
├── viz/                 # validated palette + figures
└── report/assessment.py # generates the assessment from measured results
```

---

## Methodology notes

**Why the comparison is fair.** The entity population — who exists, their
regions, sizes, time zones and habits — is drawn from a posture-independent
random stream. A posture changes only *how that world's activity is recorded on
chain*. Without this split, a posture comparison would partly measure a
different world rather than a different mitigation.

**Why membership inference is scored the way it is.** Non-members are drawn
from each client's *own* distribution, not from a generic population holdout.
Scoring against a population holdout would measure covariate shift and report
it as memorisation.

**Why the model memorises.** Logistic regression over random Fourier features
has enough capacity to fit individual rows, and the clinical label is
deliberately noisy (Bayes accuracy ≈ 0.72). A model with no capacity to
memorise would make the privacy analysis vacuous.

**Why every number is repeated.** Poisoning-detection AUC over two poisoners
and rank correlation over sixteen clients are coarse on a single draw. Every
posture is run on several independent worlds and reported with a standard
deviation.

**Why the auditor is in the threat model.** It holds no privileged input. That
is the result, not an oversight.

---

## Limitations

The absolute numbers depend on the generator's parameters; the *ordering* of
postures and the mechanisms behind each finding are the transferable results.
The generator encodes the assumptions the attacks exploit — round-number
payments, per-contributor CoinJoin change, role-based activity clocks — because
real systems exhibit them, but their strength here is a modelling choice.
Zero-knowledge is modelled by withholding data and asserting validity; real
proof systems have trusted-setup, soundness and cost properties this assessment
does not evaluate. See §6 of the assessment for the full list.
