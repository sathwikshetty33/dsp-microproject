#!/usr/bin/env python3
"""Run the full privacy/transparency assessment and write every artefact.

    python run_all.py              # full run
    python run_all.py --quick      # smaller ledger and federation, for a smoke test
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from bcai.config import DATA_DIR, DEFAULT_CONFIG, FIGURE_DIR, REPORT_DIR, Config
from bcai.pipeline import run_experiment


def build_config(quick: bool, seed: int | None) -> Config:
    cfg = DEFAULT_CONFIG
    if seed is not None:
        cfg = replace(cfg, seed=seed)
    if quick:
        cfg = replace(
            cfg,
            ledger=replace(cfg.ledger, n_entities=60, n_transactions=1_200, n_days=30),
            fl=replace(cfg.fl, n_clients=8, n_rounds=8, local_steps=60,
                       n_components=192, n_freeriders=2, n_poisoners=1),
        )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke-test run")
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    parser.add_argument("--trials", type=int, default=5, help="independent repetitions")
    parser.add_argument("--no-figures", action="store_true", help="skip plotting")
    args = parser.parse_args()

    cfg = build_config(args.quick, args.seed)
    cfg.ensure_dirs()

    print("=" * 74)
    print("Privacy and Security Limitations of Blockchain AI")
    print("Synthetic ledger + on-chain federated learning, 6 privacy postures")
    print("=" * 74)

    t0 = time.time()
    results, outcomes = run_experiment(cfg, n_repeats=args.trials)
    print(f"\nExperiment finished in {time.time() - t0:.1f}s")

    if not args.no_figures:
        from bcai.viz.figures import render_all
        print("Rendering figures ...")
        render_all(results, outcomes, FIGURE_DIR)

    from bcai.report.assessment import write_assessment
    path = write_assessment(cfg, results, outcomes, REPORT_DIR)
    print(f"\nAssessment written to {path}")
    print(f"Figures in {FIGURE_DIR}")
    print(f"Raw records and metrics in {DATA_DIR}")


if __name__ == "__main__":
    main()
