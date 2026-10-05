#!/usr/bin/env python3

"""Rank Jacktol-derived LMs on ATCO2 Gold development only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--current-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-arms", type=int, default=20)
    parser.add_argument("--minimum-absolute-improvement", type=float, default=0.003)
    args = parser.parse_args()

    paths = sorted(args.sweep_dir.glob("*.json"))
    if len(paths) != args.expected_arms:
        raise ValueError(f"Expected {args.expected_arms} sweep arms, found {len(paths)}")
    arms = [json.loads(path.read_text()) for path in paths]
    if any(arm.get("status") != "complete" for arm in arms):
        raise ValueError("At least one sweep arm is incomplete")
    arms.sort(key=lambda arm: (arm["score"]["normalized_wer"], arm["label"]))

    current = json.loads(args.current_selection.read_text())
    current_winner = current["winner"]
    baseline_wer = float(current_winner["score"]["normalized_wer"])
    winner = arms[0]
    winner_wer = float(winner["score"]["normalized_wer"])
    improvement = baseline_wer - winner_wer
    promoted = improvement >= args.minimum_absolute_improvement
    report = {
        "status": "complete",
        "selection_set": "ATCO2 Gold development only",
        "jacktol_validation_evaluated": False,
        "community_test_evaluated": False,
        "control": current_winner,
        "minimum_absolute_improvement": args.minimum_absolute_improvement,
        "winner": winner,
        "absolute_improvement_vs_gold_only_lm": improvement,
        "promoted_to_secondary_validation": promoted,
        "decision": (
            "Evaluate the frozen winner on Jacktol validation and the LibriSpeech guardrail."
            if promoted else
            "Retain the Gold-only LM; no Jacktol-derived candidate cleared the development threshold."
        ),
        "ranking": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
