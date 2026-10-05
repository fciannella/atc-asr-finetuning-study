#!/usr/bin/env python3

"""Compile the ATCO2 Gold-development n-gram sweep without opening the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--beam-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-arms", type=int, default=24)
    parser.add_argument("--minimum-absolute-improvement", type=float, default=0.005)
    args = parser.parse_args()

    paths = sorted(args.sweep_dir.glob("*.json"))
    if len(paths) != args.expected_arms:
        raise ValueError(f"Expected {args.expected_arms} sweep arms, found {len(paths)}")
    arms = [json.loads(path.read_text()) for path in paths]
    if any(arm.get("status") != "complete" for arm in arms):
        raise ValueError("At least one sweep arm is incomplete")
    arms.sort(key=lambda arm: (arm["score"]["normalized_wer"], arm["label"]))

    smoke = json.loads(args.smoke_summary.read_text())
    baseline = json.loads(args.beam_baseline.read_text())
    beam_wer = float(baseline["atc_development"]["normalized_wer"])
    malsd_wer = float(smoke["arms"]["malsd_control"]["score"]["normalized_wer"])
    winner = arms[0]
    winner_wer = float(winner["score"]["normalized_wer"])
    improvement = beam_wer - winner_wer
    promoted = improvement >= args.minimum_absolute_improvement

    report = {
        "status": "complete",
        "selection_set": "ATCO2 Gold development only",
        "community_test_evaluated": False,
        "ordinary_beam4_baseline_wer": beam_wer,
        "malsd_no_lm_control_wer": malsd_wer,
        "minimum_absolute_improvement": args.minimum_absolute_improvement,
        "winner": winner,
        "absolute_improvement": improvement,
        "promoted_to_general_guardrail": promoted,
        "decision": (
            "Run the frozen winner on LibriSpeech test-clean before any locked ATCO2 test pass."
            if promoted
            else "Do not open the locked ATCO2 test; the development gain is below threshold."
        ),
        "ranking": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
