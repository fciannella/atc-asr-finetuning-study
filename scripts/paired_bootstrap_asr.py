#!/usr/bin/env python3

"""Paired utterance bootstrap for two ASR evaluation summaries."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import unicodedata
from pathlib import Path
from typing import Any


def normalize(text: str) -> list[str]:
    value = unicodedata.normalize("NFKD", text).lower()
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = "".join(character if character.isalnum() or character in {"'", " "} else " " for character in value)
    return " ".join(value.split()).split()


def distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_word in reference:
        current = [previous[0] + 1]
        for index, hypothesis_word in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[index] + 1,
                               previous[index - 1] + (reference_word != hypothesis_word)))
        previous = current
    return previous[-1]


def identity(item: dict[str, Any], index: int) -> str:
    if item.get("id") is not None:
        return f"id:{item['id']}"
    return "audio:{}:{}:{}".format(item.get("audio_filepath"), item.get("offset", 0), index)


def load_predictions(path: Path) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            reference = normalize(str(item.get("text", "")))
            result[identity(item, index)] = (len(reference), distance(reference, normalize(str(item.get("pred_text", "")))))
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def compare(baseline_path: Path, candidate_path: Path, samples: int, seed: int) -> dict[str, Any]:
    baseline, candidate = load_predictions(baseline_path), load_predictions(candidate_path)
    if set(baseline) != set(candidate):
        raise RuntimeError(f"Prediction identities differ: baseline={len(baseline)} candidate={len(candidate)}")
    keys = sorted(baseline)
    rows = [(baseline[key][0], baseline[key][1], candidate[key][1]) for key in keys]
    words = sum(row[0] for row in rows)
    baseline_wer = sum(row[1] for row in rows) / words
    candidate_wer = sum(row[2] for row in rows) / words
    generator = random.Random(seed)
    deltas = []
    for _ in range(samples):
        sampled = [rows[generator.randrange(len(rows))] for _ in rows]
        sampled_words = sum(row[0] for row in sampled)
        deltas.append((sum(row[2] for row in sampled) - sum(row[1] for row in sampled)) / sampled_words)
    return {
        "utterances": len(rows), "reference_words": words,
        "baseline_wer": baseline_wer, "candidate_wer": candidate_wer,
        "candidate_minus_baseline": candidate_wer - baseline_wer,
        "candidate_minus_baseline_95pct_ci": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "probability_candidate_is_not_better": sum(delta >= 0 for delta in deltas) / len(deltas),
        "bootstrap_samples": samples, "bootstrap_seed": seed,
        "mean_bootstrap_delta": statistics.fmean(deltas),
        "baseline_predictions": str(baseline_path), "candidate_predictions": str(candidate_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_summary", type=Path)
    parser.add_argument("candidate_summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_summary.read_text())
    candidate = json.loads(args.candidate_summary.read_text())
    fields = ["atc_development", "jacktol_test", "uwb_test", "general_domain", "atco2_gold_test"]
    comparisons = {}
    for offset, field in enumerate(fields):
        left, right = baseline.get(field), candidate.get(field)
        if left and right:
            comparisons[field] = compare(Path(left["manifest"]), Path(right["manifest"]), args.samples, args.seed + offset)
    payload = {
        "status": "complete", "method": "paired utterance bootstrap over normalized word-error counts",
        "baseline_summary": str(args.baseline_summary), "candidate_summary": str(args.candidate_summary),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
