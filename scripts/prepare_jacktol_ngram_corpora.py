#!/usr/bin/env python3

"""Prepare leakage-safe Jacktol and ATCO2 Gold text corpora for KenLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from prepare_ngram_corpora import (
    IDENTITY_KEYS,
    identity_sets,
    normalize_text,
    read_jsonl,
    sha256,
    write_lines,
)


UNKNOWN_RE = re.compile(r"(?i)(?:\[|<)?(?:unk|unknown)(?:\]|>)?")


def audit_isolation(splits: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    identities = {name: identity_sets(records) for name, records in splits.items()}
    result: dict[str, dict[str, int]] = {}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            label = f"{left}__{right}"
            counts = {
                key: len(identities[left][key] & identities[right][key])
                for key in IDENTITY_KEYS
            }
            result[label] = counts
            if any(counts.values()):
                raise ValueError(f"Split leakage detected for {label}: {counts}")
    return result


def usable_text(records: list[dict[str, Any]], source: str) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    exclusions: list[dict[str, Any]] = []
    for row, item in enumerate(records, 1):
        raw = str(item.get("text", ""))
        if UNKNOWN_RE.search(raw):
            exclusions.append({"row": row, "id": item.get("id"), "reason": "unknown_marker"})
            continue
        text = normalize_text(raw)
        if not text:
            raise ValueError(f"{source} row {row} has empty normalized text")
        lines.append(text)
    if not lines:
        raise ValueError(f"No usable text in {source}")
    return lines, exclusions


def source_report(path: Path, records: list[dict[str, Any]], use: str) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "records": len(records), "use": use}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-train", type=Path, required=True)
    parser.add_argument("--gold-development", type=Path, required=True)
    parser.add_argument("--gold-community-test", type=Path, required=True)
    parser.add_argument("--jacktol-train", type=Path, required=True)
    parser.add_argument("--jacktol-validation", type=Path, required=True)
    parser.add_argument("--jacktol-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "gold_train": args.gold_train,
        "gold_development": args.gold_development,
        "gold_community_test": args.gold_community_test,
        "jacktol_train": args.jacktol_train,
        "jacktol_validation": args.jacktol_validation,
        "jacktol_test": args.jacktol_test,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    splits = {name: read_jsonl(path) for name, path in paths.items()}
    isolation = audit_isolation(splits)
    gold_text, gold_exclusions = usable_text(splits["gold_train"], "ATCO2 Gold train")
    jacktol_text, jacktol_exclusions = usable_text(splits["jacktol_train"], "Jacktol train")

    gold_tokens_per_copy = sum(len(text.split()) for text in gold_text)
    jacktol_tokens = sum(len(text.split()) for text in jacktol_text)
    gold_repeat = max(1, round(jacktol_tokens / gold_tokens_per_copy))
    repeated_gold = gold_text * gold_repeat
    repeated_gold_tokens = gold_tokens_per_copy * gold_repeat
    balanced = [*repeated_gold, *jacktol_text]
    balanced_tokens = repeated_gold_tokens + jacktol_tokens

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jacktol_path = args.output_dir / "jacktol-train.txt"
    balanced_path = args.output_dir / "gold-jacktol-balanced.txt"
    write_lines(jacktol_path, jacktol_text)
    write_lines(balanced_path, balanced)

    report = {
        "status": "complete",
        "policy": {
            "training_text_sources": ["ATCO2 Gold train", "Jacktol train"],
            "silver_atco2_used": False,
            "atco2_development_reference_text_used": False,
            "atco2_community_test_reference_text_used": False,
            "jacktol_validation_reference_text_used": False,
            "jacktol_test_reference_text_used": False,
        },
        "sources": {
            name: source_report(
                path,
                splits[name],
                "LM training text" if name in {"gold_train", "jacktol_train"}
                else "identifier-only isolation audit",
            )
            for name, path in paths.items()
        },
        "isolation": isolation,
        "exclusions": {
            "gold_train": gold_exclusions,
            "jacktol_train": jacktol_exclusions,
        },
        "corpora": {
            "jacktol_train": {
                "path": str(jacktol_path),
                "sha256": sha256(jacktol_path),
                "lines": len(jacktol_text),
                "tokens": jacktol_tokens,
            },
            "gold_jacktol_balanced": {
                "path": str(balanced_path),
                "sha256": sha256(balanced_path),
                "lines": len(balanced),
                "tokens": balanced_tokens,
                "gold_unique_lines": len(gold_text),
                "gold_repeat": gold_repeat,
                "gold_tokens": repeated_gold_tokens,
                "jacktol_lines": len(jacktol_text),
                "jacktol_tokens": jacktol_tokens,
                "realized_gold_token_share": repeated_gold_tokens / balanced_tokens,
                "realized_jacktol_token_share": jacktol_tokens / balanced_tokens,
            },
        },
        "normalization": "casefold; NFKD diacritic fold; Unicode punctuation/symbol removal; whitespace collapse",
    }
    output = args.output_dir / "corpus-build-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
