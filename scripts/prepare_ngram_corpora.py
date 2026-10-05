#!/usr/bin/env python3

"""Prepare leakage-safe text corpora for the ATCO2 offline n-gram pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


UNKNOWN_RE = re.compile(r"\[(?:unk|unknown)\]", re.IGNORECASE)
IDENTITY_KEYS = ("id", "record_id", "source_record_id", "audio_filepath", "audio_path")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).casefold()
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in value
    )
    return " ".join(value.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(item)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def identity_sets(records: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    result = {key: set() for key in IDENTITY_KEYS}
    for item in records:
        for key in IDENTITY_KEYS:
            value = item.get(key)
            if value not in (None, ""):
                result[key].add(str(value))
    return result


def assert_isolated(
    train: list[dict[str, Any]], development: list[dict[str, Any]], test: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    splits = {
        "gold_train": identity_sets(train),
        "gold_development": identity_sets(development),
        "gold_community_test": identity_sets(test),
    }
    comparisons: dict[str, dict[str, int]] = {}
    pairs = (
        ("gold_train", "gold_development"),
        ("gold_train", "gold_community_test"),
        ("gold_development", "gold_community_test"),
    )
    for left, right in pairs:
        label = f"{left}__{right}"
        counts = {
            key: len(splits[left][key] & splits[right][key])
            for key in IDENTITY_KEYS
        }
        comparisons[label] = counts
        if any(counts.values()):
            raise ValueError(f"Split leakage detected for {label}: {counts}")
    return comparisons


def training_text(records: Iterable[dict[str, Any]], source: str) -> list[str]:
    result: list[str] = []
    for index, item in enumerate(records, 1):
        raw = str(item.get("text", ""))
        if UNKNOWN_RE.search(raw):
            raise ValueError(f"{source} row {index} contains an unresolved unknown marker")
        text = normalize_text(raw)
        if not text:
            raise ValueError(f"{source} row {index} has empty normalized text")
        result.append(text)
    return result


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-train", type=Path, required=True)
    parser.add_argument("--gold-development", type=Path, required=True)
    parser.add_argument("--gold-community-test", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-repeat", type=int, default=64)
    parser.add_argument("--general-share", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    for path in (
        args.gold_train,
        args.gold_development,
        args.gold_community_test,
        args.general_train,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.gold_repeat < 1:
        raise ValueError("--gold-repeat must be positive")
    if not 0.0 < args.general_share < 1.0:
        raise ValueError("--general-share must be between zero and one")

    gold_train = read_jsonl(args.gold_train)
    gold_development = read_jsonl(args.gold_development)
    gold_test = read_jsonl(args.gold_community_test)
    isolation = assert_isolated(gold_train, gold_development, gold_test)

    gold_text = training_text(gold_train, "gold train")
    general_records = read_jsonl(args.general_train)
    general_candidates: list[tuple[str, dict[str, Any]]] = []
    for item in general_records:
        raw = str(item.get("text", ""))
        if UNKNOWN_RE.search(raw):
            continue
        text = normalize_text(raw)
        if text:
            general_candidates.append((text, item))

    rng = random.Random(args.seed)
    rng.shuffle(general_candidates)
    repeated_gold = gold_text * args.gold_repeat
    gold_tokens = sum(len(text.split()) for text in repeated_gold)
    target_general_tokens = math.ceil(gold_tokens * args.general_share / (1.0 - args.general_share))
    selected_general: list[tuple[str, dict[str, Any]]] = []
    selected_general_tokens = 0
    for text, item in general_candidates:
        selected_general.append((text, item))
        selected_general_tokens += len(text.split())
        if selected_general_tokens >= target_general_tokens:
            break
    if selected_general_tokens < target_general_tokens:
        raise ValueError(
            f"General pool has only {selected_general_tokens} usable tokens; "
            f"{target_general_tokens} required"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = args.output_dir / "gold-train.txt"
    balanced_path = args.output_dir / "gold-general-80to20.txt"
    general_sample_path = args.output_dir / "general-sample.jsonl"
    write_lines(gold_path, gold_text)
    write_lines(balanced_path, [*repeated_gold, *(text for text, _ in selected_general)])
    with general_sample_path.open("w", encoding="utf-8") as handle:
        for text, item in selected_general:
            provenance = {
                key: item[key]
                for key in IDENTITY_KEYS
                if key in item and item[key] not in (None, "")
            }
            provenance["text"] = text
            handle.write(json.dumps(provenance, ensure_ascii=False) + "\n")

    balanced_tokens = gold_tokens + selected_general_tokens
    report = {
        "status": "complete",
        "policy": {
            "atco2_supervision": "human-gold training split only",
            "silver_atco2_used": False,
            "development_reference_text_used": False,
            "community_test_reference_text_used": False,
            "general_english_role": "decoder-LM retention corpus only",
        },
        "seed": args.seed,
        "sources": {
            "gold_train": {
                "path": str(args.gold_train),
                "sha256": sha256(args.gold_train),
                "records": len(gold_train),
            },
            "gold_development": {
                "path": str(args.gold_development),
                "sha256": sha256(args.gold_development),
                "records": len(gold_development),
                "use": "identifier-only isolation audit",
            },
            "gold_community_test": {
                "path": str(args.gold_community_test),
                "sha256": sha256(args.gold_community_test),
                "records": len(gold_test),
                "use": "identifier-only isolation audit",
            },
            "general_train": {
                "path": str(args.general_train),
                "sha256": sha256(args.general_train),
                "records": len(general_records),
            },
        },
        "isolation": isolation,
        "corpora": {
            "gold_train": {
                "path": str(gold_path),
                "sha256": sha256(gold_path),
                "lines": len(gold_text),
                "tokens": sum(len(text.split()) for text in gold_text),
            },
            "gold_general_80to20": {
                "path": str(balanced_path),
                "sha256": sha256(balanced_path),
                "lines": len(repeated_gold) + len(selected_general),
                "tokens": balanced_tokens,
                "gold_repeat": args.gold_repeat,
                "gold_tokens": gold_tokens,
                "general_records": len(selected_general),
                "general_tokens": selected_general_tokens,
                "realized_general_token_share": selected_general_tokens / balanced_tokens,
            },
            "general_sample": {
                "path": str(general_sample_path),
                "sha256": sha256(general_sample_path),
            },
        },
        "normalization": "casefold; NFKD diacritic fold; Unicode punctuation/symbol removal; whitespace collapse",
    }
    report_path = args.output_dir / "corpus-build-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
