#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).lower()
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = "".join(character if character.isalnum() or character in {"'", " "} else " " for character in value)
    return " ".join(value.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_word in reference:
        current = [previous[0] + 1]
        for index, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[index] + 1,
                previous[index - 1] + (reference_word != hypothesis_word),
            ))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors = 0
    words = 0
    utterances = 0
    raw_differences = 0
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            reference_raw = str(item.get("text", ""))
            hypothesis_raw = str(item.get("pred_text", ""))
            reference = normalize(reference_raw).split()
            hypothesis = normalize(hypothesis_raw).split()
            errors += edit_distance(reference, hypothesis)
            words += len(reference)
            utterances += 1
            raw_differences += reference_raw != hypothesis_raw
    result = {
        "manifest": str(args.manifest),
        "utterances": utterances,
        "reference_words": words,
        "word_errors": errors,
        "normalized_wer": errors / words if words else None,
        "normalization": "lowercase, Unicode diacritic fold, punctuation/symbol removal, whitespace normalization",
        "raw_text_differences": raw_differences,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

