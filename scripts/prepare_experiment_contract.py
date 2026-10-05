#!/usr/bin/env python3

"""Freeze manifest snapshots and write a reproducible ASR experiment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(path: Path) -> dict[str, object]:
    rows = 0
    seconds = 0.0
    empty_text = 0
    missing_audio = 0
    uppercase = 0
    punctuation_or_symbols = 0
    non_ascii = 0
    languages: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    language_scores: list[float] = []
    examples: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows += 1
            text = str(item.get("text", ""))
            if not text.strip():
                empty_text += 1
            if any(character.isupper() for character in text):
                uppercase += 1
            if any(unicodedata.category(character)[0] in {"P", "S"} for character in text):
                punctuation_or_symbols += 1
            if any(ord(character) > 127 for character in text):
                non_ascii += 1
            seconds += float(item.get("duration", 0.0))
            languages[str(item.get("lang", item.get("target_lang", "unspecified")))] += 1
            sources[str(item.get("source", "unspecified"))] += 1
            if item.get("language_score") is not None:
                language_scores.append(float(item["language_score"]))
            audio = Path(str(item.get("audio_filepath", "")))
            if not audio.is_file():
                missing_audio += 1
            if len(examples) < 5 and text.strip():
                examples.append(text)

    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": rows,
        "speech_hours": seconds / 3600,
        "empty_text": empty_text,
        "missing_audio": missing_audio,
        "style": {
            "uppercase_rows": uppercase,
            "uppercase_fraction": uppercase / rows if rows else None,
            "punctuation_or_symbol_rows": punctuation_or_symbols,
            "punctuation_or_symbol_fraction": punctuation_or_symbols / rows if rows else None,
            "non_ascii_rows": non_ascii,
            "non_ascii_fraction": non_ascii / rows if rows else None,
            "examples": examples,
        },
        "languages": dict(sorted(languages.items())),
        "sources": dict(sorted(sources.items())),
        "language_score": {
            "count": len(language_scores),
            "minimum": min(language_scores) if language_scores else None,
            "maximum": max(language_scores) if language_scores else None,
        },
    }


def copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Existing snapshot differs from source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--base-model-ref", required=True)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    args = parser.parse_args()

    for path in (args.train_manifest, args.validation_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    inputs = args.experiment_root / "inputs"
    train_snapshot = inputs / "train.jsonl"
    validation_snapshot = inputs / "validation.jsonl"
    copy_immutable(args.train_manifest, train_snapshot)
    copy_immutable(args.validation_manifest, validation_snapshot)

    train_audit = audit_manifest(train_snapshot)
    validation_audit = audit_manifest(validation_snapshot)
    if train_audit["empty_text"] or validation_audit["empty_text"]:
        raise RuntimeError("Empty transcripts found in frozen manifests")
    if train_audit["missing_audio"] or validation_audit["missing_audio"]:
        raise RuntimeError("Frozen manifests reference missing audio")
    if set(train_audit["languages"]) != {"en-US"}:
        raise RuntimeError(f"Unexpected training languages: {train_audit['languages']}")
    minimum_score = train_audit["language_score"]["minimum"]
    if minimum_score is None or minimum_score < 0.9:
        raise RuntimeError(f"Training language-score gate failed: {minimum_score}")

    contract = {
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "objective": "Controlled base-architecture comparison on the frozen Full HQ ATC pool",
        "base_model": {
            "requested_ref": args.base_model_ref,
            "family": "Parakeet",
            "architecture_expected": "CTC",
            "parameters": "1.1B",
            "language": "English",
            "exact_artifact_report": str(args.experiment_root / "base-model/model_report.json"),
        },
        "dataset": {
            "source_train_manifest": str(args.train_manifest),
            "source_validation_manifest": str(args.validation_manifest),
            "snapshot_policy": "byte-for-byte manifest copies; audio remains referenced on Lustre",
            "train": train_audit,
            "validation": validation_audit,
            "language_policy": "English decision and language_score >= 0.90; manifest lang=en-US",
            "blind_atc_test_evaluated": False,
        },
        "training": {
            "seed": 42,
            "precision": "bf16-mixed",
            "optimizer": "AdamW",
            "learning_rate": 3e-5,
            "schedule": "cosine with 1% warmup",
            "exposure_target": "five nominal passes through frozen training speech",
            "checkpoint_monitor": "val_wer/min",
            "checkpoint_retention": "top 5 plus last; preserve all stage logs and summaries",
        },
        "evaluation": {
            "atc": "frozen human-labelled ATC development manifest",
            "general": "frozen LibriSpeech test-clean",
            "metric": "normalized WER: lowercase, Unicode diacritic fold, punctuation/symbol removal, whitespace normalization",
            "variants": ["untouched base", "smoke", "final", "best-validation"],
        },
        "tracking": {
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "run_ids": {
                "preflight": f"{args.experiment_id}-preflight",
                "baseline": f"{args.experiment_id}-baseline",
                "smoke": f"{args.experiment_id}-smoke-seed-42",
                "full_training": f"{args.experiment_id}-full-seed-42",
                "final_evaluation": f"{args.experiment_id}-final-eval",
                "best_evaluation": f"{args.experiment_id}-best-eval",
            },
        },
        "separation": {
            "experiment_root": str(args.experiment_root),
            "run_root_template": str(args.experiment_root / "runs/<stage>/<slurm_job_id>"),
            "artifact_root": str(args.experiment_root / "artifacts"),
            "control_root": str(args.experiment_root / "control"),
            "shared_audio_is_read_only": True,
        },
    }
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    (args.experiment_root / "contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
