#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from prepare_gold_evaluation import contains_unresolved_unknown, normalize_text


SPLITS = ("development", "test")
REQUIRED_KEYS = (
    "id",
    "audio_filepath",
    "offset",
    "duration",
    "text",
    "airport",
    "recorded_date",
    "source_record_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: {error}") from error
            missing = [key for key in REQUIRED_KEYS if key not in row]
            if missing:
                raise RuntimeError(f"{path}:{line_number}: missing keys {missing}")
            rows.append(row)
    return rows


def source_xml(human_root: Path, record_id: str) -> Path:
    primary = human_root / "DATA" / f"{record_id}.xml"
    return primary if primary.is_file() else human_root / "DATA_nonEN" / f"{record_id}.xml"


def audit_source_reference(
    row: dict[str, Any],
    human_root: Path,
    xml_cache: dict[str, list[ElementTree.Element]],
) -> tuple[str, bool]:
    record_id = str(row["source_record_id"])
    if record_id not in xml_cache:
        path = source_xml(human_root, record_id)
        xml_cache[record_id] = ElementTree.parse(path).getroot().findall("segment")
    try:
        segment_index = int(str(row["id"]).rsplit("!", 1)[1]) - 1
        raw_text = xml_cache[record_id][segment_index].findtext("text", default="")
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"Cannot resolve source XML segment for {row['id']}") from error
    return raw_text, contains_unresolved_unknown(raw_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    parser.add_argument("human_root", type=Path)
    parser.add_argument("--parent-release", type=Path)
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--skip-audio-headers", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    report_path = args.release_root / "gold_report.json"
    release_report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_hashes = dict(release_report.get("sha256", {}))
    rows_by_split = {
        split: load_jsonl(args.release_root / f"{split}.jsonl")
        for split in SPLITS
    }

    ids_by_split = {split: {str(row["id"]) for row in rows} for split, rows in rows_by_split.items()}
    duplicate_ids = ids_by_split["development"] & ids_by_split["test"]
    if duplicate_ids:
        errors.append(f"development/test ID overlap: {sorted(duplicate_ids)[:20]}")

    groups_by_split = {
        split: {(str(row["airport"]), str(row["recorded_date"])) for row in rows}
        for split, rows in rows_by_split.items()
    }
    group_overlap = groups_by_split["development"] & groups_by_split["test"]
    if group_overlap:
        errors.append(f"development/test airport-day overlap: {sorted(group_overlap)[:20]}")

    xml_cache: dict[str, list[ElementTree.Element]] = {}
    unresolved_rows: list[str] = []
    normalization_mismatches: list[str] = []
    audio_properties: dict[str, tuple[float, int, int]] = {}
    split_details: dict[str, dict[str, Any]] = {}

    for split, rows in rows_by_split.items():
        uppercase = 0
        disallowed_symbols = 0
        missing_audio = 0
        for row in rows:
            text = str(row["text"])
            uppercase += any(character.isupper() for character in text)
            disallowed_symbols += any(
                not (character.isalnum() or character in {"'", " "}) for character in text
            )
            raw_text, has_unresolved_unknown = audit_source_reference(row, args.human_root, xml_cache)
            if has_unresolved_unknown:
                unresolved_rows.append(str(row["id"]))
            if normalize_text(raw_text) != text:
                normalization_mismatches.append(str(row["id"]))

            audio_path = str(row["audio_filepath"])
            if audio_path not in audio_properties:
                path = Path(audio_path)
                if not path.is_file():
                    missing_audio += 1
                    continue
                if not args.skip_audio_headers:
                    try:
                        with wave.open(audio_path, "rb") as audio:
                            audio_properties[audio_path] = (
                                audio.getnframes() / audio.getframerate(),
                                audio.getframerate(),
                                audio.getnchannels(),
                            )
                    except (EOFError, OSError, wave.Error) as error:
                        errors.append(f"unreadable WAV {audio_path}: {error}")
            if audio_path in audio_properties:
                wav_duration, sample_rate, channels = audio_properties[audio_path]
                offset = float(row["offset"])
                duration = float(row["duration"])
                if sample_rate != 16000 or channels != 1:
                    errors.append(f"audio is not 16 kHz mono: {audio_path}")
                if offset < 0 or duration < 0.1 or offset + duration > wav_duration + 0.02:
                    errors.append(f"invalid audio interval: {row['id']}")

        if uppercase or disallowed_symbols:
            errors.append(
                f"{split} transcript-style mismatch: uppercase={uppercase}, symbols={disallowed_symbols}"
            )
        if missing_audio:
            errors.append(f"{split} has {missing_audio} missing audio files")
        split_details[split] = {
            "segments": len(rows),
            "speech_hours": round(sum(float(row["duration"]) for row in rows) / 3600, 6),
            "reference_words": sum(len(str(row["text"]).split()) for row in rows),
            "recordings": len({str(row["source_record_id"]) for row in rows}),
            "airport_days": len(groups_by_split[split]),
            "airports": dict(sorted(Counter(str(row["airport"]) for row in rows).items())),
        }

    if unresolved_rows:
        errors.append(f"included references contain unresolved [unk]: {unresolved_rows[:20]}")
    if normalization_mismatches:
        errors.append(f"source/reference normalization mismatch: {normalization_mismatches[:20]}")

    actual_hashes: dict[str, str] = {}
    checksum_errors: list[str] = []
    for filename in ("development.jsonl", "test.jsonl", "source_windows.jsonl"):
        actual_hashes[filename] = sha256(args.release_root / filename)
        if expected_hashes.get(filename) != actual_hashes[filename]:
            checksum_errors.append(filename)
    if checksum_errors:
        errors.append(f"release checksum mismatch: {checksum_errors}")

    parent_comparison: dict[str, dict[str, Any]] = {}
    if args.parent_release:
        for filename in ("development.jsonl", "test.jsonl", "source_windows.jsonl"):
            parent_hash = sha256(args.parent_release / filename)
            parent_comparison[filename] = {
                "parent_sha256": parent_hash,
                "release_sha256": actual_hashes[filename],
                "byte_identical": parent_hash == actual_hashes[filename],
            }

    evaluation_groups = groups_by_split["development"] | groups_by_split["test"]
    training_overlap: dict[str, dict[str, int]] = {}
    for manifest in args.training_manifest:
        group_hits = 0
        id_hits = 0
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                group = (str(row.get("airport", "")), str(row.get("recorded_date", "")))
                group_hits += group in evaluation_groups
                id_hits += str(row.get("id", "")) in (ids_by_split["development"] | ids_by_split["test"])
        training_overlap[str(manifest)] = {
            "airport_day_rows": group_hits,
            "exact_id_rows": id_hits,
        }
        if group_hits or id_hits:
            errors.append(
                f"training leakage in {manifest}: airport_day_rows={group_hits}, exact_id_rows={id_hits}"
            )

    report = {
        "valid": not errors,
        "release": release_report.get("release"),
        "unknown_policy": release_report.get("unknown_policy"),
        "unresolved_unknown_references": len(unresolved_rows),
        "normalization_mismatches": len(normalization_mismatches),
        "splits": split_details,
        "development_test_id_overlap": len(duplicate_ids),
        "development_test_airport_day_overlap": len(group_overlap),
        "sha256": actual_hashes,
        "parent_comparison": parent_comparison,
        "training_overlap": training_overlap,
        "errors": errors,
    }
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
