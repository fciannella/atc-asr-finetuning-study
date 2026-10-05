#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def stable_key(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def normalize_text(value: str) -> str:
    value = value.replace("[hes]", " uh ").replace("[unk]", " unknown ")
    value = re.sub(r"\[/?(?:#[^\]]+|NE(?: [^\]]+)?)\]", " ", value)
    value = re.sub(r"\(-[^)]*\)", " ", value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.lower()
    value = "".join(character if character.isalnum() or character in {"'", " "} else " " for character in value)
    return " ".join(value.split())


def contains_unresolved_unknown(value: str) -> bool:
    """Return true only for the annotation marker, not the spoken word unknown."""
    return re.search(r"\[unk\]", value, flags=re.IGNORECASE) is not None


def record_metadata(record_id: str) -> dict[str, str]:
    tokens = record_id.split("_")
    if len(tokens) < 7 or not re.fullmatch(r"\d{8}", tokens[-2]) or not re.fullmatch(r"\d{6}", tokens[-1]):
        raise ValueError(f"Cannot parse recording metadata from {record_id}")
    return {
        "airport": tokens[0],
        "channel": "_".join(tokens[2:-4]),
        "recorded_date": f"{tokens[-2][:4]}-{tokens[-2][4:6]}-{tokens[-2][6:]}",
        "recorded_timestamp": f"{tokens[-2]}{tokens[-1]}",
    }


def wav_properties(path: Path) -> tuple[float, int, int]:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate(), audio.getframerate(), audio.getnchannels()


def choose_development_sessions(
    session_seconds: dict[tuple[str, str], float], seed: str, development_fraction: float
) -> set[tuple[str, str]]:
    by_airport: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for session in session_seconds:
        by_airport[session[0]].append(session)

    development: set[tuple[str, str]] = set()
    singletons: list[tuple[str, str]] = []
    for airport, sessions in sorted(by_airport.items()):
        ordered = sorted(sessions, key=lambda item: stable_key(seed, f"{item[0]}:{item[1]}"))
        if len(ordered) == 1:
            singletons.extend(ordered)
            continue
        target = sum(session_seconds[item] for item in ordered) * development_fraction
        accumulated = 0.0
        for session in ordered[:-1]:
            if accumulated >= target:
                break
            development.add(session)
            accumulated += session_seconds[session]

    total_target = sum(session_seconds.values()) * development_fraction
    for session in sorted(singletons, key=lambda item: stable_key(seed, f"singleton:{item[0]}:{item[1]}")):
        current = sum(session_seconds[item] for item in development)
        if current < total_target:
            development.add(session)
    return development


def manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "segments": len(items),
        "speech_hours": round(sum(float(item["duration"]) for item in items) / 3600.0, 4),
        "recordings": len({str(item["source_record_id"]) for item in items}),
        "sessions": len({(str(item["airport"]), str(item["recorded_date"])) for item in items}),
        "airports": dict(sorted(Counter(str(item["airport"]) for item in items).items())),
        "duration_seconds": {
            "min": round(min((float(item["duration"]) for item in items), default=0.0), 3),
            "max": round(max((float(item["duration"]) for item in items), default=0.0), 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("human_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release", default="atc-gold-v1")
    parser.add_argument("--seed", default="atc-gold-v1")
    parser.add_argument("--development-fraction", type=float, default=0.35)
    parser.add_argument(
        "--unknown-policy",
        choices=("keep-as-unknown", "exclude-segment"),
        default="keep-as-unknown",
        help="How to handle the literal [unk] annotation marker in human references.",
    )
    parser.add_argument(
        "--immutable-output",
        action="store_true",
        help="Refuse to write into a non-empty output directory.",
    )
    args = parser.parse_args()
    if not 0.0 < args.development_fraction < 1.0:
        raise ValueError("--development-fraction must be between zero and one")

    if args.immutable_output and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty release directory: {args.output_dir}")

    accepted_path = args.human_root / "LISTS/accepted.list"
    accepted_rows = [line.strip() for line in accepted_path.open(encoding="utf-8") if line.strip()]
    accepted_ids = list(dict.fromkeys(accepted_rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exclusions: Counter[str] = Counter()
    adjustments: Counter[str] = Counter()
    samples_before_after: list[dict[str, str]] = []
    unresolved_unknown_examples: list[dict[str, object]] = []
    unresolved_unknown_segments_seen = 0
    segments: list[dict[str, Any]] = []
    source_windows: list[dict[str, Any]] = []

    for record_id in accepted_ids:
        data_directory = args.human_root / "DATA"
        if not (data_directory / f"{record_id}.wav").is_file():
            data_directory = args.human_root / "DATA_nonEN"
        wav_path = data_directory / f"{record_id}.wav"
        xml_path = data_directory / f"{record_id}.xml"
        if not wav_path.is_file() or not xml_path.is_file():
            exclusions["accepted_record_missing_wav_or_xml"] += 1
            continue
        try:
            wav_duration, sample_rate, channels = wav_properties(wav_path)
            root = ElementTree.parse(xml_path).getroot()
            metadata = record_metadata(record_id)
            start = datetime.strptime(metadata["recorded_timestamp"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except (ElementTree.ParseError, EOFError, OSError, ValueError, wave.Error):
            exclusions["unreadable_accepted_record"] += 1
            continue
        if sample_rate != 16000 or channels != 1:
            exclusions["accepted_record_not_16khz_mono"] += 1
            continue

        source_windows.append({
            "source_record_id": record_id,
            "airport": metadata["airport"],
            "channel": metadata["channel"],
            "start_utc": start.isoformat(),
            "duration": round(wav_duration, 6),
            "end_epoch": round(start.timestamp() + wav_duration, 6),
            "start_epoch": round(start.timestamp(), 6),
        })

        for segment_index, element in enumerate(root.findall("segment"), start=1):
            if element.findtext("speaker", default="").strip().upper() == "XT":
                exclusions["segment_crosstalk"] += 1
                continue
            if element.findtext("tags/correct_transcript", default="0") != "1":
                exclusions["segment_incorrect_transcript"] += 1
                continue
            if element.findtext("tags/non_english", default="0") == "1":
                exclusions["segment_non_english"] += 1
                continue
            try:
                offset = float(element.findtext("start", default="0"))
                end = float(element.findtext("end", default="0"))
            except ValueError:
                exclusions["segment_invalid_boundary"] += 1
                continue
            if end > wav_duration and end <= wav_duration + 0.15:
                end = wav_duration
                adjustments["segment_end_clamped_to_wav_within_150ms"] += 1
            duration = end - offset
            if offset < 0.0 or duration < 0.1 or end > wav_duration:
                exclusions["segment_invalid_boundary"] += 1
                continue
            raw_text = element.findtext("text", default="")
            unresolved_unknown = contains_unresolved_unknown(raw_text)
            if unresolved_unknown:
                unresolved_unknown_segments_seen += 1
                if len(unresolved_unknown_examples) < 20:
                    unresolved_unknown_examples.append({
                        "id": f"{record_id}!{segment_index:03d}",
                        "raw_text": raw_text,
                        "duration": round(duration, 3),
                    })
                if args.unknown_policy == "exclude-segment":
                    exclusions["segment_unresolved_unknown"] += 1
            text = normalize_text(raw_text)
            if not text:
                exclusions["segment_empty_normalized_text"] += 1
                continue
            if len(samples_before_after) < 12 and raw_text != text:
                samples_before_after.append({"id": record_id, "before": raw_text, "after": text})
            segments.append({
                "id": f"{record_id}!{segment_index:03d}",
                "audio_filepath": str(wav_path),
                "offset": round(offset, 3),
                "duration": round(duration, 3),
                "text": text,
                "target_lang": "en-US",
                "lang": "en-US",
                "airport": metadata["airport"],
                "channel": metadata["channel"],
                "recorded_date": metadata["recorded_date"],
                "source_record_id": record_id,
                "gold_acceptance": "human_rechecker_accepted",
                "_unresolved_unknown": unresolved_unknown,
            })

    session_seconds: Counter[tuple[str, str]] = Counter()
    for item in segments:
        session_seconds[(str(item["airport"]), str(item["recorded_date"]))] += float(item["duration"])
    development_sessions = choose_development_sessions(dict(session_seconds), args.seed, args.development_fraction)
    if args.unknown_policy == "exclude-segment":
        segments = [item for item in segments if not bool(item["_unresolved_unknown"])]
    for item in segments:
        item.pop("_unresolved_unknown", None)
    splits = {
        "development": [item for item in segments if (item["airport"], item["recorded_date"]) in development_sessions],
        "test": [item for item in segments if (item["airport"], item["recorded_date"]) not in development_sessions],
    }
    if any(not items for items in splits.values()):
        raise RuntimeError("Gold evaluation split is empty")

    manifest_hashes: dict[str, str] = {}
    for split, items in splits.items():
        items.sort(key=lambda item: str(item["id"]))
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest_hashes[path.name] = manifest_hash(path)

    windows_path = args.output_dir / "source_windows.jsonl"
    with windows_path.open("w", encoding="utf-8") as handle:
        for item in sorted(source_windows, key=lambda row: str(row["source_record_id"])):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest_hashes[windows_path.name] = manifest_hash(windows_path)

    report = {
        "release": args.release,
        "status": "frozen",
        "seed": args.seed,
        "development_fraction_target": args.development_fraction,
        "acceptance_contract": (
            "record ID appears in LISTS/accepted.list; segment correct_transcript=1; "
            "segment non_english=0; unresolved [unk] policy=" + args.unknown_policy
        ),
        "unknown_policy": args.unknown_policy,
        "unresolved_unknown_segments_seen": unresolved_unknown_segments_seen,
        "unresolved_unknown_examples": unresolved_unknown_examples,
        "split_group": "airport and UTC recording date",
        "accepted_list_rows": len(accepted_rows),
        "accepted_unique_recordings": len(accepted_ids),
        "duplicate_accepted_rows": len(accepted_rows) - len(accepted_ids),
        "eligible": summarize(segments),
        "splits": {name: summarize(items) for name, items in splits.items()},
        "source_windows": len(source_windows),
        "exclusions": dict(sorted(exclusions.items())),
        "adjustments": dict(sorted(adjustments.items())),
        "transcript_style": "NFKD; lowercase; punctuation and markup removed; whitespace normalized",
        "style_examples": samples_before_after,
        "sha256": manifest_hashes,
    }
    (args.output_dir / "gold_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
