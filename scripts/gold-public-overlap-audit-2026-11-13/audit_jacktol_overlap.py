#!/usr/bin/env python3

"""Audit and quarantine jacktol overlap from ATCO2 train/development data.

Transcript similarity is used only to generate candidates.  A candidate is
quarantined only after an acoustic comparison, and the complete source
recording is removed from training when a match is confirmed.  This is
deliberately conservative because jacktol contains a public ATCO2 subset.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, resample_poly


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()


def ngrams(tokens: tuple[str, ...], size: int = 4) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return {tokens} if tokens else set()
    return {tokens[index : index + size] for index in range(len(tokens) - size + 1)}


def token_jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_counts, right_counts = Counter(left), Counter(right)
    intersection = sum((left_counts & right_counts).values())
    union = sum((left_counts | right_counts).values())
    return intersection / union if union else 1.0


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except Exception as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from error


@functools.lru_cache(maxsize=4096)
def load_audio(path_text: str, offset: float, duration: float, target_sr: int = 8000) -> np.ndarray:
    path = Path(path_text)
    with sf.SoundFile(path) as handle:
        start = max(0, round(offset * handle.samplerate))
        frames = -1 if duration <= 0 else max(1, round(duration * handle.samplerate))
        handle.seek(start)
        samples = handle.read(frames=frames, dtype="float32", always_2d=True)
        sample_rate = handle.samplerate
    mono = samples.mean(axis=1)
    if sample_rate != target_sr:
        divisor = math.gcd(sample_rate, target_sr)
        mono = resample_poly(mono, target_sr // divisor, sample_rate // divisor).astype(np.float32)
    mono = mono - float(np.mean(mono))
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 0:
        mono = mono / peak
    return mono.astype(np.float32, copy=False)


def aligned_views(left: np.ndarray, right: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        length = min(len(left) - lag, len(right))
        return left[lag : lag + length], right[:length]
    length = min(len(left), len(right) + lag)
    return left[:length], right[-lag : -lag + length]


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def acoustic_similarity(left: np.ndarray, right: np.ndarray, sample_rate: int = 8000) -> dict[str, float]:
    if min(len(left), len(right)) < sample_rate // 4:
        return {"envelope_correlation": 0.0, "waveform_correlation": 0.0, "overlap_seconds": 0.0}
    hop = max(1, sample_rate // 100)
    left_env = np.convolve(np.abs(left), np.ones(hop, dtype=np.float32) / hop, mode="valid")[::hop]
    right_env = np.convolve(np.abs(right), np.ones(hop, dtype=np.float32) / hop, mode="valid")[::hop]
    left_z = left_env - float(np.mean(left_env))
    right_z = right_env - float(np.mean(right_env))
    correlation = correlate(left_z, right_z, mode="full", method="fft")
    lags = correlation_lags(len(left_z), len(right_z), mode="full")
    best_lag_env = int(lags[int(np.argmax(correlation))])
    aligned_left_env, aligned_right_env = aligned_views(left_env, right_env, best_lag_env)
    envelope_corr = pearson(aligned_left_env, aligned_right_env)

    predicted_lag = best_lag_env * hop
    best_waveform_corr = 0.0
    best_overlap = 0
    for lag in range(predicted_lag - hop, predicted_lag + hop + 1, max(1, hop // 8)):
        aligned_left, aligned_right = aligned_views(left, right, lag)
        if len(aligned_left) < sample_rate // 2:
            continue
        value = abs(pearson(aligned_left, aligned_right))
        if value > best_waveform_corr:
            best_waveform_corr = value
            best_overlap = len(aligned_left)
    return {
        "envelope_correlation": envelope_corr,
        "waveform_correlation": best_waveform_corr,
        "overlap_seconds": best_overlap / sample_rate,
    }


def collect_jacktol(release_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[tuple[str, ...], set[int]]]:
    records: list[dict[str, Any]] = []
    exact: dict[str, list[int]] = defaultdict(list)
    gram_index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for split in ("train", "validation", "test"):
        manifest = release_root / "manifests" / f"{split}.jsonl"
        for row in iter_jsonl(manifest):
            normalized = normalize_text(str(row.get("text", "")))
            tokens = tuple(normalized.split())
            row["_normalized_text"] = normalized
            row["_tokens"] = tokens
            row["_split"] = split
            index = len(records)
            records.append(row)
            exact[normalized].append(index)
            for gram in ngrams(tokens):
                gram_index[gram].add(index)
    return records, exact, gram_index


def candidate_pairs(
    manifest: Path,
    role: str,
    jacktol: list[dict[str, Any]],
    exact: dict[str, list[int]],
    gram_index: dict[tuple[str, ...], set[int]],
    fuzzy_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total, hours = 0, 0.0
    exact_rows, fuzzy_rows = 0, 0
    for row in iter_jsonl(manifest):
        total += 1
        duration = float(row.get("duration", 0.0))
        hours += duration / 3600
        normalized = normalize_text(str(row.get("text", "")))
        tokens = tuple(normalized.split())
        exact_pool = list(exact.get(normalized, ()))
        # Very short ATC phrases (for example "thank you") occur hundreds of
        # times and are not distinctive enough to justify an all-pairs audio
        # comparison.  Retain only the closest-duration candidates; longer
        # utterances from the same source recording remain fully searchable.
        if len(tokens) < 4 and len(exact_pool) > 25:
            exact_pool.sort(key=lambda index: abs(duration - float(jacktol[index]["duration"])))
            exact_pool = exact_pool[:25]
        indices = set(exact_pool)
        exact_indices = set(indices)
        if len(tokens) >= 4:
            fuzzy_pool: set[int] = set()
            grams = ngrams(tokens)
            for gram in grams:
                fuzzy_pool.update(gram_index.get(gram, ()))
            if len(fuzzy_pool) <= 500:
                for index in fuzzy_pool:
                    if index in indices:
                        continue
                    target = jacktol[index]
                    target_duration = float(target["duration"])
                    tolerance = max(0.75, 0.20 * max(duration, target_duration))
                    if abs(duration - target_duration) <= tolerance and token_jaccard(tokens, target["_tokens"]) >= fuzzy_threshold:
                        indices.add(index)
        if exact_indices:
            exact_rows += 1
        if indices - exact_indices:
            fuzzy_rows += 1
        for index in sorted(indices):
            target = jacktol[index]
            target_duration = float(target["duration"])
            tolerance = max(0.75, 0.20 * max(duration, target_duration))
            if abs(duration - target_duration) > tolerance:
                continue
            candidates.append(
                {
                    "source_role": role,
                    "source_id": row.get("id"),
                    "source_audio_filepath": row["audio_filepath"],
                    "source_offset": float(row.get("offset", 0.0)),
                    "source_duration": duration,
                    "source_text": row.get("text", ""),
                    "jacktol_id": target["id"],
                    "jacktol_split": target["_split"],
                    "jacktol_audio_filepath": target["audio_filepath"],
                    "jacktol_duration": target_duration,
                    "jacktol_text": target.get("text", ""),
                    "transcript_match": "exact" if index in exact_indices else "fuzzy",
                    "token_jaccard": token_jaccard(tokens, target["_tokens"]),
                }
            )
    return candidates, {
        "manifest": str(manifest),
        "manifest_sha256": sha256_path(manifest),
        "records": total,
        "speech_hours": hours,
        "rows_with_exact_candidates": exact_rows,
        "rows_with_fuzzy_candidates": fuzzy_rows,
        "candidate_pairs": len(candidates),
    }


def verify_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for number, candidate in enumerate(candidates, 1):
        source = load_audio(
            candidate["source_audio_filepath"],
            candidate["source_offset"],
            candidate["source_duration"],
        )
        target = load_audio(candidate["jacktol_audio_filepath"], 0.0, candidate["jacktol_duration"])
        similarity = acoustic_similarity(source, target)
        candidate.update(similarity)
        candidate["confirmed"] = bool(
            similarity["overlap_seconds"] >= 0.5
            and (
                (
                    similarity["envelope_correlation"] >= 0.93
                    and similarity["waveform_correlation"] >= 0.55
                )
                or (
                    candidate["transcript_match"] == "exact"
                    and similarity["envelope_correlation"] >= 0.975
                )
            )
        )
        verified.append(candidate)
        if number % 100 == 0:
            print(f"verified_candidates={number}/{len(candidates)}", flush=True)
    return verified


def write_disjoint(
    source: Path,
    destination: Path,
    excluded_recordings: set[str],
    excluded_ids: set[str],
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.incomplete.{os.getpid()}")
    kept_records = excluded_records = 0
    kept_hours = excluded_hours = 0.0
    with temporary.open("w", encoding="utf-8") as output:
        for row in iter_jsonl(source):
            duration = float(row.get("duration", 0.0))
            excluded = row["audio_filepath"] in excluded_recordings or str(row.get("id")) in excluded_ids
            if excluded:
                excluded_records += 1
                excluded_hours += duration / 3600
            else:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_records += 1
                kept_hours += duration / 3600
    temporary.replace(destination)
    return {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "path": str(destination),
        "sha256": sha256_path(destination),
        "kept_records": kept_records,
        "kept_speech_hours": kept_hours,
        "excluded_records": excluded_records,
        "excluded_speech_hours": excluded_hours,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--jacktol-release", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.75)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    jacktol, exact, gram_index = collect_jacktol(args.jacktol_release)
    train_candidates, train_source = candidate_pairs(
        args.training_manifest, "training", jacktol, exact, gram_index, args.fuzzy_threshold
    )
    validation_candidates, validation_source = candidate_pairs(
        args.validation_manifest, "validation", jacktol, exact, gram_index, args.fuzzy_threshold
    )
    verified = verify_candidates(train_candidates + validation_candidates)
    confirmed = [candidate for candidate in verified if candidate["confirmed"]]

    excluded_train_recordings = {
        candidate["source_audio_filepath"]
        for candidate in confirmed
        if candidate["source_role"] == "training"
    }
    excluded_train_ids = {
        str(candidate["source_id"])
        for candidate in confirmed
        if candidate["source_role"] == "training"
    }
    excluded_validation_recordings = {
        candidate["source_audio_filepath"]
        for candidate in confirmed
        if candidate["source_role"] == "validation"
    }
    excluded_validation_ids = {
        str(candidate["source_id"])
        for candidate in confirmed
        if candidate["source_role"] == "validation"
    }

    train_report = write_disjoint(
        args.training_manifest,
        args.output_root / "train.jsonl",
        excluded_train_recordings,
        excluded_train_ids,
    )
    validation_report = write_disjoint(
        args.validation_manifest,
        args.output_root / "validation.jsonl",
        excluded_validation_recordings,
        excluded_validation_ids,
    )
    matches_path = args.output_root / "confirmed_overlap.jsonl"
    with matches_path.open("w", encoding="utf-8") as output:
        for candidate in confirmed:
            output.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    by_split = Counter(candidate["jacktol_split"] for candidate in confirmed)
    by_role = Counter(candidate["source_role"] for candidate in confirmed)
    report = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "candidate_generation": "exact normalized transcript or >=0.75 token-Jaccard with shared 4-gram and duration tolerance",
            "confirmation": "acoustic envelope/waveform correlation",
            "quarantine": "remove every training segment from any source recording with a confirmed match",
            "test_use": "jacktol test is never used for checkpoint selection",
        },
        "jacktol_release": str(args.jacktol_release),
        "jacktol_dataset_report_sha256": sha256_path(args.jacktol_release / "dataset_report.json"),
        "source_audit": {"training": train_source, "validation": validation_source},
        "candidate_pairs": len(verified),
        "confirmed_pairs": len(confirmed),
        "confirmed_by_jacktol_split": dict(sorted(by_split.items())),
        "confirmed_by_source_role": dict(sorted(by_role.items())),
        "quarantined_training_source_recordings": len(excluded_train_recordings),
        "quarantined_validation_source_recordings": len(excluded_validation_recordings),
        "confirmed_overlap_manifest": str(matches_path),
        "confirmed_overlap_sha256": sha256_path(matches_path),
        "training_release": train_report,
        "validation_release": validation_report,
    }
    report_path = args.output_root / "overlap_audit.json"
    temporary = report_path.with_name(f"{report_path.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
