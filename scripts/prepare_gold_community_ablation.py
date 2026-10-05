#!/usr/bin/env python3

"""Freeze the ATCO2 gold release split and the sequential 8-GPU ablation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


UNK_RE = re.compile(r"(?:\[unk\]|<unk>)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.incomplete.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.incomplete.{os.getpid()}")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def group_key(record: dict[str, Any]) -> tuple[str, str]:
    airport = str(record.get("airport") or "").strip()
    recorded_date = str(record.get("recorded_date") or "").strip()
    if not airport or not recorded_date:
        raise ValueError(f"Gold record lacks airport/date grouping fields: {record.get('id')}")
    return airport, recorded_date


def duration(records: Iterable[dict[str, Any]]) -> float:
    return sum(float(record["duration"]) for record in records)


def choose_groups_near_target(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    target_seconds: float,
    seed: int,
    excluded_complete_choice: bool = False,
) -> set[tuple[str, str]]:
    """Deterministic subset-sum at 0.1-second resolution."""

    ordered = sorted(groups, key=lambda key: stable_rank(seed, "|".join(key)))
    ticks = {key: max(1, round(duration(groups[key]) * 10)) for key in ordered}
    target = round(target_seconds * 10)
    upper = target + max(ticks.values(), default=0)
    states: dict[int, tuple[tuple[str, str], ...]] = {0: ()}
    for key in ordered:
        value = ticks[key]
        for existing, path in sorted(list(states.items()), reverse=True):
            candidate = existing + value
            if candidate <= upper and candidate not in states:
                states[candidate] = path + (key,)
    all_keys = set(ordered)
    choices = []
    for total, path in states.items():
        selected = set(path)
        if not selected:
            continue
        if excluded_complete_choice and selected == all_keys:
            continue
        choices.append((abs(total - target), stable_rank(seed, repr(path)), path))
    if not choices:
        raise RuntimeError("No valid group subset found")
    return set(min(choices)[2])


def split_gold(
    records: list[dict[str, Any]],
    test_target_seconds: float,
    development_target_seconds: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)

    total_seconds = duration(records)
    train_side_target = total_seconds - test_target_seconds
    if train_side_target <= development_target_seconds:
        raise ValueError("Gold pool is too small for the requested test and development targets")

    by_airport: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(dict)
    for key, values in groups.items():
        by_airport[key[0]][key] = values
    train_side_fraction = train_side_target / total_seconds
    train_side_groups: set[tuple[str, str]] = set()
    for airport, airport_groups in sorted(by_airport.items()):
        airport_target = sum(duration(value) for value in airport_groups.values()) * train_side_fraction
        chosen = choose_groups_near_target(
            airport_groups,
            airport_target,
            seed + int(stable_rank(seed, airport)[:8], 16),
            excluded_complete_choice=True,
        )
        train_side_groups.update(chosen)

    # Keep one substantial group per represented airport out of development so
    # each airport selected for adaptation remains present in gold training.
    selected_by_airport: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in train_side_groups:
        selected_by_airport[key[0]].append(key)
    protected_train_groups = {
        max(keys, key=lambda key: (duration(groups[key]), key))
        for keys in selected_by_airport.values()
    }
    development_candidates = {
        key: groups[key]
        for key in train_side_groups
        if key not in protected_train_groups
    }
    development_groups = choose_groups_near_target(
        development_candidates,
        development_target_seconds,
        seed + 991,
    )
    training_groups = train_side_groups - development_groups
    test_groups = set(groups) - train_side_groups

    def materialize(selected: set[tuple[str, str]], label: str) -> list[dict[str, Any]]:
        output = []
        for record in records:
            key = group_key(record)
            if key in selected:
                item = dict(record)
                item["gold_release_partition"] = label
                item["gold_group_key"] = f"{key[0]}::{key[1]}"
                output.append(item)
        return output

    train = materialize(training_groups, "train")
    development = materialize(development_groups, "development")
    test = materialize(test_groups, "community_test")
    report = {
        "split_seed": seed,
        "grouping_key": ["airport", "recorded_date"],
        "targets_seconds": {
            "community_test": test_target_seconds,
            "development": development_target_seconds,
            "train_side_total": train_side_target,
        },
        "partitions": {
            "train": summarize_records(train),
            "development": summarize_records(development),
            "community_test": summarize_records(test),
        },
    }
    return train, development, test, report


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    group_keys = {group_key(record) for record in records}
    return {
        "records": len(records),
        "seconds": duration(records),
        "hours": duration(records) / 3600,
        "groups": len(group_keys),
        "airports": dict(sorted(Counter(str(record["airport"]) for record in records).items())),
        "source_recordings": len({str(record.get("source_record_id")) for record in records}),
    }


def identity_sets(records: list[dict[str, Any]]) -> dict[str, set[Any]]:
    return {
        "id": {str(record["id"]) for record in records},
        "source_record_id": {str(record.get("source_record_id")) for record in records},
        "audio_filepath": {str(record["audio_filepath"]) for record in records},
        "airport_date": {group_key(record) for record in records},
    }


def assert_disjoint(partitions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    names = list(partitions)
    sets = {name: identity_sets(partitions[name]) for name in names}
    pairwise: dict[str, Any] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps = {
                field: len(sets[left][field] & sets[right][field])
                for field in sets[left]
            }
            if any(overlaps.values()):
                raise RuntimeError(f"Partition overlap {left} vs {right}: {overlaps}")
            pairwise[f"{left}_vs_{right}"] = overlaps
    return pairwise


def validate_gold(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(record.get("id") or "") for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate gold record IDs")
    missing_audio = []
    unknown_tokens = []
    empty_text = []
    acceptance = Counter()
    languages = Counter()
    for record in records:
        text = str(record.get("text") or "").strip()
        if not text:
            empty_text.append(record.get("id"))
        if UNK_RE.search(text):
            unknown_tokens.append(record.get("id"))
        if not Path(str(record["audio_filepath"])).is_file():
            missing_audio.append(record.get("id"))
        acceptance[str(record.get("gold_acceptance") or "missing")] += 1
        languages[str(record.get("lang") or record.get("target_lang") or "missing")] += 1
    if missing_audio or unknown_tokens or empty_text:
        raise RuntimeError(
            f"Gold validation failed: missing_audio={len(missing_audio)} "
            f"unknown_tokens={len(unknown_tokens)} empty_text={len(empty_text)}"
        )
    return {
        **summarize_records(records),
        "duplicate_ids": 0,
        "missing_audio": 0,
        "unknown_sentinel_tokens": 0,
        "empty_text": 0,
        "acceptance": dict(sorted(acceptance.items())),
        "languages": dict(sorted(languages.items())),
    }


def manifest_contract(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), **summarize_records(records)}


def write_stage(
    root: Path,
    stage_id: str,
    sources: list[tuple[str, Path, float]],
    source_weights: dict[str, float],
    start_model: Path,
    seed: int,
) -> dict[str, Any]:
    stage_root = root / "stages" / stage_id
    input_cfg_path = stage_root / "train-input-cfg.json"
    input_cfg = [
        {
            "type": "nemo",
            "manifest_filepath": str(path),
            "weight": weight,
            "tags": {"domain": label, "gold_ablation_stage": stage_id},
        }
        for label, path, weight in sources
    ]
    atomic_json(input_cfg_path, input_cfg)
    config = {
        "stage_id": stage_id,
        "start_model": str(start_model),
        "train_input_cfg": str(input_cfg_path),
        "train_input_cfg_sha256": sha256(input_cfg_path),
        "source_weights": source_weights,
        "max_steps": 2000,
        "learning_rate": 1e-5,
        "min_lr": 1e-6,
        "warmup_steps": 100,
        "ema": False,
        "validation_interval": 100,
        "validation_batch_size": 16,
        "seed": seed,
        "precision": "32-true",
        "per_device_batch_size": 2,
        "accumulate_grad_batches": 1,
        "effective_batch_size": 16,
        "feature_normalization": "per_feature",
        "selected_model": str(stage_root / "artifacts/selected.nemo"),
    }
    atomic_json(stage_root / "stage-config.json", config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gold-development-source", type=Path, required=True)
    parser.add_argument("--gold-test-source", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--silver-model", type=Path, required=True)
    parser.add_argument("--silver-train", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, required=True)
    parser.add_argument("--jacktol-validation", type=Path, required=True)
    parser.add_argument("--jacktol-test", type=Path, required=True)
    parser.add_argument("--uwb-test", type=Path, required=True)
    parser.add_argument("--librispeech-test", type=Path, required=True)
    parser.add_argument("--test-hours", type=float, default=2.0)
    parser.add_argument("--development-minutes", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()

    required = [
        args.gold_development_source,
        args.gold_test_source,
        args.base_model,
        args.silver_model,
        args.silver_train,
        args.general_train,
        args.jacktol_validation,
        args.jacktol_test,
        args.uwb_test,
        args.librispeech_test,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.experiment_root.exists() and any(args.experiment_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty experiment root: {args.experiment_root}")

    gold_records = read_jsonl(args.gold_development_source) + read_jsonl(args.gold_test_source)
    validation = validate_gold(gold_records)
    train, development, community_test, split_report = split_gold(
        gold_records,
        args.test_hours * 3600,
        args.development_minutes * 60,
        args.seed,
    )
    partitions = {"train": train, "development": development, "community_test": community_test}
    disjointness = assert_disjoint(partitions)

    inputs = args.experiment_root / "inputs"
    manifest_paths = {
        "gold_train": inputs / "gold-train.jsonl",
        "gold_development": inputs / "gold-development.jsonl",
        "gold_community_test": inputs / "gold-community-test.jsonl",
    }
    write_jsonl(manifest_paths["gold_train"], train)
    write_jsonl(manifest_paths["gold_development"], development)
    write_jsonl(manifest_paths["gold_community_test"], community_test)

    external_sources = {
        "silver_train": (args.silver_train, inputs / "silver-train.jsonl"),
        "general_train": (args.general_train, inputs / "general-train.jsonl"),
        "jacktol_validation": (args.jacktol_validation, inputs / "jacktol-validation.jsonl"),
        "jacktol_test": (args.jacktol_test, inputs / "jacktol-test.jsonl"),
        "uwb_test": (args.uwb_test, inputs / "uwb-test.jsonl"),
        "librispeech_test": (args.librispeech_test, inputs / "librispeech-test.jsonl"),
    }
    external_contract: dict[str, Any] = {}
    for label, (source, destination) in external_sources.items():
        atomic_copy(source, destination)
        external_contract[label] = {
            "source": str(source),
            "path": str(destination),
            "sha256": sha256(destination),
        }

    gold_path = manifest_paths["gold_train"]
    silver_path = external_sources["silver_train"][1]
    general_path = external_sources["general_train"][1]
    stage_specs = [
        ("g1_gold_only_s1234", [("gold_atco2", gold_path, 1.0)], {"gold_atco2": 1.0, "general_english": 0.0}, args.base_model, 1234),
        ("g2_gold_replay_s1234", [("gold_atco2", gold_path, 0.85), ("general_english", general_path, 0.15)], {"gold_atco2": 0.85, "general_english": 0.15}, args.base_model, 1234),
        ("g3_silver_gold_refine_s1234", [("gold_atco2", gold_path, 0.85), ("general_english", general_path, 0.15)], {"gold_atco2": 0.85, "general_english": 0.15}, args.silver_model, 1234),
        ("g4_silver_gold_joint_s1234", [("silver_atco2", silver_path, 0.70), ("gold_atco2", gold_path, 0.15), ("general_english", general_path, 0.15)], {"silver_atco2": 0.70, "gold_atco2": 0.15, "general_english": 0.15}, args.silver_model, 1234),
        ("g3_silver_gold_refine_s42", [("gold_atco2", gold_path, 0.85), ("general_english", general_path, 0.15)], {"gold_atco2": 0.85, "general_english": 0.15}, args.silver_model, 42),
        ("g4_silver_gold_joint_s42", [("silver_atco2", silver_path, 0.70), ("gold_atco2", gold_path, 0.15), ("general_english", general_path, 0.15)], {"silver_atco2": 0.70, "gold_atco2": 0.15, "general_english": 0.15}, args.silver_model, 42),
    ]
    stages = {
        stage_id: write_stage(
            args.experiment_root, stage_id, sources, source_weights, start_model, stage_seed
        )
        for stage_id, sources, source_weights, start_model, stage_seed in stage_specs
    }

    dataset_report = {
        "status": "frozen_and_audited",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "community_test": "Never used for training, validation, checkpoint selection, or hyperparameter selection.",
            "development": "Gold development is the only checkpoint-selection set for the six new runs.",
            "group_isolation": "All records from an airport-day remain in exactly one partition.",
            "jacktol": "Secondary benchmark only; headline community-test claims require a separate model-specific Jacktol overlap audit.",
        },
        "full_gold_validation": validation,
        "source_manifests": {
            "historical_development": {"path": str(args.gold_development_source), "sha256": sha256(args.gold_development_source)},
            "historical_test": {"path": str(args.gold_test_source), "sha256": sha256(args.gold_test_source)},
        },
        "split": split_report,
        "pairwise_disjointness": disjointness,
        "manifests": {
            name: manifest_contract(manifest_paths[name], records)
            for name, records in {
                "gold_train": train,
                "gold_development": development,
                "gold_community_test": community_test,
            }.items()
        },
        "external_snapshots": external_contract,
    }
    atomic_json(inputs / "dataset-report.json", dataset_report)

    contract = {
        "status": "preflight_complete_test_locked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_root.name,
        "objective": "Measure whether a small, strictly held-out human-gold adaptation set improves the silver ATCO2 Parakeet TDT model while retaining general English.",
        "model_family": "nvidia/parakeet-tdt-0.6b-v3",
        "base_model": {"path": str(args.base_model), "sha256": sha256(args.base_model)},
        "silver_control_model": {"path": str(args.silver_model), "sha256": sha256(args.silver_model)},
        "selection_manifest": dataset_report["manifests"]["gold_development"],
        "locked_primary_test_manifest": dataset_report["manifests"]["gold_community_test"],
        "training_contract": {
            "node_gpus": 8,
            "effective_global_batch_size": 16,
            "max_steps": 2000,
            "peak_learning_rate": 1e-5,
            "schedule": "100-step warmup then cosine decay to 1e-6",
            "precision": "32-true",
            "feature_normalization": "per_feature",
            "validation_interval_steps": 100,
            "synthetic_data": False,
        },
        "execution_order": [stage_id for stage_id, *_ in stage_specs],
        "stages": stages,
        "final_evaluation": ["gold_community_test", "jacktol_test", "uwb_test", "librispeech_test_clean"],
        "release_gate": "Audio publication requires ELDA/ELRA approval; this contract freezes a technical release candidate only.",
    }
    atomic_json(args.experiment_root / "contract.json", contract)
    print(json.dumps({
        "experiment_root": str(args.experiment_root),
        "status": contract["status"],
        "gold_partitions": split_report["partitions"],
        "execution_order": contract["execution_order"],
        "locked_test_sha256": contract["locked_primary_test_manifest"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
