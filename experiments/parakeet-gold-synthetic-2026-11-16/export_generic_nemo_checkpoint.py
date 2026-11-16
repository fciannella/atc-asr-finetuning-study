#!/usr/bin/env python3

"""Export a trusted Lightning checkpoint using a compatible NeMo ASR archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    for path in (args.checkpoint, args.reference_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    model = ASRModel.restore_from(str(args.reference_model), map_location="cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.save_to(str(args.output_model))

    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "reference_model": str(args.reference_model),
        "reference_model_sha256": sha256(args.reference_model),
        "output_model": str(args.output_model),
        "output_model_sha256": sha256(args.output_model),
        "model_class": type(model).__name__,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
