# Reproducibility scripts

The selected scripts cover:

- Gold release construction and audit;
- Jacktol acoustic-overlap detection;
- experiment-contract freezing;
- normalized WER and paired-bootstrap comparison;
- Gold ablation preparation;
- leakage-safe n-gram corpus preparation;
- development-only n-gram sweep selection.

They are snapshots of the utilities used during the study. Inputs are not bundled. Run each script with `--help` and supply authorized local paths explicitly.

`evaluate_community_atc.py` runs pinned NeMo, Whisper/LoRA and CTC checkpoints and computes the primary lexical WER with source-recording bootstrap intervals. It expects a prepared run root containing `models.json`, `data-contract.json`, `test.jsonl`, `dev.jsonl`, and per-model weight receipts under `models/<slug>/download-receipt.json`. Weight files, audio crops and manifests must be supplied in authorized storage. The NeMo evaluator location is container-specific; see the [benchmark](../docs/community-atc-benchmark-2026-11-06.md) for the exact execution environment and limitations.
