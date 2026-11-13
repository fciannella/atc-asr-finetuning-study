---
language:
- en
pretty_name: ATCO2 — Known-overlap-removed evaluation pool (90.64 minutes)
size_categories:
- 1K<n<10K
task_categories:
- automatic-speech-recognition
configs:
- config_name: default
  default: true
  drop_labels: true
  data_files:
  - split: test
    path: data/test/**
  - split: extension
    path: data/extension/**
- config_name: combined
  drop_labels: true
  data_files:
  - split: test
    path:
    - data/test/**
    - data/extension/**
---

# ATCO2 evaluation pool with known overlap removed

Version **`atco2-known-overlap-removed-v2-2026-11-13`** contains **1,503 human-accepted English ATC clips / 90.64 minutes**. This replaces the earlier 1,908-clip two-hour package. All identified overlap from the completed public ATCO2 and Jacktol audits has been excluded.

**Private review copy.** Public distribution, licensing and release approvals remain pending. No open-data license is asserted. The source remains ELDA/ELRA ATCO2 human-annotated v1.1 (ELRA-S0484; ISLRN 589-403-577-685-7).

## Choose the appropriate evaluation view

| Configuration / split | Clips | Minutes | Provenance and eligibility |
|---|---:|---:|---|
| `default` / `test` | 1,269 | 75.7615 | Original community-test subset after removing known public train/validation and Jacktol source-recording overlap. Use this view for the existing P4/N4 comparison. |
| `default` / `extension` | 234 | 14.8796 | Surviving former Gold-training clips. **P4/N4 already trained on these clips. They are not held-out tests for those checkpoints.** |
| `combined` / `test` | 1,503 | 90.6411 | Both groups together. Only eligible for checkpoints whose complete training and selection lineage excludes the extension and its linked source recordings, including any Silver exposure. |

The two default splits are disjoint. The combined configuration references the same files; it does not duplicate audio in the repository. This is **approximately 1 hour 30 minutes 38 seconds**, not two hours. Another 29.36 eligible minutes would be needed to reach two hours for a future eligible combined evaluation.

## Load and inspect

```python
from datasets import load_dataset

# Existing P4/N4 comparison: 1,269 clips.
benchmark = load_dataset("nvidia/atco2-eval", "default", split="test")

# Future eligible model lineages only: 1,503 clips.
full_pool = load_dataset("nvidia/atco2-eval", "combined", split="test")
```

Pin the dataset revision when reporting results. Access requires authorization to this private repository. In the Dataset Viewer, choose the configuration/split and play the `audio` column beside `text`; viewer availability depends on private-dataset support.

Audio is 16-kHz, mono, signed 16-bit PCM WAV. Crops preserve the source samples used by the audits; reference text is copied unchanged from the frozen study manifests. No synthetic audio, generated transcripts or additional third-party material was introduced by this packaging step. The study's existing reference text is preserved; this does not assert that it is byte-identical to every upstream ELDA annotation representation.

Additional columns are `id`, `duration` (actual seconds), `airport`, `recorded_date`, `source_record_id`, `source_offset_seconds`, `origin_partition`, `previously_used_for_p4_n4_training`, and `pcm_sha256`. AudioFolder uses `file_name` in each metadata CSV to associate the WAV with the row. Source-record IDs support grouped scoring and training exclusions. Internal filesystem paths are not included.

## Exclusion contract and limits

- Original community test: exclude **639 of 1,908 clips**, the union of 213 Jacktol source-recording links and 638 exact public ATCO2 train/validation matches (212 clips occur in both groups).
- Former Gold training: exclude **159 of 393 clips**: 158 exact public-audio duplicates and one independently corroborated 4.14-second acoustic match. Conservative source-recording checks add no further clips.
- Public snapshots audited: `jlvdoorn/atco2-asr` at `3b96f5791582cf6f6f348ca46fd502a981918d85`; `jlvdoorn/atco2-asr-atcosim` at `21820a50f2f4d0a8f90838feb4efe88c3db31f95`; and `jacktol/ATC-ASR-Dataset` at `075e736bf8aed80579d829092f74355486b10bc7`.
- **Known overlap removed** describes these completed checks. It does not certify all public datasets, inaccessible base/teacher pretraining, arbitrary audio transformations, or every possible partial overlap. The 32-kHz ATCOSIM component was decoded and counted but not acoustically compared by the exact 16-kHz audit. The original-test and Gold audits have different candidate coverage, documented in the reports.
- Existing benchmarks have been inspected repeatedly. This is not a new blind test. Evaluate frozen models; exclude this version's recordings from subsequent training, tokenizer/LM fitting, checkpoint selection and decoder tuning when claiming held-out performance. Moving Gold into a package does not undo historical model exposure.

Report corpus WER with the model/checkpoint, configuration, split, dataset revision, decoding settings and text normalizer. Use source-recording groups for uncertainty estimates. Earlier 1,908-clip scores are not scores on this release. No scores for the combined 1,503-clip view are asserted here.

## Reproducibility and provenance

`release.json` records frozen manifest/mask hashes, exact sampled durations, the builder hash and validation results. `SHA256SUMS` seals every package file other than itself. The package has unique IDs and unique complete PCM hashes; its two default splits share no source-record IDs or airport/date groups. Each output WAV was independently decoded and checked against source PCM; all 1,269 original-test crops also match the evaluated audio.

- [Original test overlap audit](https://github.com/fciannella/atc-asr-finetuning-study/blob/main/docs/atco2-training-overlap-audit-2026-11-06.md)
- [Gold overlap audit and duration budget](https://github.com/fciannella/atc-asr-finetuning-study/blob/main/docs/gold-public-overlap-audit-2026-11-13.md)
- [ATCO2 source](https://atco2.org/data)

Redistribution and licensing follow the applicable ELDA/ELRA agreement and release approvals. Code licensing does not grant rights to recordings or annotations. The repository retains its existing private visibility.
