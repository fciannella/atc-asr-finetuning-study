# Executed ATCO2 source-audit scripts

These are byte-identical copies of the scripts executed for the [November 6 audit](../../docs/atco2-training-overlap-audit-2026-11-06.md). Their SHA-256 hashes are recorded in the aggregate audit JSON. They are a frozen execution record, not a standalone public benchmark: licensed evaluation audio, manifests and predictions are intentionally absent.

Run only in the restricted experiment workspace, with NumPy, SoundFile and PyArrow installed. Copy these scripts into the audit working directory. The scripts expect:

- `download-receipts.json` and pinned Parquet assets under `public-data/<owner>__<dataset>/`, retaining each asset's relative path. Expected revisions and file hashes are in the report's `audio_audit.download_receipts` field.
- A sibling `community-atc-eval-2026-11-06/` directory containing the frozen `test.jsonl`, `aggregate.json`, original predictions, `score_wer.py` and `evaluate_community_atc.py`. Test rows carry IDs, reference text, source and cropped audio paths, offsets/durations, and the existing Jacktol exclusion flag.
- The historical P2 prediction artifact at the relative experiment path recorded in `rescore_masks.py`. The scripts enforce the frozen run's 1,908-row contract and reproduce its original error totals.

Execute `audit_public_audio.py` first, then `rescore_masks.py`. The first writes aggregate `audio-audit.json` and **private** row-level `private-match-evidence.jsonl`. The second independently verifies matched WAV samples, writes **private** `private-exclusion-masks.json`, and produces aggregate `common-mask-scores.json`.

Keep the row-level evidence and masks restricted. The published report combines aggregate outputs and source metadata only. All candidate discovery, sample-rate restrictions and scope limitations are documented in the audit; these scripts do not certify complete training ancestry.
