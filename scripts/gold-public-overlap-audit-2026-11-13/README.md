# Gold training audio / public corpus audit

Frozen execution scripts for the November 13, 2026 audit of the 393-clip Gold training manifest. These scripts require the restricted OCI experiment workspace; they do not include licensed audio, transcripts or row-level evidence.

Place the four Python files under `experiments/gold-public-overlap-audit-2026-11-13/` in the existing `atc-ablation-v1` workspace. Run `audit_gold.py` and `audit_recording_links.py` with NumPy, SciPy, SoundFile and PyArrow installed. The scripts expect:

- The frozen Gold training manifest under `experiments/atco2-gold-community-ablation-v1/inputs/` and its source WAV recordings.
- The pinned public Parquet files and download receipts from `experiments/atco2-ancestry-audit-2026-11-06/`, plus restricted prior positive evidence.
- The original private test manifest under `experiments/community-atc-eval-2026-11-06/`, used only for two positive controls.
- The pinned Jacktol release under `evaluation/jacktol-atc-asr-075e736/release-v1/`.
- The prior restricted acoustic-link evidence under `datasets/fullhq-jacktol-disjoint-v1/confirmed_overlap.jsonl`.

The audit verifies public Parquet and Jacktol WAV file hashes. It searches every eligible 16-kHz mono public recording for each complete Gold crop, independently of names and transcripts. It also checks whether a complete public clip of at least 0.5 seconds is contained within a Gold crop. Every positive is checked with the independent Python WAV reader. Multiple repository copies are deduplicated when calculating the union; matching sample intervals are merged within each Gold clip.

The existing `audit_jacktol_overlap.py` helper adds a transcript/duration-candidate acoustic check, with token Jaccard threshold 0.65 and its original correlation criteria. Exact PCM findings and heuristic acoustic findings are reported separately. Neither method establishes full training-ancestry clearance. The 32-kHz ATCOSIM component is decoded and counted but excluded from this exact 16-kHz comparison.

`audit_recording_links.py` additionally searches each complete Gold source recording for each Jacktol clip of at least 0.5 seconds. It also checks prior acoustic source links by independently proving that the previously linked source and the Gold source have identical complete PCM. This supports conservative source-recording exclusion even when the particular Gold transmission has no direct clip match. Those counts are distinct from exact Gold-crop duplicates.

`audit-summary.json` and `recording-links-summary.json` contain aggregates and hashes. Keep all `private-*.jsonl` evidence in restricted storage, including `private-recording-evidence.jsonl`. The candidate evidence contains transcripts and paths. Nothing in this audit changes training manifests or evaluation assignments.

After both audits complete, `combine_audits.py` constructs a conservative recording-level exclusion union and the duration budget. It also reads the earlier `common-mask-scores.json` and a `gold-test-identity-check.json` containing the manifest-level ID, source-path, source-record-ID and airport/date intersection counts. Keep its `private-gold-exclusion-mask.json` restricted; publish only the aggregate report. This mask records findings and is not automatically applied to any dataset.
