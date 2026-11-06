# ATCO2 training-source overlap audit

Completed **November 6, 2026**. **Exact audio overlap is now confirmed with the published ATCO2 training sources used by RASR and the jlvdoorn project.** This closes the direct ATCO2 source comparison; it does not certify their full inherited training histories.

## Confirmed audio findings

| Published source split | Rows inspected | Identical complete recordings also in our test | Our test segments exactly contained in those recordings | Still present in the old Jacktol-disjoint subset |
|---|---:|---:|---:|---:|
| `jlvdoorn/atco2-asr` train | 446 | 316 | **498** | **347** |
| `jlvdoorn/atco2-asr` validation | 113 | 83 | **140** | **79** |
| `jlvdoorn/atco2-asr-atcosim` train | 8,092 total, including 446 ATCO2 rows | Same 316 matched recordings | Same **498** | Same **347** |
| `jlvdoorn/atco2-asr-atcosim` validation | 2,026 total, including 113 ATCO2 rows | Same 83 matched recordings | Same **140** | Same **79** |

**Do not add the two datasets' counts.** Their matched ATCO2 content is duplicated. The train and validation matches are disjoint: **638 unique test segments across 399 complete recordings**, independently verified. The 498 training-source matches cover **0.559821 hours**; validation matches cover **0.173898 hours**.

This is stronger than a filename or transcript match. We confirmed identical decoded 16-bit PCM for each complete recording and then compared the exact sample range of each test crop. Every reported segment matched. A second check used Python's WAV reader instead of the original SoundFile path and reproduced all 638 segment matches and 399 recording identities.

## From dataset overlap to checkpoint exposure

| Checkpoint | Available training evidence | Interpretation |
|---|---|---|
| RASR v1 | Pinned model card and published recipe name both public ATCO2 train datasets. Embedded checkpoint configuration specifies 0.1–18-second training clips. | Of the 498 exact source matches, **476 test segments** belong to public clips eligible under the documented duration/text filter. The other 22 belong to longer public clips. The 476 count is recipe-eligible overlap, not an attestation of every example actually consumed by the optimizer. |
| RASR v2 | Pinned recipe continues from v1 and again includes both ATCO2 train datasets; its embedded configuration also uses 0.1–18 seconds. | The same **476 recipe-eligible segments**, plus inherited v1 exposure. The 28.8-hour US extension does not erase that history. |
| jlvdoorn Whisper large v3 ATCO2 | Model/project association and the project's ATCO2 training script load the train split and use its validation split. | **498 source-overlap segments**; **496** lie wholly within the first 30 seconds of their public training clips, consistent with the usual Whisper input window. Two do not lie wholly within that window. Exact model-job manifests and preprocessing provenance are not published. |

The RASR manifest builder also applies the duration filter to its validation inputs: **126** of the 140 validation-source matches are eligible under that recipe. All 140 fit within the Whisper 30-second window. **Validation exposure is not gradient training**, but reuse for checkpoint selection/evaluation still prevents treating those examples as an untouched independent benchmark.

Sources: [RASR v1 recipe](https://github.com/twangodev/rasr/blob/ddff2ba501ab78f2820d159325ca3ff26699a94e/configs/train/rtx6kpro/v1-mixed.yaml), [manifest preparation](https://github.com/twangodev/rasr/blob/ddff2ba501ab78f2820d159325ca3ff26699a94e/src/rasr/train/manifest.py), [duration defaults](https://github.com/twangodev/rasr/blob/ddff2ba501ab78f2820d159325ca3ff26699a94e/src/rasr/train/config.py), [pinned v2 recipe](https://huggingface.co/twangodev/rasr-parakeet-v2/blob/cb6ed4e5e68db54b4a8787dcea9706c2160a1354/training_recipe.yaml), and [WhisperATC training script](https://github.com/jlvdoorn/WhisperATC/blob/cd0b71e5e0da9241d589d4bdbfbf16440549e577/Fine-Tune/FineTuning-ATCO2.py). These are source artifacts, not a reconstruction of private training-job logs. RASR's offline dataset-loading fallback can also read all cached Parquet files without enforcing the requested split; no evidence establishes that this fallback was used in these training jobs, so we do not count validation audio as confirmed training exposure.

## A common comparison after removing audited sources

The previous Jacktol-only mask is insufficient for these models. We therefore formed two fixed, common masks and rescored the existing predictions for **all 12 checkpoints**:

- **Jacktol + audited ATCO2 training sources excluded:** remove the union of 560 test utterances; retain **1,348 utterances, 14,874 words, 1.362562 hours**.
- **Jacktol + audited ATCO2 training and validation sources excluded:** remove the union of 639; retain **1,269 utterances, 13,851 words, 1.262691 hours**. This more conservative view is shown below.

The 639 exclusions are the union of 213 Jacktol-linked utterances and 638 public ATCO2 train/validation matches; 212 occur in both groups. The mask excludes all source matches, including the small number outside recipe windows, rather than constructing a different test for each model.

| Checkpoint | Original full-test WER | Existing Jacktol-excluded WER | New audited-source-excluded WER |
|---|---:|---:|---:|
| RASR Parakeet v1 | 12.71% | 13.36% | 15.68% |
| RASR Parakeet v2 | 14.88% | 15.28% | 17.33% |
| qenneth Parakeet | 15.91% | 16.65% | 18.18% |
| jlvdoorn Whisper large v3 | 16.20% | 16.91% | 19.82% |
| Jacktol Whisper large v3 | 16.30% | 17.57% | 20.29% |
| Our Jacktol-trained P2 | 18.38% | 18.90% | 20.11% |
| Our P4 Parakeet | 20.00% | 19.89% | 20.16% |
| Our N4 Nemotron | 22.46% | 22.33% | 22.32% |
| YouTube ATC FastConformer | 26.75% | 27.10% | 28.50% |
| STEAK synthetic Whisper | 29.01% | 29.07% | 30.50% |
| US ATC Whisper Turbo v3 | 66.70% | 66.91% | 69.25% |
| XLS-R ATCOSIM | 88.64% | 88.91% | 90.39% |

**Compare models within a column.** Original scores remain unchanged and visible. The new column has a smaller denominator and is an additional post-hoc diagnostic, not a replacement blind test or proof of general training independence. Changes between columns combine source exclusion with changes in test difficulty; they do not measure memorization causally. No new inference, fine-tuning, decoding change or checkpoint selection was performed. All original full-test error totals reproduced before rescoring. The JSON includes the training-only view, substitutions/deletions/insertions and recording-clustered 95% bootstrap intervals.

## Inherited data and remaining gaps

| Branch | What was traced | What remains unresolved |
|---|---|---|
| RASR v1 → RadioTalk synthetic speech | Noisy dataset → clean Tada speech → Qwen3 transcript dataset and LibriTTS-R reference voices. Channel simulation is described as calibrated to ATCO2's SNR distribution. | No immutable manifest of the exact selected 200k rows. Corpus-wide synthetic-audio, text-prompt and noise-source overlap was not established. |
| RASR v2 → TartanAviation | Recipe selects high-agreement pseudo-labels and excludes held-out dates; dataset metadata describes KAGC/KBTP US recordings. | The exact 35,240-row local training manifest and exclusion dates are not in the reviewed release. We did not perform a corpus-wide waveform scan of the 398-hour public utterance corpus. Different geography alone is not an acoustic proof of disjointness. |
| RASR v2 → label-generating models | Parakeet v2, Canary-Qwen 2.5B and jlvdoorn ATCO2 Whisper are named; the last has the direct ATCO2 source overlap above. | Exact teacher checkpoint revisions and all teacher-training examples are not pinned. Teacher ancestry is indirect exposure and must not be equated with direct audio training. |
| RASR base → Parakeet v3 | Public card describes Granary and NeMo ASR Set 3.0, including inherited initialization. | Exact consumed-example history, in-house data and the actual base revision used by the author are not fully available. |
| jlvdoorn base → Whisper large v3 | Public card describes weakly labeled and pseudo-labeled pretraining, including a Whisper v2 teacher. | No complete public training-example manifest; exact full-ancestry overlap cannot be certified. |
| ATCOSIM component of the combined corpus | All 9,559 32-kHz rows were decoded and counted separately from the ATCO2 component. | They were not included in the exact 16-kHz PCM matching pass. No “ATCOSIM has zero overlap” conclusion is claimed. |

**Synthetic-lineage correction:** the RASR model cards attribute dialogue generation to Llama 3.2. The linked RadioTalk dataset chain instead identifies `Qwen/Qwen3-32B-NVFP4` as the transcript generator; its Tada TTS component is Llama-based. The transcript dataset notes an August 2026 rename to make the generator explicit. A metadata-only check of all 5,000 rows in pinned noisy shard `shard-000000.parquet` also found `Qwen/Qwen3-32B-NVFP4` and `HumeAI/tada-3b-ml` in every row. This verifies that shard, not the exact 200k rows consumed by the ASR job. We record this conflict instead of treating the earlier model-card attribution as verified. This does not affect the exact ATCO2 audio matches or measured WER.

The review also found explicit dataset/base metadata links for **eight inventory repositories** (including RASR): the `amidstdebug` CTranslate2 and `sfabriece` MLX descendants of jlvdoorn, two `tclin` models linked to the combined corpus, and two `youngsangroh` models linked to that corpus. Their declared relationships are recorded in JSON. This identifies inherited or source-level risk; it is not a per-checkpoint training attestation. The wider jlvdoorn ATCO2 model family also warrants this audit when evaluated; naming alone is not used to assign exact counts.

## Reproducibility and scope

- Frozen private test manifest SHA-256: `4af85f51d32051ee84c841c92c8e5e1eb69ee07201149197f905a7edc9c13082`.
- Public ATCO2 revision: [`3b96f5791582cf6f6f348ca46fd502a981918d85`](https://huggingface.co/datasets/jlvdoorn/atco2-asr/tree/3b96f5791582cf6f6f348ca46fd502a981918d85).
- Combined corpus revision: [`21820a50f2f4d0a8f90838feb4efe88c3db31f95`](https://huggingface.co/datasets/jlvdoorn/atco2-asr-atcosim/tree/21820a50f2f4d0a8f90838feb4efe88c3db31f95).
- Their current commits date to June/July 2023, before the reviewed model releases. This strengthens the source match but does not supply a missing author training manifest.
- Nine Parquet assets were downloaded, hashed and fully read: 10,677 rows across the two overlapping repositories. Of those, 1,118 are 16-kHz ATCO2 rows, including duplicates between repositories; 9,559 are 32-kHz ATCOSIM rows.
- Exact PCM comparisons do not establish absence of arbitrary resampling, recropping, time shifts or other transforms among unmatched sources. Negative findings are limited to the stated method.
- [Executed audit and rescoring scripts](../scripts/atco2-source-audit-2026-11-06/README.md) are preserved byte-for-byte; their hashes appear in the JSON.
- CPU audit jobs **38098198** and **38098238** completed successfully. The second includes preprocessing checks, independent WAV verification and all 12 rescored checkpoints.
- [Aggregate audit JSON and source pins](../reports/atco2-training-overlap-audit-2026-11-06.json); [common-mask WER CSV](../reports/atco2-training-overlap-audit-2026-11-06.csv). Row IDs, audio, transcripts, per-utterance matches and masks remain in restricted storage; only their hashes and aggregate results are published.

**Status:** direct public ATCO2 audio overlap is established and a common exclusion-based comparison is available. Full-ancestry clearance remains open until exact training manifests, preprocessing/version details and inaccessible base/teacher histories can be reconciled. The [tracker](evaluation-tracker.md) separates these completed findings from the remaining work.
