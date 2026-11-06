# ATCO2 comparison after removing known overlap

Updated **November 6, 2026**. The current results below exclude all overlap identified by the completed Jacktol and public ATCO2 audio audits.

All results below use the **same 1,269 utterances, 13,851 reference words, 768 source recordings and 1.262691 hours (75.76 minutes)**. This subset excludes the union of known Jacktol recording links and exact matches to the audited public ATCO2 training and validation sources.

**Scope: known overlap removed.** No retained utterance has an overlap identified by the completed audits. This is not a guarantee of zero contamination across every model’s full training history: unscanned corpora and incomplete base/teacher manifests remain. The subset was defined after inspecting the original test, so it is not a new blind test.

## Measured results

Lower WER is better. All community checkpoints, P2 and P4 were evaluated offline; N4 uses its established native-streaming condition.

| Checkpoint | Mode evaluated | WER | 95% confidence interval |
|---|---|---:|---:|
| RASR Parakeet v1 | Offline | 15.68% | 14.70–16.66% |
| RASR Parakeet v2 | Offline | 17.33% | 16.26–18.40% |
| qenneth Parakeet | Offline | 18.18% | 17.17–19.22% |
| jlvdoorn Whisper large v3 | Offline | 19.82% | 18.72–20.95% |
| Our Jacktol-trained P2 | Offline | 20.11% | 19.01–21.24% |
| Our P4 Parakeet | Offline | 20.16% | 19.11–21.25% |
| Jacktol Whisper large v3 | Offline | 20.29% | 18.91–21.83% |
| Our N4 Nemotron | Native streaming | 22.32% | 21.21–23.47% |
| YouTube ATC FastConformer | Offline | 28.50% | 27.05–30.01% |
| STEAK synthetic Whisper | Offline | 30.50% | 29.11–31.94% |
| US ATC Whisper Turbo v3 | Offline | 69.25% | 64.91–75.72% |
| XLS-R ATCOSIM | Offline | 90.39% | 89.56–91.17% |

Intervals use 2,000 bootstrap draws clustered by source recording, with seed 20261106. They describe sampling uncertainty on this subset; they do not establish training independence or a statistically significant ranking between nearby models.

The [comparison CSV](../reports/evaluation-comparison-2026-11-06.csv) contains only this subset, with error counts, confidence intervals, training data and evaluation denominators. The [benchmark](community-atc-benchmark-2026-11-06.md) documents inference and scoring.

## How the evaluation subset was formed

We removed **639 of the original 1,908 utterances**: the union of 213 Jacktol-linked utterances and 638 exact public ATCO2 train/validation matches. Of those, 212 occur in both groups. The same exclusions apply to every model, including matches outside a particular recipe’s duration window.

Existing predictions were rescored using the original lexical normalizer and corpus WER formula. No new inference, fine-tuning, decoding change or checkpoint selection was performed. Original full-test error totals reproduced before rescoring. The previous full-test, Jacktol-only and training-only exclusion views are not included in the current results because they retain identified overlap. Historical artifacts remain available for audit traceability.

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
- [Aggregate audit JSON and source pins](../reports/atco2-training-overlap-audit-2026-11-06.json) (also retains historical diagnostic views); [current comparison CSV](../reports/evaluation-comparison-2026-11-06.csv). Row IDs, audio, transcripts, per-utterance matches and masks remain in restricted storage; only their hashes and aggregate results are published.

**Status:** direct public ATCO2 audio overlap is established and a common exclusion-based comparison is available. Full-ancestry clearance remains open until exact training manifests, preprocessing/version details and inaccessible base/teacher histories can be reconciled. The [tracker](evaluation-tracker.md) separates these completed findings from the remaining work.
