# Community ATC benchmark on the private two-hour ATCO2 test

Completed **2026-11-06**. All **eight shortlisted ATC-adapted checkpoints** produced predictions for all **1,908 utterances**. Existing qenneth, P4 and N4 predictions were checked against the same audio, offsets, durations and normalized references, then rescored. The wider inventory includes unverified and excluded entries; those were not inferred.

**RASR Parakeet v1 has the lowest observed primary WER: 12.71%.** This is an offline comparison with unresolved ATCO2 training overlap, not an independent-generalization claim or a streaming replacement decision. N4 remains the selected streaming release path.

## Primary results

Lower WER is better. All community runs and P4 are offline; N4 retains its established native-streaming evaluation condition.

| Model | Training data used in adaptation | Full 2h WER | Known-Jacktol-disjoint WER | Prediction source |
|---|---|---:|---:|---|
| RASR Parakeet v1 | 200k synthetic RadioTalk utterances (Llama dialogue + Tada speech), plus real ATCO2/ATCOSIM training splits | 12.71% | 13.36% | New inference |
| RASR Parakeet v2 | Inherits v1; adds 28.8 h / 35,240 TartanAviation US clips with ASR-generated labels + ADS-B correction; ATCO2/ATCOSIM replay | 14.88% | 15.28% | New inference |
| qenneth Parakeet ATC | Jacktol ATC-ASR dataset; exact training amount/revision not documented in reviewed card | 15.91% | 16.65% | Existing, identity-verified and rescored |
| jlvdoorn Whisper large v3 ATCO2 | ATCO2-ASR via WhisperATC; project describes an 80/20 train/validation split | 16.20% | 16.91% | New inference |
| Jacktol Whisper large v3 | Jacktol ATC-ASR dataset; exact training amount/revision not documented in reviewed card | 16.30% | 17.57% | New inference |
| Our P4 Parakeet (offline) | 314.716 h ATCO2 Silver + 314.721 h LibriSpeech; then 0.418 h human-accepted Gold + English replay | 20.00% | 19.89% | Existing, identity-verified and rescored |
| Our N4 Nemotron (streaming) | 314.716 h ATCO2 Silver + 314.721 h LibriSpeech; then 0.418 h human-accepted Gold + English replay | 22.46% | 22.33% | Existing, identity-verified and rescored |
| YouTube ATC FastConformer | 800+ h reported YouTube VATSIM/IVAO corpus; human simulator speech with ASR/LLM-generated labels | 26.75% | 27.10% | New inference |
| STEAK synthetic Whisper large v3 | 20k synthetic STEAK v2 ATC utterances; one pass; no real ATC audio reported in fine-tuning | 29.01% | 29.07% | New inference |
| US ATC Whisper Turbo v3 | 1,216 human-transcribed Minneapolis-area radio clips; augmentation expands these to 6,080 files | 66.70% | 66.91% | New inference |
| XLS-R 300M ATCOSIM | ATCOSIM human speech from ATC simulations (Jzuluaga/atcosim_corpus) | 88.64% | 88.91% | New inference |

Training sources for community checkpoints are publisher/project-reported; ours come from internal records. These are adaptation pools, not complete base pretraining histories. See the [12-model combined comparison](evaluation-tracker.md) for per-model evidence caveats and the [training-data map](model-training-data.md) for sources.

The full test has **2.000441 hours, 21,930 reference words and 1,168 source recordings**. The supplementary view has **1,695 utterances, 1.7494 hours, 19,077 words and 1,040 source recordings**. It removes 213 segments linked to known Jacktol source recordings, including 197 linked specifically to its training split. It is **not universally training-disjoint**: RASR and jlvdoorn also have ATCO2 ancestry that needs a separate source audit.

Community authors did not test on our unpublished package. These are our internal measurements of their published weights. Shared upstream ELDA/ATCO2 recordings can exist even when our selected package has never been published.

See [where training and test audio overlap](evaluation-tracker.md#where-training-and-test-audio-overlap) for the distinction between 213 Jacktol-linked exclusions (197 training-linked), 554 exact duplicates in the excluded Gold-additive variant, and unresolved ATCO2 ancestry. That section also defines the supplementary score and explains why these findings are not automatically attributable to P4/N4.

## Number-format sensitivity

The primary scorer preserves the study's original contract: NFKD and diacritic folding, lowercase, symbol removal except ASCII apostrophes, and whitespace collapse. It does **not** convert spoken numbers to digits. All reference rows spell numbers out; US Whisper Turbo produces digits in 1,644 of 1,908 hypotheses.

The table below is a **post-hoc diagnostic**, using the same existing Transformers 5.3.0 Whisper `EnglishTextNormalizer` on every system and both sides of the comparison. No model, decoding setting or prediction was changed. It also changes fillers, contractions, spelling and punctuation, so it is not a number-only metric or callsign correctness score. Word denominators become **17,566 full / 15,178 supplementary**; these percentages must not be directly mixed with primary WER.

| Model | English-normalized full WER | English-normalized disjoint WER |
|---|---:|---:|
| RASR Parakeet v1 | 14.88% | 15.71% |
| RASR Parakeet v2 | 17.55% | 18.11% |
| qenneth Parakeet ATC | 19.64% | 20.66% |
| jlvdoorn Whisper large v3 ATCO2 | 19.59% | 20.56% |
| Jacktol Whisper large v3 | 19.98% | 21.64% |
| Our P4 Parakeet (offline) | 24.55% | 24.54% |
| Our N4 Nemotron (streaming) | 27.54% | 27.53% |
| YouTube ATC FastConformer | 30.78% | 31.16% |
| STEAK synthetic Whisper large v3 | 34.42% | 34.56% |
| US ATC Whisper Turbo v3 | 51.78% | 51.74% |
| XLS-R 300M ATCOSIM | 96.09% | 96.42% |

US Whisper Turbo changes from **66.70%** primary WER to **51.78%** with standard English normalization. Both values belong in the assessment; output formatting alone should not determine the conclusion about this model. The normalizer and spelling-map hashes are recorded in the JSON. ATC-specific digit-sequence and callsign accuracy remain unmeasured.

## Uncertainty and error counts

| Model | Substitutions | Deletions | Insertions | Total errors | Full WER 95% CI |
|---|---:|---:|---:|---:|---|
| RASR Parakeet v1 | 1,745 | 693 | 350 | 2,788 | 11.99–13.45% |
| RASR Parakeet v2 | 2,073 | 787 | 403 | 3,263 | 14.07–15.70% |
| qenneth Parakeet ATC | 2,260 | 897 | 333 | 3,490 | 15.09–16.70% |
| jlvdoorn Whisper large v3 ATCO2 | 2,356 | 766 | 430 | 3,552 | 15.34–17.07% |
| Jacktol Whisper large v3 | 2,373 | 748 | 453 | 3,574 | 15.29–17.36% |
| Our P4 Parakeet (offline) | 2,905 | 814 | 667 | 4,386 | 19.14–20.83% |
| Our N4 Nemotron (streaming) | 3,112 | 1,285 | 528 | 4,925 | 21.54–23.42% |
| YouTube ATC FastConformer | 3,844 | 1,321 | 702 | 5,867 | 25.70–27.87% |
| STEAK synthetic Whisper large v3 | 3,877 | 1,720 | 764 | 6,361 | 27.90–30.13% |
| US ATC Whisper Turbo v3 | 8,824 | 4,497 | 1,307 | 14,628 | 63.27–71.36% |
| XLS-R 300M ATCOSIM | 10,804 | 8,443 | 191 | 19,438 | 87.98–89.35% |

Intervals use 2,000 bootstrap draws clustered by source recording, seed 20261106. The paired RASR v1 minus P4 difference on the supplementary view is **−6.54 percentage points**, 95% CI **[−7.31, −5.79]**. These intervals describe sampling variability within this test; they do not correct training overlap, geographic coverage or multiple comparisons.

## Checkpoint identity and ATC relevance

Each new candidate has documented air-traffic-control adaptation in its pinned card or companion project, rather than only “ATC” in its name. The [assessment](community-atc-model-assessment.md) contains the supporting sources and caveats.

| Evaluated checkpoint | Revision | Declared ATC adaptation |
|---|---|---|
| [twangodev/rasr-parakeet-v1](https://huggingface.co/twangodev/rasr-parakeet-v1/tree/1faf953af2f07946566ff426e4609d0325f61b33) | `1faf953af2f07946566ff426e4609d0325f61b33` | Synthetic US ATC plus ATCO2/ATCOSIM train anchors |
| [twangodev/rasr-parakeet-v2](https://huggingface.co/twangodev/rasr-parakeet-v2/tree/cb6ed4e5e68db54b4a8787dcea9706c2160a1354) | `cb6ed4e5e68db54b4a8787dcea9706c2160a1354` | v1 ancestry plus 28.8h TartanAviation pseudo-labels and European rehearsal |
| [qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC](https://huggingface.co/qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC/tree/ac5200eec345e4e674a8a685ef52ab6b2cf4fa95) | `ac5200eec345e4e674a8a685ef52ab6b2cf4fa95` | Jacktol ATC |
| [jlvdoorn/whisper-large-v3-atco2-asr](https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr/tree/f7a0dddfea9de70f9f5b2e1332ae19734e60081f) | `f7a0dddfea9de70f9f5b2e1332ae19734e60081f` | ATCO2 per companion WhisperATC project; card omits split detail |
| [jacktol/whisper-large-v3-finetuned-for-ATC](https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC/tree/d82c726fc7021239f28988126005285ee91c338e) | `d82c726fc7021239f28988126005285ee91c338e` | Jacktol ATC |
| [nvidia/nv-atc02-finetune-p4](https://huggingface.co/nvidia/nv-atc02-finetune-p4/tree/4b17d465eb441cc6ddb332a27a340b62aa42d051) | `4b17d465eb441cc6ddb332a27a340b62aa42d051` | Internal fine-tune; see N4/P4 comparison |
| [nvidia/nv-atc02-finetune-n4](https://huggingface.co/nvidia/nv-atc02-finetune-n4/tree/bb06688439dd09aca9a072a5c0c52e7f2aef07d1) | `bb06688439dd09aca9a072a5c0c52e7f2aef07d1` | Internal fine-tune; see N4/P4 comparison |
| [niclaswue/youtube-atc-fastconformer](https://huggingface.co/niclaswue/youtube-atc-fastconformer/tree/71dd41b077fa7633742e303cb298c87b2598c241) | `71dd41b077fa7633742e303cb298c87b2598c241` | 800+ hours of virtual ATC YouTube material; pseudo-labels and LLM fusion |
| [camillebrl/whisper-large-v3-synthetic-20k-matched-psd_snrm10_drv6_o4](https://huggingface.co/camillebrl/whisper-large-v3-synthetic-20k-matched-psd_snrm10_drv6_o4/tree/0d2381ed1eed8a5c8170a0252fd890f8ac89ff7b) | `0d2381ed1eed8a5c8170a0252fd890f8ac89ff7b` | STEAK v2; 20,000 synthetic utterances |
| [thomaseibner/whisper-large-v3-turbo-us-atc-v3](https://huggingface.co/thomaseibner/whisper-large-v3-turbo-us-atc-v3/tree/d1ec48290647ea6daf0602c48ef77c7d288d5983) | `d1ec48290647ea6daf0602c48ef77c7d288d5983` | 1,216 human-transcribed US radio clips plus augmentation |
| [Jzuluaga/wav2vec2-xls-r-300m-en-atc-atcosim](https://huggingface.co/Jzuluaga/wav2vec2-xls-r-300m-en-atc-atcosim/tree/b55f7042e0a6f64b93ec71af2b02ad080941fac1) | `b55f7042e0a6f64b93ec71af2b02ad080941fac1` | ATCOSIM |

US Whisper Turbo is a LoRA adapter loaded onto `openai/whisper-large-v3-turbo` at revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Every newly downloaded weight file was SHA-256 verified before use. File hashes, sizes, exact assets and the base pin are in the [machine-readable report](../reports/community-atc-benchmark-2026-11-06.json). P4/N4 links refer to private release repositories.

## Execution and checks

- Successful Slurm job **38095055**, eight A100-SXM4-80GB GPUs, all eight workers exited successfully; elapsed **5m52s**. No test rows were skipped. Empty hypotheses are retained and scored as deletions.
- Frozen source manifest SHA-256: `4af85f51d32051ee84c841c92c8e5e1eb69ee07201149197f905a7edc9c13082`. Exact offset/duration crops were checked sample-for-sample against the original 16 kHz mono PCM. Maximum clip length is 15.27 seconds.
- Primary scorer SHA-256: `cc05898d6e8dc243d6a12995799b8bc042216e4e2bf0bccf42f2f1d67e146b96`. Substitution/deletion/insertion totals independently match edit-distance counts for every row. Full and supplementary denominators are identical across models.
- New inference uses BF16, batch 8, no AMP, fixed utterance boundaries, no external LM, audio enhancement, reference-derived prompt, airport resolver or ADS-B context. Eight development utterances were used for loading/inference checks before the test; checkpoints were pinned before test scoring.
- Whisper: English transcription, greedy decoding, one beam, no sampling, `max_new_tokens=440`, no timestamps, SDPA. The US model's suggested production repetition penalty and resolver are not used; this is a shared disclosed inference condition, not a tuned best-case result.
- XLS-R: explicit Wav2Vec2 processor, masked/padded batch inputs, greedy CTC argmax and valid-feature-length trimming. NeMo: official standalone evaluator, greedy batch decoding, `max_symbols_per_step=10`, CUDA graph decoding; TDT mode for Parakeet and RNNT for hybrid FastConformer.
- P4/qenneth keep their existing offline BF16 predictions. N4 keeps native streaming FP32 with chunk configuration `[56,3]`. All three frozen full-test error totals reproduce exactly.
- Software: torch `2.7.0a0+79aa17489c.nv25.4`, Transformers `5.3.0`, PEFT `0.18.0`, safetensors `0.5.3`. The container reports NeMo package metadata `0.10.1`; its installed source supplies the actual evaluator. This metadata alone should not be treated as a portable NeMo release pin.

The unexpectedly high XLS-R WER prompted an independent **development-only** check in job 38095237: scalar FP32 inference gives 95 errors across 108 words, versus 96 with the original batched BF16 path. Four of eight normalized hypotheses match exactly. Poor transfer persists under this check; small precision/batching differences remain, and an optional language-model decoder was not evaluated.

An initial attempt stopped during development checks because of a container path alias, processor auto-selection and a renamed NeMo decoding field. These were fixed before test inference. No failed test run was selected or silently discarded.

An instrumentation label in the executed Whisper runner mistook missing EOS in a returned sequence for a generation-limit event. Transformers removes EOS during postprocessing, so this is **not a valid truncation count**. The report corrects the label; prediction text and all scores are unchanged. Actual generation-limit hits are unknown. The current [runner](../scripts/evaluate_community_atc.py) uses the corrected diagnostic name; the report retains the executed runner's hash.

Per-model elapsed times are retained in JSON, but the eight models shared a node, and NeMo timings include test model reload while HF timings exclude initial loading. They are not a controlled latency ranking. Native streaming latency and callsign/number entity accuracy were not measured here.

## Interpretation and artifacts

RASR v1 is the strongest observed offline candidate on both primary views. Its ATCO2 ancestry and synthetic/teacher lineage require an exact source audit before interpreting the difference as generalization or considering release eligibility. Jacktol-trained qenneth and Whisper have known shared-source concerns; the supplementary view addresses only the already audited Jacktol links. License metadata is not approval of downstream release.

This completes the first evaluation group, not all 287 screened repositories or all 38 detailed profiles. Expanded groups remain unmeasured. Results do not alter the selected N4 streaming release.

- [JSON: counts, intervals, pins, hashes and diagnostics](../reports/community-atc-benchmark-2026-11-06.json)
- [CSV: comparison metrics](../reports/community-atc-benchmark-2026-11-06.csv)
- [Community assessment and remaining candidates](community-atc-model-assessment.md)
- [Original N4/P4 release comparison](checkpoint-comparison-n4-p4.md)

Audio, reference transcripts and per-utterance hypotheses remain in the restricted evaluation storage. These repository artifacts contain aggregate results and reproducibility metadata only.

A subsequent [evaluation of our historical Jacktol-trained P2](jacktol-p2-current-test-2026-11-06.md) adds the matched-test result for that separate curriculum and resolves the older Gold-additive split mismatch.
