# Community ATC benchmark after removing known overlap

Updated **November 6, 2026**. This page reports measured results for all 12 evaluated checkpoints on the common subset remaining after the completed overlap audits.

All results below use the **same 1,269 utterances, 13,851 reference words, 768 source recordings and 1.262691 hours (75.76 minutes)**. This subset excludes the union of known Jacktol recording links and exact matches to the audited public ATCO2 training and validation sources.

**Scope: known overlap removed.** No retained utterance has an overlap identified by the completed audits. This is not a guarantee of zero contamination across every model’s full training history: unscanned corpora and incomplete base/teacher manifests remain. The subset was defined after inspecting the original test, so it is not a new blind test.

## Primary results

Lower WER is better. Every count and confidence interval in this table belongs to the retained subset.

| Checkpoint | Mode evaluated | WER | 95% confidence interval | Substitutions | Deletions | Insertions | Total word errors |
|---|---|---:|---:|---:|---:|---:|---:|
| RASR Parakeet v1 | Offline | 15.68% | 14.70–16.66% | 1,413 | 500 | 259 | 2,172 |
| RASR Parakeet v2 | Offline | 17.33% | 16.26–18.40% | 1,590 | 517 | 294 | 2,401 |
| qenneth Parakeet | Offline | 18.18% | 17.17–19.22% | 1,680 | 598 | 240 | 2,518 |
| jlvdoorn Whisper large v3 | Offline | 19.82% | 18.72–20.95% | 1,872 | 530 | 343 | 2,745 |
| Our Jacktol-trained P2 | Offline | 20.11% | 19.01–21.24% | 2,007 | 483 | 295 | 2,785 |
| Our P4 Parakeet | Offline | 20.16% | 19.11–21.25% | 1,861 | 485 | 446 | 2,792 |
| Jacktol Whisper large v3 | Offline | 20.29% | 18.91–21.83% | 1,920 | 505 | 385 | 2,810 |
| Our N4 Nemotron | Native streaming | 22.32% | 21.21–23.47% | 1,966 | 805 | 321 | 3,092 |
| YouTube ATC FastConformer | Offline | 28.50% | 27.05–30.01% | 2,648 | 805 | 494 | 3,947 |
| STEAK synthetic Whisper | Offline | 30.50% | 29.11–31.94% | 2,582 | 1,129 | 514 | 4,225 |
| US ATC Whisper Turbo v3 | Offline | 69.25% | 64.91–75.72% | 5,778 | 2,962 | 852 | 9,592 |
| XLS-R ATCOSIM | Offline | 90.39% | 89.56–91.17% | 6,755 | 5,636 | 129 | 12,520 |

Intervals use 2,000 bootstrap draws clustered by source recording, with seed 20261106. They describe sampling uncertainty on this subset; they do not establish training independence or a statistically significant ranking between nearby models.

RASR v1 has the lowest observed WER here. N4 remains the selected streaming release path; the offline comparison does not establish streaming suitability or controlled latency. Training sources and evidence caveats are listed beside the results in the [evaluation tracker](evaluation-tracker.md).

## Dataset exclusions and scoring

We removed the union of 213 Jacktol-linked utterances, 498 exact public ATCO2 training-source matches and 140 validation-source matches. Because 212 are present in both the Jacktol and ATCO2 groups, the total removed is **639**, leaving **1,269**. The previous Jacktol-only subset retained identified ATCO2 overlap. The [source audit](atco2-training-overlap-audit-2026-11-06.md) explains exact matches, recipe eligibility and remaining ancestry gaps.

The primary scorer applies NFKD and diacritic folding, lowercase, symbol removal except ASCII apostrophes, and whitespace collapse. It does not convert spoken numbers to digits. WER is the sum of substitutions, deletions and insertions divided by **13,851 reference words**. Empty hypotheses remain in the scoring denominator.

Number formatting affects this metric, particularly for US Whisper Turbo. A secondary English-normalized score has not been computed for this 1,269-utterance subset, so none is reported here. ATC-specific callsign and number accuracy also remain unmeasured.

These are our measurements of published weights. Community authors did not evaluate on our unpublished package; shared upstream recordings explain the identified overlap. Results are post-hoc and do not certify full training independence.

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

Our historical Jacktol-trained P2 is the checkpoint pinned and hashed in the [P2 evaluation record](jacktol-p2-current-test-2026-11-06.md); it is distinct from the excluded Gold-additive variant.

US Whisper Turbo is a LoRA adapter loaded onto `openai/whisper-large-v3-turbo` at revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Every newly downloaded weight file was SHA-256 verified before use. File hashes, sizes, exact assets and the base pin are in the [machine-readable report](../reports/community-atc-benchmark-2026-11-06.json). P4/N4 links refer to private release repositories.

## Original inference execution and checks

These conditions describe how the original predictions were generated. The results above select only the retained 1,269 utterances; inference was not rerun. P2 came from its separately documented evaluation. CPU audit/rescoring job **38098238** completed successfully and independently verified all 638 ATCO2 segment matches.

- Successful Slurm job **38095055**, eight A100-SXM4-80GB GPUs, all eight workers exited successfully; elapsed **5m52s**. No test rows were skipped. Empty hypotheses are retained and scored as deletions.
- Frozen source manifest SHA-256: `4af85f51d32051ee84c841c92c8e5e1eb69ee07201149197f905a7edc9c13082`. Exact offset/duration crops were checked sample-for-sample against the original 16 kHz mono PCM. Maximum clip length is 15.27 seconds.
- Primary scorer SHA-256: `cc05898d6e8dc243d6a12995799b8bc042216e4e2bf0bccf42f2f1d67e146b96`. Substitution/deletion/insertion totals independently match edit-distance counts for every row. The current comparison uses 13,851 reference words for every model.
- New inference uses BF16, batch 8, no AMP, fixed utterance boundaries, no external LM, audio enhancement, reference-derived prompt, airport resolver or ADS-B context. Eight development utterances were used for loading/inference checks before the test; checkpoints were pinned before test scoring.
- Whisper: English transcription, greedy decoding, one beam, no sampling, `max_new_tokens=440`, no timestamps, SDPA. The US model's suggested production repetition penalty and resolver are not used; this is a shared disclosed inference condition, not a tuned best-case result.
- XLS-R: explicit Wav2Vec2 processor, masked/padded batch inputs, greedy CTC argmax and valid-feature-length trimming. NeMo: official standalone evaluator, greedy batch decoding, `max_symbols_per_step=10`, CUDA graph decoding; TDT mode for Parakeet and RNNT for hybrid FastConformer.
- P4/qenneth keep their existing offline BF16 predictions. N4 keeps native streaming FP32 with chunk configuration `[56,3]`. All three frozen full-test error totals reproduce exactly.
- Software: torch `2.7.0a0+79aa17489c.nv25.4`, Transformers `5.3.0`, PEFT `0.18.0`, safetensors `0.5.3`. The container reports NeMo package metadata `0.10.1`; its installed source supplies the actual evaluator. This metadata alone should not be treated as a portable NeMo release pin.

The unexpectedly high XLS-R WER prompted an independent **development-only** check in job 38095237: scalar FP32 inference gives 95 errors across 108 words, versus 96 with the original batched BF16 path. Four of eight normalized hypotheses match exactly. Poor transfer persists under this check; small precision/batching differences remain, and an optional language-model decoder was not evaluated.

An initial attempt stopped during development checks because of a container path alias, processor auto-selection and a renamed NeMo decoding field. These were fixed before test inference. No failed test run was selected or silently discarded.

An instrumentation label in the executed Whisper runner mistook missing EOS in a returned sequence for a generation-limit event. Transformers removes EOS during postprocessing, so this is **not a valid truncation count**. The report corrects the label; prediction text and all scores are unchanged. Actual generation-limit hits are unknown. The current [runner](../scripts/evaluate_community_atc.py) uses the corrected diagnostic name; the report retains the executed runner's hash.

Per-model elapsed times are retained in JSON, but the eight models shared a node, and NeMo timings include test model reload while HF timings exclude initial loading. They are not a controlled latency ranking. Native streaming latency and callsign/number entity accuracy were not measured here.

## Artifacts and scope

- [Current comparison CSV](../reports/evaluation-comparison-2026-11-06.csv): only the common 1,269-utterance subset, with training data, error counts and confidence intervals.
- [Audit JSON](../reports/atco2-training-overlap-audit-2026-11-06.json): use `common_mask_rescoring.scores.<model>.exclude_jacktol_and_audited_atco2_training_and_validation` for the results above. Other views are historical diagnostics.
- [Original inference evidence](../reports/community-atc-benchmark-2026-11-06.json): checkpoint pins, hashes, decoding conditions and earlier metrics retained for traceability, not the current comparison.
- [P2 checkpoint and execution evidence](../reports/jacktol-p2-current-test-2026-11-06.json).
- [Community assessment and remaining candidates](community-atc-model-assessment.md).

Only these 12 checkpoints have results in the current comparison; broader inventory entries remain unmeasured. Audio, reference transcripts and per-utterance predictions remain in restricted storage.
