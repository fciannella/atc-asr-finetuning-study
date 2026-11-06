# Nemotron Transcribe ATC bounded pilot

**Status:** data preparation/audit and recipe freezing passed. GPU preflight is queued with `QOSMaxJobsPerUserLimit` while another job occupies the user allowance; training and evaluation jobs depend on successful earlier phases. No pilot WER or completed optimization is reported yet. This is a dated submission record, not a live dashboard.

The user selected a bounded pilot on November 6, 2026. Goal: measure whether a small ATC LoRA adaptation improves decoded ATCO2 development WER over the same original base, while limiting general-English regression. This is a separate offline SpeechLM experiment; N4 remains the streaming release candidate.

## Frozen recipe

- Base: `nvidia/Nemotron-3.5-Transcribe-30B-A4B@d64e69ed50d55926815cbac332d1f3017ab105d4`, weight SHA-256 `fe3b898b42a65837180c65dedd2e62b92836cee704ae3cbee2f473946fc7d0c9`.
- Reuse the validated Dante A100 SpeechLM2 image and `NVIDIA-NeMo/Speech@e5d79d5a49cbec49d79440b0070aac0c32a8dbb8`; start from the original base, not the Italian adapter. Retain the timestamp head and verify all required tensor restores and exported LoRA merges.
- Data: deterministic hash-ranked 20 hours from the audited ATCO2 Silver pool and 5 hours of LibriSpeech replay, then deterministic shuffle. This is an 80/20 duration mixture of unique selected audio, not an 80/20 example sampler. Silver labels are supplied machine transcripts. No Jacktol, UWB or Gold test/development examples enter training; Gold training is not included in this first pilot.
- Protect all 1,908 original test utterances and all 100 development utterances, even though final scoring retains only 1,269. Exclude matching source metadata/airport-date groups, exact source-file/cropped-PCM duplicates and eight-word reference overlap. Remove training duplicates. Record filters and actual counts. These checks cannot certify unknown base pretraining or transformed copies.
- Train LoRA rank 32, alpha 64 and modality projection. Freeze encoders, routers, auxiliary branches and timestamp head. BF16, 2,048 packed tokens, bucketing off, AdamW `1e-4`, 50 warmup updates, 500 total updates. Preserve candidates at 100/250/500; validation-loss checks every 100 updates. Count the actual sampler and report the fraction of an epoch; do not describe 500 updates as full-corpus training.
- Fixed ATC instruction: “Transcribe only the words actually spoken in this English air traffic control audio. Write spoken numbers as words. Do not add explanations or infer missing callsigns.” English replay and guard receive a generic English transcription instruction. No reference text or callsign hints enter inference prompts. No prompt or recipe search on test.

## Execution and decision rules

1. CPU preparation checks and freezes manifests, labels, requests, audio hashes and exclusions.
2. One A100: architecture coverage, full base checksum, actual sampler count, disposable 20-step smoke (finite gradients, real LoRA/projection changes, two validation events), then base development and English guard decoding. Full training reinitializes from the original base.
3. One A100: the 500-update training run, contingent on successful preflight and smoke.
4. One A100: export and verify all three candidates; decode each on the same 100-row ATCO2 development set and 128-row deterministic LibriSpeech test-clean guard. Select lowest development WER among candidates whose guard rises by at most 2 absolute WER points versus base; ties favor earlier steps. If every candidate fails, report the best development candidate as diagnostic, with no recommendation.
5. One A100: after a durable selection lock, decode the original base and selected model on the same 1,269-utterance subset with known overlap removed. Apply the existing ATC lexical normalizer and pool word errors over 13,851 words. Keep test out of selection.

Four dependent GPU jobs request one A100 80GB each, sequentially, with a 3h45m limit each (15 GPU-hours maximum allocation if all hit the limit). Pending jobs reserve no GPU. Actual wall time and peak memory will be reported; Dante timings are not a calibrated ATC cost estimate. Dataset preparation uses a separate CPU job. No further tuning cycle is included.

## Success and limits

Engineering success requires exact restoration, finite optimization, real trainable-weight updates, successful validation events, verified exports and matched prediction pairs. Model recommendation additionally requires improved development WER and a passing English guard. Final test results report observed gains or regressions without selecting a new checkpoint. We do not yet have a matched base score, so an improvement probability or promised target WER would be speculative.

The 128-row guard is a custom evaluation-only regression subset, not an official LibriSpeech benchmark score. The existing ATCO2 test has been repeatedly inspected and is a regression comparison, not a new blind test. Full pretrained/teacher ancestry remains unknown. No streaming, timestamp-accuracy or release-approval claim follows from this run.

The experiment is adapted from the user's Dante strategy and its archived `finetune-speech-llm` skill. Its task-specific departures are fixed prompts without a test sweep, native NeMo evaluation instead of vLLM, frozen speech encoders, and the bounded 500-update schedule. Source audio, training manifests, predictions and checkpoints remain on restricted Lustre. Only aggregate summaries may be published in the study repository.


## Submitted jobs and frozen data

| Phase | Slurm job | State at submission check |
|---|---:|---|
| Data preparation and overlap checks | 38102754 | Completed, exit 0 |
| Freeze recipe and input hashes | 38102857 | Completed, exit 0 |
| GPU preflight, 20-step smoke, base development/guard | 38102859 | Queued |
| 500-update training | 38102860 | Waiting for preflight |
| Candidate export, development/guard evaluation, selection | 38102861 | Waiting for training |
| Matched base/selected test evaluation and report | 38102862 | Waiting for selection |

The frozen training data contains **25,822 Silver clips / 20.000167 hours** plus **1,423 English clips / 5.001900 hours**, totaling **27,245 clips**. Before reaching the ATC sampling target, 1,344 candidates were excluded for eight-word overlap with held-out references. The completed checks found no remaining source metadata, exact source-file or cropped-PCM collision with the protected held-out sets; this does not certify absence of transformed audio or pretraining overlap.

Development contains **100 utterances / 1,108 reference words**. The custom English guard contains **128 utterances / 2,645 words**. Final test is the same **1,269 utterances / 13,851 words** as the [current comparison](evaluation-tracker.md). All test results will remain unreported until base and selected checkpoint decoding complete.

[Aggregate submission record](../reports/nemotron-transcribe-atc-pilot-2026-11-06.json) records exact checkpoint/configuration identities, dataset counts and job IDs. Large weights, row-level training/evaluation data and predictions stay in restricted storage. This submission does not upload or release a new model to Hugging Face.
