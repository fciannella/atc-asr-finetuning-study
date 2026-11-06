# Nemotron Transcribe ATC bounded pilot

**Status:** the distributed configuration/sampler audit and one-GPU preflight passed. The first eight-GPU smoke failed during checkpoint initialization before optimization; its dependent training/evaluation jobs were cancelled. A separate recovery attempt (`r2`) corrects initialization of the new LoRA parameters and its GPU preflight is now running. Training has not started and no pilot WER is reported. This is a dated submission record, not a live dashboard.

The user selected a bounded pilot on November 6, 2026. Goal: measure whether a small ATC LoRA adaptation improves decoded ATCO2 development WER over the same original base, while limiting general-English regression. This is a separate offline SpeechLM experiment; N4 remains the streaming release candidate.

## Frozen recipe

- Base: `nvidia/Nemotron-3.5-Transcribe-30B-A4B@d64e69ed50d55926815cbac332d1f3017ab105d4`, weight SHA-256 `fe3b898b42a65837180c65dedd2e62b92836cee704ae3cbee2f473946fc7d0c9`.
- Reuse the validated Dante A100 SpeechLM2 image and `NVIDIA-NeMo/Speech@e5d79d5a49cbec49d79440b0070aac0c32a8dbb8`; start from the original base, not the Italian adapter. Retain the timestamp head and verify all required tensor restores and exported LoRA merges.
- Data: deterministic hash-ranked 20 hours from the audited ATCO2 Silver pool and 5 hours of LibriSpeech replay, then deterministic shuffle. This is an 80/20 duration mixture of unique selected audio, not an 80/20 example sampler. Silver labels are supplied machine transcripts. No Jacktol, UWB or Gold test/development examples enter training; Gold training is not included in this first pilot.
- Protect all 1,908 original test utterances and all 100 development utterances, even though final scoring retains only 1,269. Exclude matching source metadata/airport-date groups, exact source-file/cropped-PCM duplicates and eight-word reference overlap. Remove training duplicates. Record filters and actual counts. These checks cannot certify unknown base pretraining or transformed copies.
- Train LoRA rank 32, alpha 64 and modality projection. Freeze encoders, routers, auxiliary branches and timestamp head. BF16, 2,048 packed tokens per GPU (16,384 global token budget per update), bucketing off, AdamW `1e-4`, 50 warmup updates, 500 total updates. Preserve candidates at 100/250/500; validation-loss checks at the smaller of 100 updates and 60% of the measured per-rank epoch. The completed sampler audit measured 176 batches per rank per epoch: 500 updates are approximately 2.84 epochs, with validation every 100 updates. All 27,245 unique examples are covered with no omissions or padding duplicates. Keeping 500 updates with eight GPUs increases the per-update token budget eightfold versus the superseded one-GPU plan; the learning rate is unchanged.
- Fixed ATC instruction: “Transcribe only the words actually spoken in this English air traffic control audio. Write spoken numbers as words. Do not add explanations or infer missing callsigns.” English replay and guard receive a generic English transcription instruction. No reference text or callsign hints enter inference prompts. No prompt or recipe search on test.

## Execution and decision rules

1. CPU preparation checks and freezes manifests, labels, requests, audio hashes and exclusions. The replacement configuration audit checks complete coverage and equal batch counts across eight finite, map-style sampler partitions, and records any tail padding.
2. One A100: architecture coverage, full base checksum, and base development and English guard decoding.
3. **Eight A100s:** disposable 20-update smoke test. Require all eight workers, distinct data partitions, finite gradients, real LoRA/projection changes and two validation events. Discard smoke weights.
4. **Eight A100s:** the 500-update training run, initialized from the original base and contingent on successful preflight and smoke. Use eight `torchrun` workers and NeMo Automodel FSDP2 (`dp_size=8`, `dp_replicate_size=1`; tensor, pipeline, context and expert parallel sizes are one). Save checkpoints collectively across ranks; write shared reports on rank zero.
5. One A100: export and verify all three candidates; decode each on the same 100-row ATCO2 development set and 128-row deterministic LibriSpeech test-clean guard. Select lowest development WER among candidates whose guard rises by at most 2 absolute WER points versus base; ties favor earlier steps. If every candidate fails, report the best development candidate as diagnostic, with no recommendation.
6. One A100: after a durable selection lock, decode the original base and selected model on the same 1,269-utterance subset with known overlap removed. Apply the existing ATC lexical normalizer and pool word errors over 13,851 words. Keep test out of selection.

Each GPU phase has a 3h45m time limit. Smoke and training each request one OCI node with eight A100 80GB GPUs, 64 CPU cores and 768GB host RAM. Preflight, selection and test each request one GPU, eight CPU cores and 192GB RAM. The combined maximum allocation is 71.25 GPU-hours if all five phases reach their limits; this is a cap, not a runtime estimate. Pending jobs reserve no GPU, and the existing per-user job limit may delay launch. Actual wall time and peak memory will be reported. No further tuning cycle is included.

## Success and limits

Engineering success requires exact restoration, finite optimization, real trainable-weight updates, successful validation events, verified exports and matched prediction pairs. Model recommendation additionally requires improved development WER and a passing English guard. Final test results report observed gains or regressions without selecting a new checkpoint. We do not yet have a matched base score, so an improvement probability or promised target WER would be speculative.

The 128-row guard is a custom evaluation-only regression subset, not an official LibriSpeech benchmark score. The existing ATCO2 test has been repeatedly inspected and is a regression comparison, not a new blind test. Full pretrained/teacher ancestry remains unknown. No streaming, timestamp-accuracy or release-approval claim follows from this run.

The experiment is adapted from the user's Dante strategy and its archived `finetune-speech-llm` skill. Its task-specific departures are fixed prompts without a test sweep, native NeMo evaluation instead of vLLM, frozen speech encoders, and the bounded 500-update schedule. Source audio, training manifests, predictions and checkpoints remain on restricted Lustre. Only aggregate summaries may be published in the study repository.


## Submitted jobs and frozen data

| Phase | Slurm job | GPUs | State at recovery submission |
|---|---:|---:|---|
| Original data preparation and overlap checks | 38102754 | 0 | Completed; audited inputs reused |
| Eight-rank configuration and sampler audit | 38106371 | 0 | Completed; unchanged sampler evidence reused |
| First GPU preflight and base development/guard | 38106405 | 1 | Completed |
| First distributed smoke | 38106406 | 8 | Failed during checkpoint initialization; zero optimization steps |
| Recovery GPU preflight and base development/guard | 38109816 | 1 | Running |
| Recovery distributed 20-update smoke | 38109817 | 8 | Waiting for preflight |
| Recovery 500-update training | 38109818 | 8 | Waiting for smoke |
| Recovery export, development/guard and selection | 38109819 | 1 | Waiting for training |
| Recovery matched base/selected test and report | 38109820 | 1 | Waiting for selection |

The original one-GPU jobs **38102859–38102862 were cancelled before starting**. After the first eight-GPU smoke failed, **38106410, 38106411 and 38106413** were automatically cancelled before starting. Every recovery GPU phase requires the preceding phase to succeed; invalid dependencies cancel downstream jobs.

### Initialization recovery

The pinned distributed loader defaults to requiring every model parameter in the checkpoint. The original base has no fine-tuning adapters, so it rejected the 244 newly created LoRA tensors before optimization. Recovery sets `model.init_from_checkpoint_strict=false`, the loader's supported option for intentional partial initialization. The mandatory pre-optimization callbacks still require all **1,307 original tensors** to restore exactly at loaded dtype, permit only new LoRA parameters as additional tensors, require initially zero LoRA B and keep timestamp parameters frozen. Missing base weights remain a fatal error in these checks.

The failed run remains unchanged in restricted storage. Recovery uses a new run directory and frozen hashes, repeats GPU preflight and the eight-GPU smoke, and retains the same dataset, optimizer and evaluation rules. The completed sampler audit is reused after verifying unchanged sampler settings and byte-identical relocated manifests. GPU validation of the initialization fix is still pending.

The frozen training data contains **25,822 Silver clips / 20.000167 hours** plus **1,423 English clips / 5.001900 hours**, totaling **27,245 clips**. Before reaching the ATC sampling target, 1,344 candidates were excluded for eight-word overlap with held-out references. The completed checks found no remaining source metadata, exact source-file or cropped-PCM collision with the protected held-out sets; this does not certify absence of transformed audio or pretraining overlap.

Development contains **100 utterances / 1,108 reference words**. The custom English guard contains **128 utterances / 2,645 words**. Final test is the same **1,269 utterances / 13,851 words** as the [current comparison](evaluation-tracker.md). All test results will remain unreported until base and selected checkpoint decoding complete.

[Aggregate submission record](../reports/nemotron-transcribe-atc-pilot-8gpu-2026-11-06-r2.json) records exact checkpoint/configuration identities, dataset counts and job IDs. Large weights, row-level training/evaluation data and predictions stay in restricted storage. This submission does not upload or release a new model to Hugging Face.
