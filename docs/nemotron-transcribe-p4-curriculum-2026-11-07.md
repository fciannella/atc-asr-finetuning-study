# Nemotron Transcribe: P4-style two-stage curriculum

**Status, November 9:** the full data audit passed and Silver training reached update 7,797 before a job timed out. The last complete checkpoint is update 7,000. Recovery is validating an exact packing optimizations and shorter jobs on eight GPUs. Gold refinement and final evaluation have not started. This is a dated experiment record, not a live dashboard.

The November 7 run transfers Parakeet P4's successful Silver → human-Gold curriculum to `nvidia/Nemotron-3.5-Transcribe-30B-A4B`. It starts from the original Transcribe base, not the earlier 500-update pilot. It is a curriculum reproduction across architectures, not an identical Parakeet training run.

## Training contract

| Ingredient | Silver adaptation | Human-Gold refinement |
|---|---|---|
| Initialization | Original Transcribe base | Development-selected, verified and merged Silver checkpoint |
| Available source pools, before this audit | 314.716 h Silver + 314.721 h English | 393 human-accepted Gold clips / 0.417877 h + English pool |
| Source sampling weights | Silver 80%, English 20% | Gold 85%, English 15% |
| Optimizer updates | 20,000 | 2,000 |
| Peak learning rate | `1e-4` | `1e-5` |
| Warmup / cosine floor | 200 updates / `1e-6` | 100 updates / `1e-6` |
| Candidate checkpoints | 9,000; 15,000; 20,000 | 500; 1,000; 2,000 |
| Seed | 42 | 42 |
| Optimizer | AdamW, weight decay 0.001 | Fresh AdamW and schedule, weight decay 0.001 |
| Hardware | Eight OCI A100 80GB GPUs | Eight OCI A100 80GB GPUs |

Source weights control repeated dataset draws, not a fixed 80/20 split of unique audio hours. The complete eligible source pools are available to the sampler. The completed audit retained 377,400 Silver clips / 290.341636 h, 90,069 English clips / 314.717682 h, and all 393 Gold clips / 0.417877 h. It excluded 19,050 Silver clips and one English clip under the conservative eight-word held-out-reference filter; a text match alone does not establish audio contamination.

Both stages train the speech encoder's ASR branch and normalization, modality projection, and rank-32/alpha-64 LoRA adapters on the language backbone. Other encoder branches, routers and the timestamp head remain frozen. Gold begins from the merged Silver model with fresh LoRA adapters. Silver uses BF16 and 2,048 packed tokens per GPU, giving a 16,384-token global budget per update. Gold retains BF16 but uses two examples per GPU (global batch 16), matching P4's Gold example-batch size.

These settings differ from P4's ASR architecture and its float32 Gold precision. Optimizer-step counts and compute are not directly equivalent between architectures. Candidate selection uses decoded development WER rather than importing Parakeet's selected step number.

## Data and scoring boundaries

- Reuse P4's frozen Silver, English and human-Gold sources. No Jacktol or UWB adaptation.
- Protect all original 1,908 test utterances, all 100 development utterances and the fixed 128-row English guard using source identities, airport/date groups, exact source-file hashes and cropped-PCM hashes.
- Apply the previous conservative eight-word reference-overlap filter to Silver and English. For human Gold, shared phrase text alone is not evidence of audio contamination; source and audio separation remain mandatory. Require all 393 Gold training clips to pass those checks.
- Keep raw sources unchanged and derive lowercase lexical labels. Preserve the pilot's fixed transcription prompts and decoding budget; no test-driven prompt or decoder search.
- Use the same 100-row development set and custom 128-row English guard for selection. The guard may increase by at most two absolute WER points over the original base. Ties favor the earlier checkpoint. An all-failing set produces only a diagnostic candidate.
- Lock the Gold selection before scoring the original base, selected Silver parent and Gold model on the same 1,269-utterance / 13,851-word test subset with known overlap removed. Verify matched audio crops and prompts.

Capped repetitive outputs remain fully counted in WER, including insertion errors. Report cap frequency alongside S/D/I. Do not discard those utterances, trim repeated words, change the cap after examining test outputs, or stop valid score aggregation merely because the cap was reached.

The existing test has been repeatedly inspected. This audit does not certify unknown base-model/teacher ancestry or every transformed recording. The English guard participates in selection and is not an untouched general-domain benchmark.

## Execution and validation

The regular OCI partition limits jobs to four hours. The original submission used 1,000-update chunks with 3h45m allocation limits. Recovery switches to 500-update chunks after two short continuation checks. The full 20,000/2,000-update scheduler horizons remain fixed. Each continuation restores model weights, optimizer, scheduler, all eight data-loader states, callback counters and per-rank random state.

An engineering smoke on the previously audited small pool first tests 20 updates, checkpoint save, and 20 further updates after restoring the distributed checkpoint. Each actual stage also repeats this smoke against its own configuration and inputs. Smoke weights are discarded. Mandatory checks include exact parent-weight restoration, zero new LoRA B, finite gradients, real encoder/projection/LoRA changes, distinct rank sampler streams, observed source-mixture proportions and validation events.

The first engineering attempt failed during CUDA initialization before training and was retried on another node. The retry passed all 40 updates on eight GPUs, restored at update 20 with 733 optimizer-state entries and one scheduler state, and verified finite gradients on every rank. It observed changes in 356 encoder tensors, two projection tensors and 99 LoRA B tensors. The sampled source mix was 80.16% Silver and 19.84% English, with distinct streams on all eight ranks. These are engineering checks, not accuracy results. Main training requires both the complete data audit and a successful engineering continuation check. Every subsequent phase depends on successful earlier phases; failures cancel downstream jobs.

The prior frozen-encoder pilot ran around three seconds per steady-state update. At that rate alone, 22,000 updates take roughly 18.3 hours, before encoder-training cost, startup, checkpointing, queueing and evaluation. This is context for scale, not a completion estimate.

## Why this differs from the pilot

The earlier pilot completed 500 updates but did not reproduce P4's curriculum: it used 20 h Silver plus 5 h English, frozen speech encoders, and no Gold refinement. Its development WER was 44.49% for the base and 48.38% for the best tied candidates. Repetitive insertions outweighed fewer substitutions/deletions. Those are development results, not the 1,269-row test results in the community comparison.

This experiment changes data scale, weighted sampling, encoder adaptation and Gold refinement together. It can test whether the combined curriculum improves Transcribe; it cannot identify which ingredient caused any gain or regression.

## November 9 timeout and recovery

The original jobs completed through Silver update 7,000. Job `38145699` timed out during the next chunk at update 7,797, and its dependent jobs were cancelled. The update-7,000 checkpoint exists with all eight distributed shards and metadata; its model, optimizer and scheduler state had already restored successfully in that final attempt. Updates 7,001–7,797 were not checkpointed and must be repeated.

The main issue was growing data-loader replay overhead. Before the timed-out chunk, restoring the streaming sampler took approximately three hours; steady-state model updates remained around three seconds. Merely reducing the number of new updates per job would eventually fail as the replay grew.

Recovery uses a separately frozen runtime overlay, retaining the original code, checkpoints, manifests and training configurations. It memoizes immutable per-example token lengths inside the exact packing sampler's 10,000-example buffer. It also stops subset-sum search once the earliest full-capacity solution is found: later candidates cannot improve utilization and the original tie rule already prefers the earlier solution. Exhaustive and randomized tests compare the exact selected indices across 65,732 cases, including 10,000-item buffers. Both optimizations preserve batch choices, data order, rank sharding and checkpoint format. Two eight-rank comparisons passed: exact ordered batch identities, audio crops and token arrays matched before and after both optimizations, including continuation from the preceding sampler implementation. The original versus final 100-batch sampling test took approximately 257 versus 56–58 seconds per rank (about 4.5× faster). This measures sampler work, not end-to-end training throughput or WER. CPU audit jobs `38273475` and `38273732` completed successfully.

The current GPU recovery job is `38274206`, running on one node with eight GPUs, 64 CPUs and 768 GiB RAM. The first recovery chunk has not yet completed at this observation.

The recovery sequence is update 7,000 → 7,020 → 7,040 → 7,500, then 500-update chunks through 20,000. Gold also uses 500-update chunks through 2,000. Both short recovery jobs must finish within 60 minutes before allowing their successors. All GPU allocations remain below the four-hour partition limit, at 3h45m. Each successful job submits only its next dependent job, avoiding the earlier queue-submission limit. Training and smoke phases use eight GPUs; standalone evaluation uses one.

The original 20,000/2,000-update scheduler horizons, source weights, candidate checkpoints, development selection rule and final evaluation protocol remain unchanged. No new final WER is available.

[Aggregate run record](../reports/nemotron-transcribe-p4-curriculum-2026-11-07.json) records pins, the original job history and recovery status. Source audio, row-level manifests, predictions and checkpoints remain in restricted storage. This run does not publish a model or establish release approval.

References: [P4 lineage](checkpoint-comparison-n4-p4.md#training-lineage-and-selection-rationale), [Gold experiments](experiments/02-gold-refinement.md), [earlier Transcribe pilot](nemotron-transcribe-atc-pilot-2026-11-06.md), [current common-test comparison](evaluation-tracker.md).
