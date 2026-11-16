# Nemotron Transcribe: P4-style two-stage curriculum

**Status, November 16 at 05:21 PST:** Stopped by user request to switch to Parakeet and synthetic-data experiments. Job **38636736** was cancelled; the automatic successor chain is disabled and the old verifier fails closed if accidentally invoked. The last complete Silver checkpoint remains **9,000 / 20,000** updates, with all eight shards preserved. No later optimizer updates or new checkpoint were recorded in this final attempt. Transcribe Gold and final evaluation did not run. See the [new campaign](parakeet-gold-synthetic-2026-11-16.md). Later sections below preserve the dated recovery history; they are not the current execution status.

The November 7 run transfers Parakeet P4's successful Silver → human-Gold curriculum to `nvidia/Nemotron-3.5-Transcribe-30B-A4B`. It starts from the original Transcribe base, not the earlier 500-update pilot. It is a curriculum reproduction across architectures, not an identical Parakeet training run.

## Training contract

| Ingredient | Silver adaptation | Human-Gold refinement |
|---|---|---|
| Initialization | Original Transcribe base | Development-selected, verified and merged Silver checkpoint |
| Available source pools, before this audit | 314.716 h Silver + 314.721 h English | 393 human-accepted Gold clips / 0.417877 h + English pool |
| Source sampling weights | Silver 80%, English 20% | Gold 85%, English 15% |
| Optimizer updates | 20,000 | Original plan: 2,000; now on hold |
| Peak learning rate | `1e-4` | `1e-5` |
| Warmup / cosine floor | 200 updates / `1e-6` | 100 updates / `1e-6` |
| Candidate checkpoints | 9,000; 15,000; 20,000 | 500; 1,000; 2,000 |
| Seed | 42 | 42 |
| Optimizer | AdamW, weight decay 0.001 | Fresh AdamW and schedule, weight decay 0.001 |
| Hardware | Eight OCI A100 80GB GPUs | Eight OCI A100 80GB GPUs |

Source weights control repeated dataset draws, not a fixed 80/20 split of unique audio hours. The complete eligible source pools are available to the sampler. The completed audit retained 377,400 Silver clips / 290.341636 h, 90,069 English clips / 314.717682 h, and originally all 393 Gold clips / 0.417877 h. The November 16 audit leaves 47 Gold clips / 2.722633 minutes after excluding the newly frozen evaluation release and shared recording groups. Gold is on hold pending more separate human-labeled data; see the recovery section below. It excluded 19,050 Silver clips and one English clip under the conservative eight-word held-out-reference filter; a text match alone does not establish audio contamination.

Both stages train the speech encoder's ASR branch and normalization, modality projection, and rank-32/alpha-64 LoRA adapters on the language backbone. Other encoder branches, routers and the timestamp head remain frozen. Gold begins from the merged Silver model with fresh LoRA adapters. Silver uses BF16 and 2,048 packed tokens per GPU, giving a 16,384-token global budget per update. Gold retains BF16 but uses two examples per GPU (global batch 16), matching P4's Gold example-batch size.

These settings differ from P4's ASR architecture and its float32 Gold precision. Optimizer-step counts and compute are not directly equivalent between architectures. Candidate selection uses decoded development WER rather than importing Parakeet's selected step number.

## Data and scoring boundaries

- Reuse P4's frozen Silver, English and human-Gold sources. No Jacktol or UWB adaptation.
- Protect all original 1,908 test utterances, all 100 development utterances and the fixed 128-row English guard using source identities, airport/date groups, exact source-file hashes and cropped-PCM hashes.
- Apply the previous conservative eight-word reference-overlap filter to Silver and English. For human Gold, shared phrase text alone is not evidence of audio contamination; source and audio separation remain mandatory. The original 393-row Gold pool passed those original checks. The November 16 revision additionally excludes the newly frozen evaluation release from Gold training.
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

The short recovery jobs `38274206` and `38276171` completed in 56m04s and 57m42s, respectively. They verified finite gradients on all eight ranks, saved complete checkpoints at 7,020 and 7,040, and successfully continued the model/optimizer/scheduler state. Sampler restoration took approximately 51.4 and 53.1 minutes.

The following job, `38279510`, timed out after 3h45m29s without a training log or recorded training step. Its node was subsequently observed flagged for network diagnostics. That supports investigating a startup or node-access problem; it does not establish the exact cause.

Retry `38328460` kept the frozen training code and configuration and resumed from 7,040 on another node. Its launch explicitly excludes the flagged node. A separately recorded submission wrapper passes the same explicit exclusion to successor jobs; `38328494` is prequeued behind the retry and its exclusion was verified. That retry subsequently completed. Later chunks reached a saved checkpoint at update 9,000, as recorded below.

The recovery sequence is update 7,000 → 7,020 → 7,040 → 7,500, then 500-update chunks through 20,000. The original recovery plan also used 500-update Gold chunks through 2,000; Gold is now held under the November 16 decision. Both short recovery jobs must finish within 60 minutes before allowing their successors. All GPU allocations remain below the four-hour partition limit, at 3h45m. Each successful job submits only its next dependent job, avoiding the earlier queue-submission limit. Training and smoke phases use eight GPUs; standalone evaluation uses one.

As of that November 10 recovery, the original 20,000/2,000-update scheduler horizons, source weights, candidate checkpoints and development selection rule were unchanged. The November 16 scope below holds Gold and evaluates Silver. No new final WER is available.

## November 16 continuation and evaluation protection

The last completed Silver checkpoint is update 9,000. Its report verifies 9,000 finite-gradient steps on eight ranks, changes to 357 encoder tensors, two projection tensors and 99 LoRA B tensors, and a complete distributed checkpoint. A subsequent restore recovered 733 optimizer-state entries and one scheduler state. Job `38345332` then timed out at 3h45m04s. Its log reached update 9,006, but those six additional updates were not checkpointed.

In that attempt, sampler restoration took about 111 minutes. After validation, the log entered another sampler restoration and did not finish it before the lease expired. Inspection of the pinned TorchData implementation identified a double restore: the sampler and its iterator are the same Lhotse object, yet both serialized state payloads are loaded. The first load replays the sampler; the second schedules another replay for the next iterator reset. All eight step-9,000 checkpoint payload pairs are equal. A separately frozen recovery overlay restores an aliased Lhotse sampler only once when the payloads match, and rejects ambiguous unequal states. Other loaders keep their original behavior. Audit job `38636295` passed on all eight simulated ranks in the pinned runtime, reproducing the unpatched duplicate-restore bug and verifying exact continuation batches, token arrays and audio-crop identities for packed and unpacked sampling, epoch transitions without duplicate replay, ambiguous-state rejection and ordinary distinct-iterator loaders. These are data-loader tests; the new eight-GPU model continuation is a separate gate.

The recovery keeps the original model, optimizer, schedule and manifests for Silver. It first continues 9,000 → 9,020 → 9,040, then uses 500-update chunks through 20,000. The first resume still has to replay the legacy sampler once. The second continuation must finish within 60 minutes before the remaining chain is allowed to proceed. Checkpoint integrity, finite gradients, input hashes and eight-rank state remain required gates. Candidate exports and development evaluations use separate allocations, followed by final paired evaluation of the selected Silver checkpoint and the original base. The active 33-job chain contains only Silver tasks and submits no Gold jobs. Training uses eight GPUs; standalone model evaluation uses one. Each allocation is limited to 3h45m and submits a successor only after success.

The November 13 release has 1,503 clips: the original 1,269-row common test plus a 234-row extension drawn from the previous Gold training pool. Those extension clips cannot remain in future Gold training. Exact clip/source matching removes 234 clips / 14.879633 minutes, leaving 159 clips / 10.192967 minutes. Applying the existing airport/date and Gold recording-group separation rule also removes 112 of those 159 clips. The stricter result is 47 clips / 2.722633 minutes. A recording-group match is a conservative exclusion and does not by itself prove identical audio. A versioned derivative run, `nemotron-transcribe-p4-eval-protected-2026-11-16`, preserves the original run and keeps the extension out of adaptation. The completed audit finds zero matches in the 377,400 saved Silver rows and 90,069 English rows against all 1,503 protected release rows using source identity, source bytes, cropped PCM, and available recording-group metadata. The user chose to finish and evaluate Silver and hold Gold until more separate human-labeled data is available. The 47-row derivative is retained as an audited preparation artifact; no Gold model is trained from it. The final comparison still uses the frozen 1,269-row test, with selection on the existing development and English-guard sets; the extension is protected and does not participate in checkpoint selection. Unknown upstream model ancestry remains unresolved.

[Aggregate run record](../reports/nemotron-transcribe-p4-curriculum-2026-11-07.json) records pins, the original job history and recovery status. Source audio, row-level manifests, predictions and checkpoints remain in restricted storage. This run does not publish a model or establish release approval.

References: [P4 lineage](checkpoint-comparison-n4-p4.md#training-lineage-and-selection-rationale), [Gold experiments](experiments/02-gold-refinement.md), [earlier Transcribe pilot](nemotron-transcribe-atc-pilot-2026-11-06.md), [current common-test comparison](evaluation-tracker.md).

Recovery source: [sampler fix and audit](../scripts/nemotron-transcribe-recovery-2026-11-16/README.md).

Later November 16 policy decision: airport/date grouping may be relaxed for future Gold candidates while source identity and audio-overlap exclusions remain required. The [reclaim audit](gold-training-reclaim-audit-2026-11-16.md) identifies 798 candidate clips / 54.46 minutes under this policy. The active Silver inputs and job chain are unchanged; Gold training remains held.
