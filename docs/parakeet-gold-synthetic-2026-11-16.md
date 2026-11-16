# Parakeet Gold and synthetic-data experiments

**Observed at 2026-11-16T13:59:00.373229+00:00:** P-G2 job **38639412** is training on **8 GPUs**, at **250 / 2,000** updates with finite gradients on every rank. The save/resume smoke **38639411** passed. Standalone candidate evaluation **38639413** is queued behind successful training. Baseline evaluation **38639106** is complete. No new test-set WER is available.

November 16, 2026. The user authorized stopping Nemotron Transcribe and starting new Parakeet experiments plus development-error-guided synthetic-data research. This record supersedes the unexecuted parts of the earlier plan.

Transcribe job **38636736** was cancelled at 05:21 PST. Its automatic successors are disabled; all saved artifacts remain, with the last complete Silver checkpoint at **9,000 updates**. No Transcribe Gold stage or final evaluation ran.

## Parent eligibility changed the experiment

The proposed P-G3 arm would have refined the old P3 Silver checkpoint selected at step 9,001. Auditing **checkpoint-selection data as well as training data** found the following exact source/audio matches:

| Available P3 ancestry | Rows audited | Protected matches | Result |
|---|---:|---:|---|
| ATCO2 Silver training | 396,450 | 0 | No identified matches |
| English training | 90,070 | 0 | No identified matches |
| Silver validation used for checkpoint selection | 685 | 650 | Parent excluded from this campaign |

The 650 validation matches divide into **528 common-test clips / 30.0561 minutes**, **105 extension clips / 7.3742 minutes**, and **17 development clips / 1.0560 minutes**. All four checks agreed: IDs, source path/basename, complete-file SHA256 and cropped PCM SHA256. Thus **633 frozen evaluation clips / 37.4303 minutes** were in this parent's checkpoint-selection set. These counts are not additional training duplicates and must not be added to the earlier public-training-overlap counts.

P4 inherits this selected P3 parent. Its historical 20.16% common-test WER remains a numerical observation but is **selection-exposed**, not an independently held-out result. P4 additionally trained on all 234 extension clips in its Gold stage. The finding does not establish the same selection exposure for N4 or public checkpoints. The existing benchmark pages now flag this limitation.

**Decision:** withhold P-G3. Start **P-G2 from the original pretrained Parakeet v3**. A future Silver→Gold arm needs a newly trained Silver parent whose selection data also exclude the frozen tests. Full original base-model/teacher ancestry remains unavailable; do not claim universal zero contamination.

## Frozen data and training recipe

| Item | Contract |
|---|---|
| Real Gold | **798 unique clips / 54.4580 minutes**: 639 reclaimed original-test exclusions plus 159 reclaimed former-Gold clips |
| Gold audit | Zero matches against all 1,503 evaluation clips and 100 development clips by source identity, source-file hash and crop PCM; no duplicate PCM; all labels valid; no duration exclusions |
| Airport/date policy | Shared airport/date allowed by user; source/audio exclusions remain |
| Label style | Lowercase lexical labels; no Gold digits; existing labels already match the scorer |
| Evaluation | Frozen 1,269-clip common test plus 234-clip extension; combined 90.6411 minutes; never used to select checkpoints or generation prompts |
| Selection | Existing 100-clip development set plus 128-clip English guard |
| Initialization | `nvidia/parakeet-tdt-0.6b-v3`, revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`; no-fused archive SHA256 `1effc181f031d045e17e2ed6e3d7d22e502e3bacf637c1e023ab1626fdba6460` |
| Compute | **8 OCI GPUs**, FP32, 2 examples/GPU, global batch 16, seed 42 |
| Sampling | Gold 85%, English replay 15%; repeated weighted draws, not unique-hours fractions |
| Schedule | AdamW, LR `1e-5`, 100-step warmup, cosine floor `1e-6`, **2,000 updates** |
| Augmentation | SpecAugment (2 frequency masks / 10 time masks / widths 27 and 0.05); **no effective speed perturbation**; max duration **30 s** (historical Gold used 17 s) |
| Candidates | 250 / 500 / 1,000 / 1,500 / 2,000 updates |
| Selection rule | Lowest standalone lexical development WER among candidates with English guard degradation ≤0.2 percentage points versus the initial base; earlier-step tie break; report failure if none qualifies |
| Final test | Deferred until candidate/recipe selection is locked; no test-based winner selection |

Container `nemotron-asr-finetune-v0.8.sqsh` is pinned: PyTorch 2.7.0a0+79aa17489c.nv25.04, NeMo 2.8.0rc0 at `2d8fcad824680d21cdc316e0b0a7dce119190a15`, Lightning 2.4.0, Lhotse 1.33.0. Code/data/model hashes gate execution. The engineering smoke runs 20 updates, saves, restores optimizer/scheduler/callback state, and continues to 40; its weights are discarded. All-rank finite-gradient checks and encoder/decoder/joint parameter-change checks gate training. This is not a claim of bitwise-identical dataloader replay. The first smoke completed 20 finite updates and saved a checkpoint, but its sparse decoder-weight sample missed changes and failed the final gate. The replacement hashes every trainable parameter in each component. Retry **38639411 passed all 40 updates**, restored 651 optimizer-state entries at step 20, and verified encoder/decoder/joint changes. The pinned Lhotse loader ignored the inherited legacy `augmentor` setting; this is now explicitly **no speed perturbation**. After the smoke, inactive keys were removed and identical SpecAugment settings were assigned to the actual `spec_augmentation` forward-path attribute. Original attempts and code snapshots are preserved.

Jobs stay below the four-hour lease. The launcher uses success dependencies and cancels invalid dependents; it never treats submission as successful training. See the machine-readable report for the latest observed states.

## Baseline development error analysis

The original pretrained checkpoint completed standalone evaluation. These scores are **development/guard metrics**, not leaderboard test WER.

| Split | Clips | Reference words | Word errors | Lexical WER |
|---|---:|---:|---:|---:|
| ATC development | 100 | 1,108 | 563 | **50.81%** |
| English guard | 128 | 2,645 | 69 | **2.61%** |

Development errors: **410 substitutions, 101 deletions, 52 insertions**. Raw case/punctuation-sensitive WER is 73.56%; the primary lexical scorer removes those effects but does not equate digits with spoken number words. Thirteen clips differ in digit-versus-word representation.

| Heuristic token class | Reference words | Attributed errors |
|---|---:|---:|
| Numbers | 281 | 94 |
| Phonetic alphabet | 140 | 81 |
| ATC command terms | 161 | 87 |
| Callsign-related airline terms | 21 | 17 |
| Other | 505 | 284 |

These are mutually exclusive token classes, not verified semantic entities or full callsign accuracy. Insertions are assigned by inserted token. “Other” is broad and needs review; error counts cannot establish noise, accent or clipping causes. The small development set limits targeting confidence. Alignment reconstruction and edit counts passed 1,000 generated correctness cases.

## Synthetic pilot

Data Designer 0.10.0 is installed in an isolated Python 3.12 project environment; Typer is pinned to 0.20.0 because 0.29.1 broke the CLI's `click_type` option. The configured project-local alias `atc-nemotron` uses `nvidia/nemotron-3.5-lightning-30b-a3b` through NVIDIA's API. Credentials are loaded from the existing local secret only into process memory; none are stored in configs or the repository. The served text model revision is not independently pinned by this endpoint; save request settings, generation date and actual outputs.

The first free-form LLM previews were rejected: they mixed unrelated instructions, emitted invalid squawk/frequency patterns and gave poor target-category coverage. Ten free-form audio examples are quarantined. The retained revision uses **Data Designer samplers plus controlled command templates**, with no LLM paraphrasing. It constrains headings to valid ranges, squawks to four octal digits excluding emergency codes, and uses explicitly defined runway/frequency phrases. It is still a research preview, not expert-validated aviation material.

The revised comparison consists of **10-record text previews per arm**, not a large synthetic training corpus:

- Generic ATC: equal category mass over controlled numbers, phonetic taxi instructions, runway instructions, frequency changes and ordinary pilot requests. Command templates determine coherent roles/phases; callsigns are independently sampled.
- Targeted ATC: same schema, with actionable-category masses 94 numbers / 81 phonetic / 87 commands / 17 callsign-related, from aggregate baseline development errors. The broad “other” category is excluded from targeting pending review. No development/test sentences, hypotheses, exact entities or recordings are sent as generation examples.
- Stock-voice Magpie TTS: 16 kHz mono PCM, varied built-in English voices; no voice cloning. Generate paired clean and artificial radio variants (300–3,400 Hz bandpass plus 20 dB SNR Gaussian noise).
- Quality checks: text shape/callsign/length, audio duration/RMS/clipping, full provenance and independent Whisper-large-v3 transcription. Whisper disagreements flag review; numeric formatting and ASR mistakes can make its WER misleading. Listening and aviation-phraseology review remain required before admitting examples to training.

TTS function/version and actual voices are discovered from the authenticated serving API and saved. Reference: [NVIDIA stock-voice discovery](https://docs.nvidia.com/nim/speech/latest/tts/voices.html), [text model/API card](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard). Record the applicable service/model terms with any scaled dataset; preview generation is not release approval.

Both revised previews finished TTS and independent Whisper checks: **20 clean clips / 1.4095 minutes**, plus 20 paired artificial-radio variants. All passed duration/RMS/clipping checks; none is training-eligible yet. Generic preview: 10 clips / 0.7547 min; targeted: 10 / 0.6548 min. These small previews do not yet match total duration or provide balanced category coverage.

Whisper's ordinary lexical disagreement is 50.48% generic and 55.88% targeted, largely because it emits digits. A separate numeric-format diagnostic (digit-by-digit expansion and `niner`→`nine`, without homophone correction) reduces those figures to **3.81% (4/105 words)** and **14.71% (15/102)**. These are synthetic-audio QC diagnostics, **not benchmark model results** and not proof that the generated audio says every label correctly. Numeric/callsign disagreements remain review items; benchmark normalization is unchanged. See the [aggregate preview report](../reports/parakeet-gold-synthetic-2026-11-16-synthetic-preview.json).

After the real-Gold checkpoint is selected, reanalyze its residual development errors before freezing a larger targeted corpus. The proposed controlled training arms are real Gold only, real Gold + generic synthetic, and real Gold + targeted synthetic. Match initialization, update budget, English replay, voice/channel distributions and synthetic audio volume between the two synthetic arms. The proposed fixed-compute mix is Gold 70% / synthetic 15% / English 15%; versus the 85/15 baseline this **replaces some Gold draws**, so the measured effect is a sampling-allocation change, not simply more total training compute. Larger generation and synthetic training have not been launched by the preview.

No synthesized examples are admitted into the training manifests yet. Freeze final candidates before using test results; the repeatedly inspected benchmark is not a new blind holdout. Keep raw licensed audio, references, per-clip predictions and private error-review records out of GitHub.

[Machine-readable status](../reports/parakeet-gold-synthetic-2026-11-16.json) · [Training contract](../reports/parakeet-gold-synthetic-2026-11-16-contract.json) · [Code](../experiments/parakeet-gold-synthetic-2026-11-16/)
