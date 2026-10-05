# Adapting ASR to Air Traffic Control: What We Learned from Data Scale, Model Choice, and English Replay

*Experiment report · August 2026*

Air traffic control is an unusually demanding speech-recognition domain. The audio is narrow-band and noisy, speakers use accented English, transmissions are short and compressed, and a single error in a callsign, number, runway, or frequency can change the meaning of an utterance.

We set out to answer a practical question: how far can we adapt strong NVIDIA English ASR models to ATC speech while retaining their ability to recognize ordinary English?

The short answer is that domain adaptation works quickly, but specialization is easy to overdo. Ten hours of carefully selected ATC audio captured most of the initial gain. More data and a higher learning rate improved ATC accuracy further, but ATC-only training caused increasingly severe general-English forgetting. Mixing ATC with general English changed the picture: it produced models that were much better on ATC while remaining close to their untouched general-domain baselines.

This article describes the fine-tuning skills that structured the work, the dataset we built, the experiments we ran, and the lessons we would carry into the next iteration.

The complete shareable pipeline for the strongest balanced Nemotron run is available under `experiments/nemotron-mixed-2to1/` in the accompanying GitHub repository. It includes the recipe, Lhotse data YAML, effective configuration, transcript-free training log, and standalone evaluation results.

## The fine-tuning skills behind the study

This work used two complementary agent skills. They are not model checkpoints or training libraries themselves. They are reusable, versioned operating procedures that tell an agent how to scope an ASR adaptation problem, choose the right intervention, configure NeMo correctly, and evaluate the result without losing track of the original goal.

| Skill | Responsibility | Primary output |
|---|---|---|
| `nemotron-asr-finetune` | High-level ASR adaptation orchestrator | A scoped path from the user goal to data, training, evaluation, iteration, and deployment |
| `nemo-speech-asr-finetune` | NeMo Speech training and offline-evaluation specialist | Model-specific recipes, trained `.nemo` artifacts, checkpoint comparisons, and standalone WER results |

### Where the skills live

The orchestration skill is developed in the public [Nemotron Speech skills repository](https://github.com/nvidia-riva/Nemotron-speech-skills), at [`skills/nemotron-asr-finetune`](https://github.com/nvidia-riva/Nemotron-speech-skills/tree/main/skills/nemotron-asr-finetune). It is also mirrored in the broader [`NVIDIA/skills` catalog](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune).

On the system used for these experiments, both skills are installed in the shared per-user agent skill directory, which makes them available across projects rather than only inside this experiment repository:

```text
~/.agents/skills/nemotron-asr-finetune/
~/.agents/skills/nemo-speech-asr-finetune/
```

The `nemo-speech-asr-finetune` execution skill is currently a separately installed companion skill. It is referenced by the orchestrator, but it is not a sibling directory in the public `nvidia-riva/Nemotron-speech-skills` repository as of August 2026. This distinction matters when reproducing the setup: installing only the public orchestrator does not automatically provide the execution skill unless the companion skill is also linked or installed in the agent skill directory.

Each skill is a directory rather than a single prompt. Its `SKILL.md` defines the contract and routing behavior; `references/` contains the detailed stage-specific guidance; and `assets/` can provide reusable templates such as the experiment ledger. The orchestrator also includes path-selection guidance, its sub-skill registry, and evaluation cases.

### What `nemotron-asr-finetune` provides

The orchestration skill begins before any training command is written. Its purpose is to prevent an expensive fine-tune from becoming the default answer to every ASR problem.

Its capabilities include:

- **Problem scoping:** records the target domain or language, representative recognition failures, available real audio and text, latency constraints, deployment target, and the metric that defines success.
- **Baseline and guardrail definition:** requires a target-domain evaluation set and, when broad English behavior must be preserved, a general-domain regression set.
- **Cheapest-sufficient-path selection:** chooses among word boosting, custom vocabulary or pronunciation, an n-gram language model, model fine-tuning, and—only as a last resort—training or cross-language transfer from scratch.
- **Workflow routing:** hands synthetic text generation to `data-designer`, NeMo training and offline WER to `nemo-speech-asr-finetune`, and Riva/NIM deployment and served-endpoint WER to `nemotron-speech`.
- **Data strategy:** distinguishes real target-domain audio from synthetic or augmented sources, keeps sources separately weighted for ablations, and flags missing representative data.
- **Iteration policy:** compares the result with the stated target, performs error-driven analysis, changes one meaningful lever at a time, and preserves a blind holdout for final claims.
- **Planning support:** estimates required data, GPU-hours, cost, and likely turnaround from a measured pilot rather than inventing a universal hours-to-WER formula.

The customization ladder is an important part of the capability:

| Intervention | Best suited to | Training required? |
|---|---|---:|
| Word boosting | A bounded list of names, commands, or jargon | No |
| Custom vocabulary or pronunciation | Out-of-vocabulary terms or systematic pronunciations | No; build/deploy-time customization |
| N-gram language model | Domain phrasing when text is available but audio is scarce | No acoustic-model training |
| Fine-tuning | Accents, radio channel, noise, acoustic mismatch, and broader domain behavior | Yes |
| Cross-language transfer or training from scratch | A language or script without a suitable checkpoint | Yes; highest data and compute cost |

ATC presented genuine acoustic and linguistic mismatch—radio noise, accents, clipped speech, callsigns, and domain phraseology—so the orchestrator selected fine-tuning. The cheaper techniques remain compatible with a fine-tuned model and could still be added at serving time.

### What `nemo-speech-asr-finetune` provides

Once fine-tuning is selected, the NeMo Speech skill owns the execution details and the offline evaluation contract. Its guidance is organized into five stages.

| Stage | Capabilities |
|---|---|
| Setup and checkpoint selection | Validates the NeMo environment, data access and storage; chooses a checkpoint from language, streaming, latency, memory, alignment, punctuation, and deployment requirements. |
| Data and Lhotse | Audits transcript style; validates manifests; checks duration and token-rate filtering; configures sharding, tarred or non-tarred input, duration buckets, OOMptimizer batch profiles, deterministic sampling, and weighted source blends. |
| Architecture and tokenizer | Detects CTC, RNNT, TDT, hybrid, or AED/Canary models; preserves architecture-specific loss and decoding settings; decides whether to keep, replace, extend, or aggregate tokenizers. |
| Training and selection | Uses step-based optimization, learning-rate warmup and decay, bfloat16 where appropriate, `val_wer` checkpoint monitoring, top-k retention, final export, and optional checkpoint averaging. |
| Evaluation and refinement | Scores saved artifacts with a fixed normalized-WER contract; compares base, final, best-validation, and averaged artifacts; tracks general-domain regression; categorizes errors; and proposes replay, reweighting, targeted data, or lower-LR refinement. |

The skill contains safeguards for failure modes that are easy to miss in long ASR experiments:

- Mixed casing, punctuation, number, symbol, or language conventions are treated as a dataset error before training.
- Lhotse batch modes must not leave conflicting duration, token, or static-batch settings active.
- Duration and token-per-second filters are audited because silently filtered examples can change the effective dataset.
- CTC, RNNT, TDT, hybrid, and AED models receive different scripts, batching profiles, loss settings, and decoding choices.
- The pretrained tokenizer is preserved unless the target language or symbol inventory genuinely requires a change.
- RNNT and TDT recipes avoid fused loss/WER in this workflow and use explicit Lhotse/OOMptimizer batch profiles.
- In-training `val_wer` selects candidates, but it is not reported as the final model result.
- Evaluation uses saved prediction manifests and the same text transform for references and hypotheses.
- The latest checkpoint is not assumed to be the best; final, best-validation, and averaged artifacts are evaluated independently.
- Checkpoint averaging is retained only when it beats the best individual checkpoint.

### How those capabilities shaped our experiments

The skills were not merely documentation consulted after the fact. Their contracts determined several major choices in the study.

| Skill capability | How it appeared in the ATC study |
|---|---|
| Measured pilot before scaling | We began with 10 hours, then built nested 50-, 100-, and 314.7-hour releases. |
| Domain plus general guardrail | Every primary model was evaluated on human-labelled ATC and LibriSpeech `test-clean`. |
| Weighted Lhotse blends | The 1:1, 2:1, and 4:1-style replay experiments changed sampling weights without copying or concatenating audio. |
| Architecture-aware execution | We trained Nemotron streaming RNNT, Parakeet CTC 1.1B, and Parakeet TDT v3 without treating them as interchangeable recipes. |
| Preprocessing and tokenizer preservation | The failed post-hoc normalization probe demonstrated why a model's input contract must be preserved; TDT v3 retained its native per-feature normalization. |
| Best/final/averaged artifact comparison | The TDT run selected step 9,001; the final 20,000-step model and top-five average did not beat it on ATC. |
| Replay as an anti-forgetting intervention | General-English replay converted heavily specialized models into balanced candidates. |
| Blind and decontaminated evaluation | The gold ATC test remained sealed, and the later Jacktol transfer study quarantined confirmed source overlap before evaluation. |

The general-domain guardrail became especially important. If we had monitored only ATC WER, some of our most specialized models would have appeared to be the best results. The paired evaluation showed that they had forgotten much of the behavior that made the base models broadly useful.

## The ATC dataset

The delivered corpus is much larger than the subset used for training. It contains more than four thousand hours of speech signal and millions of automatically transcribed recordings. For controlled fine-tuning, we selected a smaller English-only pool with stricter confidence, duration, text-rate, format, and leakage requirements.

The dataset layers below overlap and should not be added together.

| Dataset layer | Size | Speech or signal hours | Label type | Use in this study |
|---|---:|---:|---|---|
| Complete raw delivery | 3,088,603 recordings | 4,281.9 h | Automatic CNET | Source corpus and profiling |
| Raw English-classified subset | 2,609,290 recordings | 3,631.8 h | Automatic CNET | Candidate pool |
| Full high-quality English selection | 396,461 segments | 314.721 h | Filtered CNET top-1 pseudo-labels | Primary ATC training set |
| Human-accepted gold subset | 2,401 segments | 2.518 h | Human accepted | Evaluation source only |
| Gold development split | 1,007 segments | 1.043 h | Human accepted | Model and checkpoint evaluation |
| Sealed gold test split | 1,394 segments | 1.476 h | Human accepted | Withheld from iterative experimentation |

The 314.7-hour training selection spans nine airports. It is not geographically balanced: Prague contributes 191.75 hours and Bern contributes 88.71 hours, together accounting for roughly 89% of the selected speech. Zurich and Brno contribute another 31.49 hours, while the remaining five airports form a small long tail. Our smaller 10-, 50-, and 100-hour releases were nested and airport-aware, so increasing the scale did not redefine the earlier samples.

### What “high quality” means here

The training set is high-confidence **silver** data, not human ground truth. Its labels are CNET top-1 hypotheses filtered at a mean posterior of at least 0.90. Here, CNET confidence is the automatic recognizer's posterior for its preferred word sequence; it is a useful selection signal, but not a guarantee that a transcript is correct. We also required:

- English classification appropriate to the experiment.
- 16 kHz mono audio.
- Speech segments between 0.5 and 30 seconds.
- A text rate between 1.0 and 5.2 tokens per second.
- Disjoint training and gold-evaluation airport-days.
- Stable manifests and selection order so every run can be reconstructed.

Human-accepted transcripts were reserved for development and the sealed test. An audit of the accepted gold set found no unresolved `[unk]` markers, no development/test overlap, and no overlap with the Full HQ training release at either the exact-ID or airport-day level.

These are objective quality and integrity checks. They do not replace an exhaustive independent listening review, so we do not claim that every difficult callsign, number, or radio transmission is perfectly aligned and transcribed.

For the later external-transfer experiment, we added a stricter acoustic overlap audit against the pinned Jacktol ATC evaluation release. We confirmed 298 shared pairs in the source material and quarantined every affected source recording. This removed 11 ATC training segments and 322 development segments. The resulting transfer experiment used 396,450 ATC training segments (314.716 hours) and a clean 685-utterance ATC development set. Its internal ATC WER is therefore not directly comparable with the earlier 1,007-utterance benchmark.

## A fixed evaluation contract

All headline WER values are standalone evaluations of saved model artifacts, not training-log estimates. The original experiment series uses lowercase, punctuation-insensitive WER. The later transfer study adds Unicode diacritic folding, symbol removal, and whitespace normalization. In each comparison, references and predictions receive the same transform. This focuses the metric on lexical recognition rather than output-style differences.

For the original experiment series, every model was evaluated on:

- The fixed 1,007-utterance human-labelled ATC development set.
- LibriSpeech `test-clean` as the general-English forgetting guardrail.
- A sealed ATC test split that was not used for iterative model selection.

We retained the final model separately from the best-validation checkpoint and evaluated both. Where we averaged top checkpoints, the averaged model was retained only if standalone scoring showed an improvement. Experiment configuration and scalar histories were recorded in Weights & Biases.

## Experiment 1: How much ATC data is enough?

We began with Nemotron 3.5 ASR Streaming 0.6B and a peak learning rate of `3e-5`. The training releases were nested: the 50-hour set contains the 10-hour set, and the 100-hour set contains the 50-hour set.

| Run | Training data | Optimization | ATC dev WER | General WER |
|---|---:|---|---:|---:|
| Untouched Nemotron | 0 h | No adaptation | 75.75% | 3.52% |
| 10-hour pilot | 10 h | 9,791 steps, LR `3e-5` | 45.41% | 5.46% |
| 50-hour run | 50 h | 24,001 steps, LR `3e-5` | 44.21% | 5.77% |
| 100-hour short run | 100 h | 10,000 steps, LR `3e-5` | 46.18% | 5.41% |
| 100-hour extended run | 100 h | 20,000 steps, LR `3e-5` | 44.82% | 5.74% |
| Full HQ | 314.7 h | 62,944 steps, LR `3e-5` | 43.10% | 6.09% |

The first 10 hours reduced ATC WER by 30.34 absolute points, a 40.1% relative error reduction. Moving from 10 to 50 hours produced only another 1.20 points, and the complete 314.7-hour pool improved by 1.11 points over the 50-hour result.

This does not mean that the additional audio had no value. It means that, under this recipe, the model learned the most common ATC vocabulary and phrase structure very quickly and then reached a plateau. The 100-hour comparison also showed that training exposure mattered: 20,000 steps outperformed 10,000 steps on the identical dataset, but still did not beat the 50-hour run.

The general-domain trend was already a warning. Every ATC-only run increased LibriSpeech WER, and the regression grew to 2.57 absolute points on Full HQ.

## Experiment 2: Learning rate and feature normalization

We next tested whether the plateau was an optimization problem.

The intermediate `6e-5` arm completed its 20,000-step screen but was not promoted to standalone evaluation, so we make no final quality claim from its in-training validation metric. The `1e-4` arm was much more decisive: its selected 14,000-step checkpoint reached 32.85% ATC WER, 10.25 points better than the Full HQ `3e-5` model.

The gain came with severe forgetting. General WER rose to 16.88%. A higher learning rate had not merely escaped a local minimum; it had moved the whole model aggressively toward an ATC-specialist solution.

We also tested a proposed per-feature normalization change. Applied post hoc to an already trained Nemotron checkpoint, without updating its weights, it raised ATC WER from 43.10% to 84.41% and general WER from 6.09% to 89.00%. The conclusion is not that per-feature normalization is universally harmful—Parakeet TDT v3 uses it natively—but that preprocessing is part of a model's learned contract. It cannot safely be changed as a checkpoint-only switch.

## Experiment 3: Changing the model architecture

We then fine-tuned Parakeet CTC 1.1B on the same frozen 314.7-hour Full HQ data. This isolated model capacity and architecture from dataset selection.

| Model | Training | ATC dev WER | General WER |
|---|---|---:|---:|
| Untouched Parakeet CTC 1.1B | No adaptation | 65.22% | 2.04% |
| Full HQ final | 88,515 steps, LR `3e-5` | 27.39% | 34.21% |
| Full HQ best-validation checkpoint | Selected from the same run | **26.92%** | 34.63% |

The best-validation checkpoint reduced ATC errors by 58.7% relative to untouched Parakeet and established a new ATC-only frontier for the study. It also suffered catastrophic forgetting: LibriSpeech WER increased by 32.59 absolute points. This was a strong ATC specialist, not a balanced general-purpose model.

## Experiment 4: Replaying general English during ATC adaptation

The forgetting results changed the objective. Instead of asking only for the lowest ATC WER, we asked for the best ATC WER that preserved general English.

We paired the 314.7-hour ATC set with a deterministic 314.7-hour subset of LibriSpeech `train-clean-360`. Both sources were English and used consistent normalized transcript style. Lhotse source weights controlled how often each source appeared without duplicating the audio.

| Run | Sampling mix | Steps and peak LR | ATC dev WER | General WER |
|---|---|---|---:|---:|
| Nemotron replay 1:1 | 50% ATC / 50% English | 40,000, `1e-4` | 37.85% | 3.39% |
| Nemotron replay 2:1 | 66.7% ATC / 33.3% English | 40,000, `1e-4` | 37.38% | 3.39% |
| Parakeet CTC replay 1:1 | 50% ATC / 50% English | 56,250, `1e-5` | **33.34%** | **2.15%** |

The 1:1 Nemotron run removed the measured general-domain regression: it cut ATC WER in half relative to untouched Nemotron while slightly improving LibriSpeech from 3.52% to 3.39%. Changing only the sampler ratio to 2:1 improved ATC by another 0.47 points with effectively unchanged general WER.

Parakeet CTC with equal replay became the strongest balanced model in the original evaluation series. Its selected step-54,002 checkpoint retained nearly all of the untouched model's 2.04% general score while reducing ATC WER from 65.22% to 33.34%. Compared with ATC-only Parakeet, replay gave up 6.42 ATC points but recovered more than 32 points of general-English WER.

This was the clearest experimental result of the project: replay did not merely regularize training a little. It changed an unusably specialized model into a credible balanced model.

## Experiment 5: Parakeet TDT v3 and cross-dataset transfer

Our latest experiment used `nvidia/parakeet-tdt-0.6b-v3`, pinned to a specific model revision. The model uses a TDT decoder and native per-feature normalization. Training combined the decontaminated 314.716-hour ATC release with 314.7 hours of LibriSpeech replay, sampled at 80% ATC and 20% general English. The run used 20,000 steps, a peak LR of `1e-4`, 200 warmup steps, cosine decay, and retained the top five checkpoints plus the final checkpoint.

Checkpoint selection used only the clean internal ATC development set. The pinned Jacktol test was evaluated only after selection and was never used to choose a checkpoint or guide an iteration.

| Artifact | Clean internal ATC dev WER | General WER | Jacktol test WER |
|---|---:|---:|---:|
| Untouched Parakeet TDT v3 | 54.69% | 2.168% | 56.02% |
| Final 20,000-step model | 33.37% | **2.138%** | — |
| Top-five checkpoint average | 33.16% | 2.180% | — |
| Best-validation checkpoint, step 9,001 | **33.15%** | 2.178% | **24.88%** |

Within this decontaminated evaluation, the selected model reduced internal ATC errors by 39.4% and external Jacktol errors by 55.6% relative to its untouched base. General WER changed by only 0.01 percentage points. The top-five average was one ATC word worse than the best individual checkpoint, confirming that checkpoint averaging is an option to test rather than an automatic improvement.

The Jacktol result is especially useful because it measures transfer to a separately released ATC benchmark. It should not, however, be compared directly with the earlier 1,007-utterance ATC development numbers: overlap quarantine changed the internal development set, and the Jacktol corpus has its own acoustic and transcription distribution.

## The experiment ledger at a glance

The main comparable results on the original fixed ATC development set are summarized below.

| Family | Best ATC-only result | Best balanced result | Untouched general WER | Balanced general WER |
|---|---:|---:|---:|---:|
| Nemotron 3.5 ASR Streaming 0.6B | 32.85% | 37.38% | 3.52% | 3.39% |
| Parakeet CTC 1.1B | **26.92%** | **33.34%** | 2.04% | 2.15% |

The lowest ATC-only WER is not the same as the best overall model. For a product that must recognize both ATC and ordinary English, the balanced replay models are the more defensible selections.

## What we learned

### A little well-selected domain data goes a long way

The first 10 hours delivered most of Nemotron's scale-study improvement. This is encouraging for new domains: a carefully selected pilot can reveal whether fine-tuning is promising before committing to the full corpus.

### More hours do not fix an optimization or label ceiling by themselves

The 10-, 50-, 100-, and 314.7-hour results flattened around the mid-40% range under the conservative Nemotron recipe. Additional pseudo-labelled data brought diminishing returns. Data quality, airport balance, optimization, and model architecture mattered more than raw scale after the first domain jump.

### Aggressive adaptation can hide catastrophic forgetting

The higher-learning-rate Nemotron and ATC-only Parakeet results looked excellent if ATC WER was viewed alone. Their general WER exposed the cost. A domain metric without a guardrail is insufficient for selecting a broadly useful ASR model.

### Replay is the strongest anti-forgetting control we tested

Both Nemotron replay ratios and both balanced Parakeet experiments preserved general English while producing large ATC improvements. The exact best ratio remains model-dependent, but the principle was robust.

### Preprocessing belongs to the model contract

Normalization should be preserved from the pretrained model or changed through a deliberate retraining experiment. Mutating it after training can invalidate the learned feature distribution.

### Best validation, final, and averaged checkpoints are different candidates

The latest TDT run selected step 9,001 over the final 20,000-step artifact, while top-five averaging did not improve the selected checkpoint. Artifact selection must be based on standalone evaluation, not filename or training completion alone.

### Leakage auditing can change the benchmark

The Jacktol overlap audit prevented inflated transfer claims, but it also reduced the internal development set. Results remain valid within their stated contracts; numbers from different contracts should not be placed on the same scale curve without qualification.

## Where we would go next

The next experiments should build on the balanced TDT result rather than return to ATC-only scale:

1. Repeat the balanced TDT recipe with multiple seeds to estimate run-to-run variance.
2. Compare 1:1, 2:1, and 4:1 ATC-to-English sampling while holding every other variable fixed.
3. Add airport-aware validation slices so Prague and Bern do not mask long-tail airport behavior.
4. Perform error analysis for callsigns, numbers, runways, frequencies, accents, noise, and clipped transmissions.
5. Test a short lower-LR refinement phase from the best balanced checkpoint, retaining it only if both ATC and general WER improve.
6. Use the sealed ATC test once, after the recipe and model-selection policy are frozen, for the final claim.
7. Add a stratified subjective listening review to distinguish model errors from label or alignment defects in difficult radio segments.

## Conclusion

The experiments began as a dataset-scale study and evolved into a study of specialization versus retention. Fine-tuning consistently taught the models ATC language, but ATC-only optimization made it easy to trade away general English without noticing. Balanced replay was the intervention that reconciled the two objectives.

Our strongest ATC specialist remains Parakeet CTC 1.1B at 26.92% WER on the original development set. Our strongest balanced result on that same benchmark is Parakeet CTC replay at 33.34% ATC and 2.15% general WER. The newer Parakeet TDT v3 experiment adds a promising transfer result—24.88% on the pinned Jacktol test—while preserving its general baseline, under a stricter decontaminated evaluation contract.

The broader lesson is simple: domain WER, general WER, data provenance, and checkpoint selection must be treated as one experiment. Optimizing only the first number produces impressive specialists; optimizing the full contract produces models we can actually trust.
