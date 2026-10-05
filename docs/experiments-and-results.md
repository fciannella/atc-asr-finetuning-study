# What We Learned Fine-Tuning ASR for Air-Traffic Control

## Experiments and results from Silver adaptation to Gold refinement

This is the results companion to [How to Fine-Tune ASR for a New Domain](fine-tuning-guide.md). The first article explains the model families, fine-tuning skills, data roles, and evaluation workflow. This article follows the experiments in the order in which they answered our questions.

The study was not a single training run. It progressed through five phases:

1. Measure how Nemotron ASR responds to increasing amounts of Silver ATCO2 data.
2. Test optimization, model architecture, and general-English replay.
3. Introduce the small, isolated human-Gold ATCO2 training split.
4. Reproduce the public Jacktol result and measure cross-corpus transfer.
5. Improve the frozen acoustic model through decoding and n-gram language models.

The main lesson is that the data role and training sequence mattered as much as the number of hours.

## How to read the numbers

All WER values in this article are standalone, normalized word error rates. Lower is better. References and hypotheses are lowercased, Unicode diacritics are folded, punctuation and symbols are removed, and whitespace is normalized before scoring.

Four evaluation sets appear repeatedly:

| Evaluation set | Role |
| --- | --- |
| ATCO2 locked community test | Primary target-domain test: 2.000 hours, 1,908 human-Gold segments |
| Jacktol test | Public ATC comparison: 0.762 hours, 813 segments |
| UWB ATC test | Independent University of West Bohemia ATC transfer guardrail: 2.434 hours, 2,822 segments |
| LibriSpeech test-clean | General-English forgetting guardrail: 5.403 hours, 2,620 segments |

Development data selected checkpoints, decoder settings, and language-model weights. The locked ATCO2 test was not used for those decisions. Some early experiments originally used a historical ATC development set; we later re-evaluated all surviving Silver checkpoints on the same locked ATCO2 Gold test so that the retrospective comparison below has one consistent target metric.

## Phase 1: How much Silver ATCO2 data was useful?

We started from `nemotron-3.5-asr-streaming-0.6b.nemo`. The Silver releases were nested: the 50-hour release contained the 10-hour release, the 100-hour release contained the 50-hour release, and the Full HQ release contained 314.721 hours and 396,461 confidence-filtered CNET transcripts.

The initial recipe used a peak learning rate of `3e-5`.

| Training release | Steps | ATCO2 Gold test WER | LibriSpeech WER |
| --- | ---: | ---: | ---: |
| Untouched Nemotron | 0 | 75.57% | 3.52% |
| 10 h Silver | 9,791 | 44.02% | 5.46% |
| 50 h Silver | 24,001 | 42.71% | 5.77% |
| 100 h Silver | 10,000 | 44.77% | 5.41% |
| 100 h Silver | 20,000 | 43.63% | 5.74% |
| 314.7 h Full HQ Silver | 62,944 | 41.79% | 6.09% |

The first ten hours produced the decisive jump: 31.55 absolute WER points on the current locked test. Adding another 304.7 hours improved the result by only 2.23 more points under the same conservative recipe.

The 100-hour comparison also showed why “hours” cannot be interpreted without the optimization budget. At 10,000 steps, 100 hours underperformed the 50-hour run. Doubling the horizon recovered most of the regression, but still did not beat the Full HQ result.

**Finding:** the model learned the vocabulary, phrase structure, and radio domain quickly. After that first adaptation, label quality, optimization, source balance, and model architecture were stronger levers than raw Silver volume.

## Phase 2: Optimization, architecture, and forgetting

### Learning rate unlocked specialization—with a cost

The Nemotron Full HQ learning-rate screen changed the result more than the scale study. Raising the peak learning rate from `3e-5` to `1e-4` and selecting the 14,000-step checkpoint reduced locked ATCO2 WER from 41.79% to 32.12%.

However, LibriSpeech WER rose from the untouched model’s 3.52% to 16.88%. The same setting that made the optimizer move decisively into the ATC domain also caused severe general-domain forgetting.

The intermediate `6e-5` arm completed and reached 35.04% best in-training validation WER near 19,000 steps. It was not promoted to standalone scoring after losing the selection screen, so it should not be presented as a final ATCO2 or LibriSpeech result.

### Feature normalization was not a post-hoc switch

We cloned the completed Nemotron checkpoint and changed global feature normalization to per-feature normalization without retraining. ATC WER rose to 84.41% and LibriSpeech WER to 89.00%.

This does not show that per-feature normalization is inherently bad—Parakeet TDT uses it natively. It shows that preprocessing is part of the checkpoint’s learned contract. Changing it after training invalidates the model.

### Architecture materially changed the frontier

We next trained Parakeet CTC 1.1B on the same 314.7-hour Silver release.

| Model | Training | ATCO2 Gold test WER | LibriSpeech WER |
| --- | --- | ---: | ---: |
| Parakeet CTC 1.1B base | None | 63.51% | 2.04% |
| Parakeet CTC 1.1B | 314.7 h Silver | 26.78% | 34.63% |

The ATC-only Parakeet model was far stronger than the initial Nemotron scale runs, but it was an ATC specialist rather than a balanced ASR model. Its general-English regression was catastrophic.

### General-English replay controlled forgetting

We paired the 314.721-hour Silver ATCO2 pool with a deterministic 314.721-hour LibriSpeech pool. Sampling weights controlled how frequently each source appeared; they did not imply that all source audio was consumed equally in every epoch.

| Experiment | Source pools and sampling | ATCO2 Gold test WER | LibriSpeech WER |
| --- | --- | ---: | ---: |
| Nemotron replay 1:1 | 314.721 h ATCO2 + 314.721 h English; 50% / 50% | 36.55% | 3.39% |
| Nemotron replay 2:1 | Same pools; 66.7% / 33.3% | 35.96% | 3.39% |
| Parakeet CTC replay 1:1 | Same pools; 50% / 50% | 32.81% | 2.15% |
| Parakeet TDT Silver control | 314.716 h audited ATCO2 + 314.721 h English; 80% / 20% | 30.28% | 2.18% |

Replay recovered almost all general-English quality while retaining substantial ATC improvement. The Parakeet TDT run became the Silver control and the parent checkpoint for the Gold experiments.

**Finding:** there was no useful single-axis leaderboard. The ATC-only CTC model had the best Silver ATC number, while the TDT control was a much better starting point for continued work because it preserved English and supported clean Gold refinement.

## Phase 3: What did 25 minutes of human-Gold training add?

We isolated the human-Gold ATCO2 data into non-overlapping roles:

- Training: 0.418 hours, 393 segments, 19 airport-day groups.
- Development: 0.100 hours, 100 segments.
- Locked community test: 2.000 hours, 1,908 segments.

Train, development, and test had zero overlap by airport-date, audio path, record ID, and source-record ID. Only the 0.418-hour training split supplied Gold supervision.

We then ran a controlled G1–G4 ablation with Parakeet TDT 0.6B v3. Each Gold refinement used 2,000 steps, peak LR `1e-5`, and the same scoring contract.

| Run | Starting checkpoint | Training composition | ATCO2 WER | Jacktol WER | UWB WER | LibriSpeech WER |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| G1, seed 1234 | Untouched Parakeet TDT | 0.418 h Gold only | 26.44% | 32.56% | 43.29% | 10.91% |
| G2, seed 1234 | Untouched Parakeet TDT | 0.418 h Gold + 314.721 h English pools; 85% / 15% | 24.30% | 30.40% | 41.38% | 2.67% |
| G3, seed 1234 | Silver-adapted TDT control | Same Gold + English pools; 85% / 15% | 20.51% | 21.56% | 32.51% | 2.41% |
| G3, seed 42 | Silver-adapted TDT control | Same Gold + English pools; 85% / 15% | **20.00%** | **21.36%** | **32.23%** | 2.31% |
| G4, seed 1234 | Silver-adapted TDT control | 314.716 h Silver + 0.418 h Gold + 314.721 h English; 70% / 15% / 15% | 27.45% | 23.17% | 34.00% | 2.38% |
| G4, seed 42 | Silver-adapted TDT control | Same three pools; 70% / 15% / 15% | 27.38% | 23.08% | 33.73% | **2.24%** |

G1 proved that even 25 minutes of matching Gold data could strongly adapt the base model. It also showed why a domain number cannot stand alone: LibriSpeech degraded to 10.91%.

G2 added English replay and recovered most general quality, but its cross-ATC results remained weak. G3 was the important result. Broad Silver adaptation first, followed by a small low-LR Gold correction with English replay, reduced the Silver control from 30.28% to 20.00% on the locked test while holding LibriSpeech near the 2.18% control.

G4 continued sampling Silver during the Gold correction and performed much worse. The abundant pseudo-labels diluted the scarce human signal. The model benefited from Silver as an earlier domain-learning stage, not as a dominant source during the final correction.

**Finding:** Gold was most valuable as a precise second-stage correction after broad domain adaptation.

## Phase 4: Refining the G3 model

### Beam search provided a statistically supported gain

The G3 seed-42 acoustic model scored 20.00% with greedy decoding. Beam width 4 reduced the locked ATCO2 result to **19.08%** and left LibriSpeech effectively unchanged at 2.28%.

A paired 5,000-sample bootstrap estimated the improvement at 0.92 WER points, with a 95% interval from 0.68 to 1.15 points. No resample favored the greedy control. This was a real decoder gain, not ordinary measurement noise.

### Checkpoint averaging helped development, but not the locked test

The top three G3 seed-42 validation checkpoints were averaged before the test was opened. The averaged artifact scored 18.77% on Gold development and became the development-selected finalist. Its final standalone results were:

| Evaluation | WER |
| --- | ---: |
| ATCO2 locked community test | 19.21% |
| Jacktol test | 20.36% |
| UWB test | 31.23% |
| LibriSpeech test-clean | 2.32% |

The average generalized well, but it finished 0.13 points behind the single seed-42 beam-4 checkpoint on ATCO2. Averaging was useful, not automatically superior.

### The recipe reproduced across seeds

Additional G3 runs scored 19.58% for seed 17 and 19.04% for seed 73 on Gold development with beam-4 decoding. Together with seeds 42 and 1234, these runs supported a recipe effect rather than a lucky initialization. The extra seeds were development confirmations; they were not used to repeatedly probe the locked test.

### Plausible refinements did not all help

| Refinement | Gold development WER | Outcome |
| --- | ---: | --- |
| Top-three G3 average, beam 4 | **18.77%** | Development-selected control |
| Seed 17 reproduction | 19.58% | Reproduced the recipe, but did not win selection |
| Seed 73 reproduction | 19.04% | Close to the frontier, but did not beat averaging |
| Hard-slice reweighting | 20.40% | Over-emphasizing difficult Gold examples hurt overall accuracy |
| Mild gain + white-noise proxy | 19.68% | Safe, but generic noise did not reproduce the remaining radio errors |
| Phrase boosting | 19.40% or worse | MALSD and positive boost weights lost to ordinary beam 4 |

The negative results were informative. Difficulty sampling, generic noise, and a list of 200 training-derived phrases were not the missing final-mile lever.

## Phase 5: Reproducing the public Jacktol result

The public `qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC` checkpoint reports 5.99% Jacktol WER. Using our local normalized scorer, the unchanged checkpoint reached 6.06%, which closely reproduced the published claim.

Our first flat emulation—20,000 steps on Jacktol plus English—failed at 55.86% Jacktol WER. The failure revealed that matching a model name, dataset, and learning rate was not enough. The successful recipe was a staged curriculum using Jacktol, UWB, and English before target-specific polishing.

| Stage | Training mixture | Steps / LR | Jacktol test | UWB test | LibriSpeech |
| --- | --- | --- | ---: | ---: | ---: |
| A1 | 5.896 h Jacktol + 10.534 h UWB + 100.344 h English; 30% / 30% / 40% | 14,465 / `3e-5` | 9.05% | 16.57% | 3.98% |
| A2 | Same pools; 40% / 35% / 25% | 23,144 / `2e-5` | 6.56% | **12.76%** | **4.11%** |
| P1 | Jacktol + English; 80% / 20% | 5,000 / `1e-5` | 6.43% | 13.24% | 4.20% |
| P2 | Jacktol + English; 85% / 15% | 4,000 / `2e-5` | **5.93%** | 13.54% | 4.07% |
| P3 | Jacktol + English; 90% / 10% | 3,000 / `1.5e-5` | **5.91%** | 13.66% | 4.10% |

P3 had the lowest Jacktol score, but it corrected only two more words than P2 out of 8,510 and was worse on both transfer guardrails. P2 was therefore the selected balanced endpoint.

This reproduced the public result rather than merely approaching it: 5.93% for selected P2 and 5.91% for P3, compared with 5.99% reported publicly and 6.06% measured locally for the public checkpoint.

### Cross-corpus evaluation explained an apparent mismatch

The public checkpoint scored 15.91% on the full 2.000-hour ATCO2 test. We found confirmed source-recording overlap between Jacktol and part of ATCO2. After removing 213 affected segments, it scored 16.65% on the remaining 1.749 hours. The disjoint score is the cleaner transfer estimate, but it is evaluated on a different subset and must not be placed directly beside full-test numbers without qualification.

Our P2 checkpoint scored 17.79% on the full ATCO2 test. Adding 0.678 hours of leakage-safe ATCO2 Gold to P2’s training mixture improved ATCO2 to **15.44%**, left Jacktol exactly at 5.93%, and changed UWB and LibriSpeech only slightly. A later, more aggressive repair stage regressed Jacktol and UWB and was rejected.

**Finding:** the very low Jacktol result came from matched human labels and a carefully staged curriculum, not from a mysterious model advantage. Cross-dataset evaluation and overlap auditing were necessary to interpret it correctly.

## Phase 6: N-gram language-model fusion

The last phase kept the development-selected G3 acoustic checkpoint frozen and changed only decoding. We trained compact KenLM n-gram models from training text; no development or test transcript entered an LM corpus.

The first LM family used the isolated ATCO2 Gold training transcripts. The second added the official Jacktol training text and balanced the two domain-text sources. Three- and four-gram variants and fusion weights were selected on development data before one frozen locked-test pass.

| Decoder | LM training text | ATCO2 locked WER | Jacktol test WER | LibriSpeech WER |
| --- | --- | ---: | ---: | ---: |
| G3 averaged acoustic model, beam 4 | None | 19.21% | 20.36% | 2.32% |
| Gold-domain 4-gram | ATCO2 Gold train only | 18.12% | — | Guardrail passed |
| Balanced-domain 4-gram, alpha 0.1 | ATCO2 Gold train + Jacktol train, 50.32% / 49.68% text balance | **17.61%** | **18.70%** | **2.43%** |

The Gold-only LM removed another 1.09 absolute ATCO2 points without retraining the acoustic model. Adding Jacktol training text improved ATCO2 further and also improved Jacktol transfer. LibriSpeech moved from 2.32% to 2.43%, remaining close to the acoustic finalist.

A larger fusion weight looked better on ATCO2 development but failed the general-English guardrail. We therefore selected alpha `0.1`, not the development-only minimum. This is precisely why decoding parameters need the same domain-plus-general evaluation contract as acoustic training.

**Finding:** once the acoustic model was strong, a small training-text-only language model produced a larger final-mile gain than the tested generic augmentation and phrase-boosting strategies.

## The headline results, with their contracts

There is no honest single “best model” without naming the evaluation and product objective.

| Claim | Result | Contract |
| --- | ---: | --- |
| Best Silver-only ATCO2 specialist | 26.78% | Parakeet CTC; severe English forgetting |
| Best balanced Silver control | 30.28% ATCO2 / 2.18% LibriSpeech | Parakeet TDT with English replay |
| Best measured acoustic-only ATCO2 result | 19.08% | G3 seed 42, beam 4, locked 2 h Gold test |
| Development-selected averaged acoustic finalist | 19.21% ATCO2 / 2.32% LibriSpeech | Top-three G3 checkpoint average |
| Best ATCO2 result after n-gram fusion | **17.61%** | Frozen averaged G3 + balanced 4-gram, alpha 0.1 |
| Best absolute Jacktol reproduction | 5.91% | P3; slightly weaker transfer guardrails |
| Selected balanced Jacktol reproduction | **5.93%** | P2; only two more errors than P3 |

## What changed our understanding

### 1. More hours were not automatically better

Ten Silver hours captured most of the initial Nemotron gain. Hundreds of additional hours produced diminishing returns until we changed optimization and architecture.

### 2. Learning rate and forgetting had to be optimized together

The higher Nemotron learning rate dramatically improved ATC and dramatically harmed English. The apparent breakthrough was incomplete until the guardrail was measured.

### 3. Replay was a reliable anti-forgetting control

Both Nemotron and Parakeet retained much more general capability when English examples remained in the sampling mixture. Replay ratios were part of the model objective, not just a data-loader detail.

### 4. Silver and Gold served different purposes

Silver data taught broad channel and domain structure. Scarce Gold data was most effective as a later correction. Continuing to replay Silver during that correction weakened the Gold signal.

### 5. Training order mattered

The Jacktol reproduction succeeded only after broad mixed-domain stages followed by narrower polishing. A flat run with similar ingredients failed. A dataset list is not a complete recipe.

### 6. Decoder work was worth doing

Beam search provided a statistically supported gain. Phrase boosting did not help, but n-gram fusion did. “No more acoustic training” did not mean “no more accuracy was available.”

### 7. Negative experiments protected us from false conclusions

Post-hoc normalization, generic radio-proxy noise, hard-slice oversampling, excessive Gold repair, and positive phrase boosting all sounded plausible. Controlled evaluation showed why they should not be promoted.

## The practical recipe that emerged

For a new specialized ASR domain with abundant weak labels and scarce human references, our evidence supports this sequence:

1. Freeze a leakage-safe Gold development and test contract.
2. Measure the untouched pretrained model on both domain and general speech.
3. Use broad Silver data for initial domain and channel adaptation.
4. Include general-domain replay whenever the deployed model must preserve general speech.
5. Refine the best balanced checkpoint on scarce Gold data at a lower learning rate.
6. Repeat the winning recipe across seeds rather than trusting one run.
7. Select and export checkpoints before opening the locked test.
8. Evaluate beam search, averaging, and n-gram fusion as separate controlled stages.
9. Report the domain score, transfer scores, and forgetting guardrail together.

That process is the real result of the study. The final WER matters, but the reusable contribution is knowing which evidence must accompany it—and which experiment should come next.
