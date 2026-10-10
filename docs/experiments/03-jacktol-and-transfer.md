# 3. Jacktol: public references, curricula, and transfer

[Experiment index](README.md) · [Next: decoding and augmentation](04-decoding-and-augmentation.md)

**Question:** Can a matched-label recipe approach public Jacktol performance, and what does that imply for other ATC recordings?

This is a separate question from the small-Gold correction experiment. Jacktol provided an accessible public dataset and checkpoint reference, not merely a pipeline smoke test. It also let us compare target-specific training with Silver ATCO2 transfer. The earlier replay trade-off did not, by itself, prove that a particular Jacktol curriculum was necessary.

Jacktol, ATCO2, and the University of West Bohemia (UWB) corpus all contain ATC speech. They are not established to have identical acoustic distributions, and known Jacktol/ATCO2 recording overlap limits transfer claims. See the [data contract](00-data-and-evaluation.md).

## Public and supplied reference checkpoints

| Reference | Jacktol WER, reported | Jacktol WER, local | Evidence |
| --- | ---: | ---: | --- |
| [qenneth Parakeet TDT 0.6B v3](https://huggingface.co/qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC) | 5.99% | 6.06% | Pinned checkpoint `ac5200ee`; 516 errors / 8,510 words locally |
| [Jacktol Whisper Large v3](https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC) | 6.50% | Not measured | Model-card reference only |
| Supplied Parakeet P4 bundle | 6.27% | Not measured | Bundle-reported Jacktol score, not a local reproduction |

Reported values are historical snapshots from the study, not a claim about today's leaderboard. Local reconciliation of the supplied P4 measured 20.06% on **historical ATC development** and 2.22% on English. Those results do not establish its score on the later frozen two-hour ATCO2 test.

## Early comparisons

Both local training arms below started from `nvidia/parakeet-tdt-0.6b-v3`.

| Experiment | Data and optimization | Jacktol test | English |
| --- | --- | ---: | ---: |
| Silver transfer | 314.716 h Silver + 314.721 h English, 80/20; 20k steps, LR `1e-4` | 24.88% | 2.18% |
| Flat Jacktol emulation | 5.896 h Jacktol + 100.344 h English, 80/20; 20k, LR `1e-5`; style-normalized text | 55.86% | 4.05% |

The flat run was a negative result, not an exact implementation of the successful curriculum. Label representation, sources, optimizer schedule, and sequence later changed together; no single cause of its failure was isolated.

## Parakeet's successful staged curriculum

The initial checkpoint was `nvidia/parakeet-tdt-0.6b-v3`, pinned to revision `8844af98`. A1/A2 source pools were **5.896 h Jacktol, 10.534 h UWB, and 100.344 h LibriSpeech**. P1-P3 retained Jacktol and English, without UWB or synthetic audio. Each stage began from the preceding selected checkpoint.

| Stage | Sampling, Jacktol / UWB / English | Steps; peak LR | Jacktol test | UWB test | English |
| --- | --- | --- | ---: | ---: | ---: |
| A1, broad adaptation | 30 / 30 / 40 | 14,465; `3e-5` | 9.05% | 16.57% | 3.98% |
| A2, more ATC emphasis | 40 / 35 / 25 | 23,144; `2e-5` | 6.56% | 12.76% | 4.11% |
| P1, target refinement | 80 / 0 / 20 | 5,000; `1e-5` | 6.43% | 13.24% | 4.20% |
| P2, retained endpoint | 85 / 0 / 15 | 4,000; `2e-5` | 5.93% | 13.54% | 4.07% |
| P3, further specialization | 90 / 0 / 10 | 3,000; `1.5e-5` | 5.91% | 13.66% | 4.10% |

P3 made only two fewer Jacktol errors than P2. We retained P2 as the balanced historical endpoint; that is a retrospective benchmark trade-off, not a claim of statistically significant superiority or a new blind selection. Matching the approximate public WER did **not** reproduce the public model's exact training recipe.

## Nemotron: first extend Jacktol-only training

These runs began with pretrained Nemotron 3.5 ASR Streaming 0.6B, using the 5.896-hour Jacktol training pool and **no English replay**. Later phases extended previous final weights with new optimizer/scheduler segments. Thus, this is not a pure experiment in changing only a single uninterrupted step budget.

| Cumulative training budget | Jacktol test | Full ATCO2 test | Disjoint ATCO2 | English |
| --- | ---: | ---: | ---: | ---: |
| Pretrained control | 72.22% | 77.00% | 77.79% | 3.50% |
| 5k | 14.11% | 29.07% | 29.41% | 6.30% |
| 10k | 11.01% | 24.84% | 25.27% | 6.86% |
| 20k | 9.34% | 21.54% | 22.12% | 7.52% |
| 30k | 8.05% | 19.75% | 20.44% | 7.76% |

The final 10k extension used peak LR `1e-5`, 200 warmup steps, a `1e-6` floor, seed 1234, and a global batch of 16 on eight GPUs in float32. Jacktol validation selected checkpoints. The frozen aggregate report does not contain complete recipes for every earlier extension; do not infer identical schedules from this final phase.

More steps continued to help Jacktol but also continued to degrade English.

## Nemotron: then match the Parakeet curriculum

We next reset to pretrained Nemotron and matched the A1/A2/P1/P2/P3 source pools, sampling weights, step budgets, and peak learning rates above. The experiment retained Nemotron's native streaming RNN-T architecture and preprocessing, rather than replacing them with Parakeet's.

| Stage | Jacktol test | UWB test | Full ATCO2 test | Disjoint ATCO2 | English |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained | 72.22% | 82.53% | 77.00% | 77.79% | 3.50% |
| A1 | 11.03% | 24.86% | 27.73% | 28.18% | 5.18% |
| A2 | 7.90% | 21.15% | 21.71% | 22.37% | 5.20% |
| P1 | 7.57% | 21.10% | 20.72% | 21.38% | 5.19% |
| P2 | 7.53% | 20.99% | 20.04% | 20.79% | 5.27% |
| P3 | 7.31% | 20.68% | 19.22% | 19.96% | 5.30% |

All stages used eight GPUs and global batch 16. Evaluation was native cache-aware greedy streaming, context `[56,3]`, float32, and no external LM. P3 is the **Nemotron LM starting checkpoint**, not the Gold-refined G3 model from the previous chapter.

Important replication limits remain: early historical Parakeet stages used different GPU/accumulation settings; validation cadence was translated to optimizer steps; A1 validated EMA weights but exported ordinary weights, while A2 exported EMA. Historical uppercase Jacktol and lowercase UWB/English labels were retained. Resume segments used deterministic fresh sampler seeds without restoring exact sampler position. These are matched-data curricula, not bitwise reproductions or an architecture-only controlled trial.

## Cross-corpus and Gold-only diagnostics

The unchanged public qenneth checkpoint scored **15.91% full ATCO2** and **16.65% known-recording-disjoint ATCO2**. The latter removes known overlap, not every possible shared source.

Other exploratory results are preserved with their limitations:

| Experiment | Starting point and recipe | Jacktol test | UWB test | English |
| --- | --- | ---: | ---: | ---: |
| All-Gold curiosity run | Pretrained TDT; all 2.518 h ATCO2 Gold; no English; 2k, LR `1e-5` | 35.53% | 46.23% | 11.66% |
| Gold-additive P2 recipe | **P1** checkpoint; 5.896 h Jacktol + 0.678 h older Gold pool + 100.344 h English; 76.89/8.11/15; 4k, LR `2e-5` | 5.93% | 13.70% | 4.08% |
| Gold-repair stage | P2 checkpoint; Jacktol 5.896 h + UWB 10.534 h + Gold 1.043 h + English 100.344 h; 45/25/15/15; 2k, LR `5e-6` | 7.54% | 16.00% | 4.06% |

The curiosity run consumed Gold development and test audio and has cross-corpus overlap risk. It is **not a clean benchmark, an upper bound, or an eligible ATCO2 comparison model**. The Gold-additive recipe starts from P1 to compare against ordinary P2; it is not further fine-tuning of P2 weights. The repair stage was rejected after transfer regression.

The older Gold-extension ledger also records ATCO2 values of 17.79% for P2, 15.44% for the additive arm, and 16.91% for repair. Its entries do not pin the same per-row full-test manifest/error denominator as the newer reports. We preserve these as historical observations, **not as rows in the current two-hour-test leaderboard**. Their larger Gold pools must not be confused with the later 0.418-hour training contract.

**What we learned:** matched training data and recipe details matter. Jacktol progress and ATCO2 progress are related but different objectives. Reproduction requires data, normalization, initialization, sampling, scheduling, and evaluation evidence, not just a model name and LR.

## Evidence

[jacktol-gold-study.json](../../reports/jacktol-gold-study.json): `references`, `reproduction_stages`, `probes`, and `gold_extensions`. [nemotron-comparisons-2026-10-07.json](../../reports/nemotron-comparisons-2026-10-07.json): Jacktol-only/10k/20k/30k campaigns and `nemotron-matched-parakeet-curriculum-v1`. The latter includes per-stage recipes and checkpoint hashes. The common evaluation-membership audit is not a proof of training-data disjointness.
