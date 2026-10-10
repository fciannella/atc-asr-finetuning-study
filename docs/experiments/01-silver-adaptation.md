# 1. Silver adaptation: scale, optimization, and replay

[Experiment index](README.md) · [Next: Gold refinement](02-gold-refinement.md)

**Question:** Does adding more confidence-filtered ATCO2 training audio keep improving domain recognition, and what happens to general English?

## Scale under a conservative learning rate

Each run started from pretrained `nemotron-3.5-asr-streaming-0.6b.nemo`, not from the previous scale run. The nested Silver releases used machine-generated CNET transcripts. Peak LR was `3e-5`, with cosine decay and a `1e-6` floor. There was no English replay in these runs.

| Silver hours | Executed steps | Schedule horizon / warmup | ATCO2 full test WER | English WER |
| --- | ---: | --- | ---: | ---: |
| 0, pretrained control | 0 | None | 75.57% | 3.52% |
| 10 | 9,791 | 10,000 / 100 | 44.02% | 5.46% |
| 50 | 24,001 | 25,000 / 250 | 42.71% | 5.77% |
| 100 | 10,000 | 10,000 / 100 | 44.77% | 5.41% |
| 100 | 20,000 | 20,000 / 200 | 43.63% | 5.74% |
| 314.721, Full HQ | 62,944 | 62,944 / 629 | 41.79% | 6.09% |

ATCO2 values are the later common-test retrospective, not the original development scores. Frozen exports were re-evaluated without retraining. The first 10 hours accounted for most of the gain under this recipe; the table does not isolate data scale from total optimization exposure.

## Optimization and model family

| Change | Start and training | ATC WER | English WER | Evaluation boundary |
| --- | --- | ---: | ---: | --- |
| Nemotron LR `6e-5` | Pretrained; Full HQ; 20k steps | 35.04% | Not measured | Best **in-training validation**, near 19k; not a final score |
| Nemotron LR `1e-4` | Pretrained; Full HQ; 20k, selected 14k | 32.12% | 16.88% | Full Gold test |
| Per-feature normalization switch | Completed Full HQ export; no retraining | 84.41% | 89.00% | **Historical ATC development**, not full Gold test |
| Parakeet CTC 1.1B control | `nvidia/parakeet-ctc-1.1b`; untouched | 63.51% | 2.04% | Full Gold test |
| Parakeet CTC 1.1B adaptation | Same pretrained model; Full HQ; LR `3e-5`; 88,515 steps, selected 66,002 | 26.78% | 34.63% | Full Gold test |

The `6e-5` run completed but was not promoted to standalone evaluation. Its absent scores are **not pending work**. The normalization probe demonstrates checkpoint/preprocessing incompatibility, not that per-feature normalization is generally harmful.

Higher LR and the CTC recipe gave stronger specialization but much worse English retention. Different architectures, parameter counts, pretrained checkpoints, and recipes changed together, so this is not an architecture-only causal comparison. In later chapters, "Parakeet" means **TDT 0.6B v3**, not this CTC 1.1B model.

## Replay to preserve English

The first three rows use 314.721 h Silver and 314.721 h LibriSpeech source pools. The TDT control uses the 314.716 h audited Silver pool and the same English pool. Ratios below are sampling weights, not unique-hour partitions.

| Model and initial checkpoint | Silver / English | Steps; peak LR | Full Gold test WER | English WER |
| --- | --- | --- | ---: | ---: |
| Pretrained Nemotron streaming 0.6B | 50 / 50 | 40,000; `1e-4` | 36.55% | 3.39% |
| Pretrained Nemotron streaming 0.6B | 66.7 / 33.3 | 40,000; `1e-4` | 35.96% | 3.39% |
| Pretrained Parakeet CTC 1.1B | 50 / 50 | 56,250, selected 54,002; `1e-5` | 32.81% | 2.15% |
| Pretrained `nvidia/parakeet-tdt-0.6b-v3` | 80 / 20 | 20,000, selected 9,001; `1e-4` | 30.28% | 2.18% |

The TDT export became the parent for the Parakeet Gold study. Its historical ATC development WER was 33.15%; 30.28% is its later full-test score. The subsequent Nemotron Gold study prepared its own parent and baseline; those appear in the next chapter rather than replacing these historical controls.

**What we learned:** data volume, optimization, and forgetting are separate questions. A lower ATC number was not sufficient to select a balanced recognizer. Replay recovered general-English performance at some cost to specialization.

## Evidence

[experiment-study.json](../../reports/experiment-study.json): `silver_experiments` holds the 13 original scale/optimization/replay entries; `silver_retrospective.results` holds common-test rescoring and baseline controls. [Retrospective methodology](../silver-gold-retrospective.md) explains audio-offset handling and comparison limits. IDs such as `scale-10h-v1`, `nemotron-fullhq-lr1e4-normna-20k-v1`, and `parakeet-tdt-v3-jacktol-transfer-v1` connect these tables to the original records.
