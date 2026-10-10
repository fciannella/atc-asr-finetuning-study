# 2. Gold refinement: the G1-G4 experiments

[Experiment index](README.md) · [Next: Jacktol and transfer](03-jacktol-and-transfer.md)

**Question:** Is a small, trusted training split more useful on its own, mixed with replay, or after broad domain adaptation?

## Four recipes, two model families

| Recipe | Initial checkpoint | Available pools | Sampling weights |
| --- | --- | --- | --- |
| G1, Gold only | Pretrained model | 0.418 h Gold | Gold 100% |
| G2, Gold with replay | Pretrained model | 0.418 h Gold + 314.721 h English | Gold 85%, English 15% |
| G3, staged refinement | Same-family Silver/English parent | Same pools as G2 | Gold 85%, English 15% |
| G4, continued Silver mixture | Same parent as G3 | 314.716 h Silver + 0.418 h Gold + 314.721 h English | Silver 70%, Gold 15%, English 15% |

Every Gold arm had a 2,000-step budget and peak LR `1e-5`. Gold development selected the exported checkpoints before test scoring. A seed determines stochastic sampling and training choices; it does not identify a different dataset or a model trained from random weights.

Parakeet used `nvidia/parakeet-tdt-0.6b-v3`; its parent was the [80/20 Silver control](01-silver-adaptation.md), selected at step 9,001. Nemotron used pretrained Nemotron 3.5 ASR Streaming 0.6B and a newly prepared same-family 80/20 parent, with 20k steps, peak LR `1e-4`, 200 warmup steps, and `1e-6` floor.

The Nemotron campaign used eight GPUs. Its Gold stages used float32, two examples per GPU, and a 17-second duration limit. The parent used BF16 and 90 seconds nominal global duration batching. These safety settings and native preprocessing differ from Parakeet: this mirrors the experimental questions and mixtures, not every implementation detail.

## Parakeet TDT 0.6B v3

All scores below are standalone greedy WER percentages. ATCO2 is the **full two-hour Gold test**.

| Run | ATCO2 test | Jacktol test | UWB test | English |
| --- | ---: | ---: | ---: | ---: |
| Silver parent | 30.28 | 24.88 | Not tabulated here | 2.18 |
| G1, seed 1234 | 26.44 | 32.56 | 43.29 | 10.91 |
| G2, seed 1234 | 24.30 | 30.40 | 41.38 | 2.67 |
| G3, seed 1234 | 20.51 | 21.56 | 32.51 | 2.41 |
| G3, seed 42 | 20.00 | 21.36 | 32.23 | 2.31 |
| G4, seed 1234 | 27.45 | 23.17 | 34.00 | 2.38 |
| G4, seed 42 | 27.38 | 23.08 | 33.73 | 2.24 |

## Nemotron streaming 0.6B

Evaluation retained native cache-aware greedy streaming, float32, attention context `[56,3]`, and `max_symbols_per_step=10`. These controls belong to this campaign, not the older Silver retrospective.

| Run | Gold development | Full ATCO2 test | English |
| --- | ---: | ---: | ---: |
| Pretrained control | 76.26 | 77.00 | 3.50 |
| New 80/20 Silver parent | 40.61 | 36.52 | 3.48 |
| G1, seed 1234 | 24.64 | 26.09 | 5.16 |
| G2, seed 1234 | 26.17 | 26.89 | 3.72 |
| G3, seed 1234 | 21.66 | 22.97 | 3.68 |
| G3, seed 42 | 21.30 | 22.46 | 3.63 |
| G4, seed 1234 | 32.04 | 30.08 | 3.54 |
| G4, seed 42 | 33.21 | 30.69 | 3.56 |

The blog uses descriptive names for these rows and keeps seed comparisons here. It does not imply the seeds of all controls are matched.

## Interpretation

G1 shows why a small matched-label experiment is interesting, but it is not a safe default production recipe: English forgetting is substantial. G2 tests replay. G3 tests **initialization from a broadly adapted parent**, while G4 tests whether Silver should remain in the final mixture.

G3 had the strongest domain/English balance among these tested recipes in both families. The plausible explanation is that the first stage learns broad domain structure and the second corrects with trusted labels. These experiments do not independently prove that mechanism. G4 also changes effective Gold exposure, so its weaker result is not proof that any Silver example in a final stage is harmful.

The Nemotron G3 checkpoint at 22.46% is **not** the Nemotron Jacktol-curriculum P3 checkpoint used in the LM chapter. Parakeet's individual G3 seed-42 checkpoint **is** the later fresh-greedy LM starting point. Its historical 20.00% score is retained here; its fresh control is reported separately there.

## Evidence

- [Parakeet Gold entries](../../reports/experiment-study.json), `gold_experiments`; original stage IDs `g1_gold_only_s1234`, `g2_gold_replay_s1234`, and G3/G4 seed variants.
- [Nemotron campaign](../../reports/nemotron-comparisons-2026-10-07.json), `nemotron-gold-community-ablation-v1`, includes recipes, checkpoint hashes, evaluator identity, and error counts.
- [Data roles and comparison limits](00-data-and-evaluation.md).
