# ATCO2 Human-Gold and Jacktol Comparison

This snapshot compares the frozen data contracts used in the study. It contains aggregate statistics only; audio and row-level manifests are deliberately excluded.

| Dimension | Jacktol ATC-ASR | ATCO2 human Gold | Interpretation |
| --- | --- | --- | --- |
| Complete inventory | 8,122 clips · 7.405 h | 2,401 segments · 2.518 h | Jacktol is approximately 2.9 times larger by duration. |
| Training allocation | 6,497 clips · 5.896 h | 393 segments · 0.418 h | Most ATCO2 Gold is reserved for evaluation. |
| Evaluation allocation | 0.748 h validation + 0.762 h test | 0.100 h development + 2.000 h community test | ATCO2 prioritizes a substantially larger locked test. |
| Label evidence | Upstream cleaned and manually filtered; no row-level acceptance flag in the frozen manifests | Human re-checker accepted on every included row | ATCO2 provides auditable per-record Gold acceptance. |
| Transcript convention | Uppercase and aggressively spoken-normalized | Lowercase accepted natural spoken form | Normalized WER ignores casing, but lexical normalization still affects training and raw scoring. |
| Median / P90 duration | 3.05 s / 5.70 s | 3.22 s / 6.43 s | ATCO2 has a slightly longer upper tail. |
| Words / vocabulary | 82,765 words · 1,167 types | 27,846 words · 1,084 types | ATCO2 retains nearly as many observed word types despite its smaller duration. |
| Operational metadata | Airport and channel absent from the frozen manifests | 7 airports · 9 channel labels | ATCO2 supports airport and channel slice analysis. |
| Audio representation | Utterance-level 16 kHz mono WAV | Segments within 16 kHz mono source recordings | ATCO2 evaluation must respect segment offsets and end times. |

## Frozen split inventory

| Dataset | Split | Records | Hours | Role |
| --- | --- | ---: | ---: | --- |
| Jacktol | Train | 6,497 | 5.896 | Community-model training |
| Jacktol | Validation | 812 | 0.748 | Checkpoint selection |
| Jacktol | Test | 813 | 0.762 | Final Jacktol evaluation |
| ATCO2 Gold | Train | 393 | 0.418 | Gold-refinement training |
| ATCO2 Gold | Development | 100 | 0.100 | Checkpoint and recipe selection |
| ATCO2 Gold | Community test | 1,908 | 2.000 | Locked final evaluation |

## Cross-dataset overlap

The audit identified 322 ATCO2 Gold segments, totaling 0.364 hours and belonging to 199 source recordings, that are connected to confirmed Jacktol source recordings. Of those, 197 locked-community-test segments totaling 0.236 hours are connected to Jacktol training material.

Candidate pairs were generated from transcript agreement and confirmed using acoustic correlation. Counts across Jacktol splits are not additive because one source recording may correspond to examples in more than one split.

Consequently, cross-corpus transfer claims must report whether they use the full ATCO2 test or the Jacktol-disjoint subset. The study's disjoint ATCO2 view contains 1,695 records and 1.749 hours after removing 213 affected test segments.

## Source

Jacktol dataset: <https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset>, pinned in the study to revision `075e736bf8aed80579d829092f74355486b10bc7`.
