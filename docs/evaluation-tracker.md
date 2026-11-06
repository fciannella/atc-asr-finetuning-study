# ATC evaluation tracker

Updated **November 6, 2026**. This page links the completed evaluations, evidence, and open work. It is a dated record, not an automated job dashboard.

## Current two-hour comparison

Nine new checkpoint evaluations are complete: eight community checkpoints and our historical Jacktol P2. Existing qenneth, P4 and N4 predictions were identity-verified and rescored, giving **12 compared checkpoints**. This does not mean the entire community inventory has been evaluated.

| Checkpoint | Mode evaluated | Full-test WER | Known-Jacktol-disjoint WER |
|---|---|---:|---:|
| RASR Parakeet v1 | Offline | 12.71% | 13.36% |
| RASR Parakeet v2 | Offline | 14.88% | 15.28% |
| qenneth Parakeet | Offline | 15.91% | 16.65% |
| jlvdoorn Whisper large v3 | Offline | 16.20% | 16.91% |
| Jacktol Whisper large v3 | Offline | 16.30% | 17.57% |
| Our Jacktol-trained P2 | Offline | 18.38% | 18.90% |
| Our P4 Parakeet | Offline | 20.00% | 19.89% |
| Our N4 Nemotron | Native streaming | 22.46% | 22.33% |
| YouTube ATC FastConformer | Offline | 26.75% | 27.10% |
| STEAK synthetic Whisper | Offline | 29.01% | 29.07% |
| US ATC Whisper Turbo v3 | Offline | 66.70% | 66.91% |
| XLS-R ATCOSIM | Offline | 88.64% | 88.91% |

Full test: **1,908 utterances / 21,930 words / 2.000441 hours**. The supplementary view contains **1,695 utterances / 19,077 words / 1.7494 hours**, excluding known Jacktol-linked source recordings. Both columns use the same lexical normalizer and corpus WER formula. Neither the full package being unpublished nor the Jacktol exclusion proves independence from every candidate's training history.

RASR v1 has the lowest observed WER, with unresolved ATCO2 training ancestry. N4 remains the selected streaming release path. P2 has inherited UWB training; it does not satisfy the current requirement to exclude UWB throughout training. US Whisper's digit formatting affects primary WER; its secondary standard-English-normalized full WER is 51.78%, versus 66.70% primary. See the benchmark for that normalization applied uniformly to all 11 systems in the original group. P2 has only the primary metric in this update.

The [training-data map](model-training-data.md) lists the source datasets, label origin, inherited training and evidence gaps for all 12 checkpoints.

## Evidence and completed work

| Work | Status | Record |
|---|---|---|
| Release checkpoint identities and metrics | Documented | [N4/P4 comparison](checkpoint-comparison-n4-p4.md) |
| Community model discovery | 287 screened repositories; 38 detailed profiles | [Assessment](community-atc-model-assessment.md), [inventory](community-atc-model-inventory.md) |
| Eight shortlisted ATC model inference runs | Complete; all 1,908 rows per model | [Benchmark and methods](community-atc-benchmark-2026-11-06.md), [JSON](../reports/community-atc-benchmark-2026-11-06.json), [CSV](../reports/community-atc-benchmark-2026-11-06.csv) |
| Existing qenneth/P4/N4 predictions | Identity checked and rescored | Same benchmark; no new inference claimed for these three |
| Historical Jacktol P2 on current test | Complete; all 1,908 rows | [P2 evaluation](jacktol-p2-current-test-2026-11-06.md), [JSON](../reports/jacktol-p2-current-test-2026-11-06.json) |
| Historical Gold-additive split audit | Complete: older test was 1,394 rows / 1.48h; 554 training segments overlap current test | Same P2 report; additive variant excluded from the current comparison |
| Number-format diagnostic and CTC development check | Complete | Community benchmark; no model tuning or prediction changes |

## Remaining work

| Item | Status / next evidence needed |
|---|---|
| Exact training overlap for RASR, jlvdoorn and other ATCO2 descendants | Unresolved; compare source recordings and segments across their full available training ancestry |
| Broader community candidates | Tier B/C and other inventory entries are not yet inferred; verify usable weights, speech task and provenance before expanding |
| ATC-specific callsign and number accuracy | Not measured; define a shared scoring contract beyond general English normalization |
| Comparable streaming and latency measurements | Not established for community candidates; current community runs are offline and elapsed times are not a controlled latency ranking |
| Release rights and approvals | Benchmark results do not resolve dataset restrictions, teacher/synthetic terms, privacy review or model-release approvals; follow the release review separately |

## How to maintain this record

For each new run, add a dated report with checkpoint identity/hash, dataset manifest hash, decoding settings, complete utterance and reference-word counts, errors and WER, and overlap limitations. Link it here and in the reports index. Keep historical snapshots intact and explicitly document corrections; never substitute a different split without labeling it. Model-card claims and our measured scores remain separate.

Only aggregate results and reproducibility metadata belong in this public repository. Source audio, transcripts, per-utterance predictions, model weights and private storage manifests remain outside it.
