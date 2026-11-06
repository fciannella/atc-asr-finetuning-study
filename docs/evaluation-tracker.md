# ATC evaluation tracker

Updated **November 6, 2026**. This page links the completed evaluations, evidence, and open work. It is a dated record, not an automated job dashboard.

## Current two-hour comparison

Nine new checkpoint evaluations are complete: eight community checkpoints and our historical Jacktol P2. Existing qenneth, P4 and N4 predictions were identity-verified and rescored, giving **12 compared checkpoints**. This does not mean the entire community inventory has been evaluated.

| Checkpoint | Mode evaluated | Training data used in adaptation | Evidence / caveat | Full-test WER | Known-Jacktol-disjoint WER | Audited-source-excluded WER |
|---|---|---|---|---:|---:|---:|
| RASR Parakeet v1 | Offline | 200k synthetic RadioTalk utterances (Qwen3 dialogue per dataset metadata + Tada speech; model card conflicts), plus real ATCO2/ATCOSIM training splits | 498 exact test segments in public train sources; 476 eligible under RASR recipe; exact job membership unavailable | 12.71% | 13.36% | 15.68% |
| RASR Parakeet v2 | Offline | Inherits v1; adds 28.8 h / 35,240 TartanAviation US clips with ASR-generated labels + ADS-B correction; ATCO2/ATCOSIM replay | Same 498 train-source matches / 476 recipe-eligible; inherits v1; full ancestry unresolved | 14.88% | 15.28% | 17.33% |
| qenneth Parakeet | Offline | Jacktol ATC-ASR dataset; exact training amount/revision not documented in reviewed card | Publisher-reported; known Jacktol/ATCO2 recording links | 15.91% | 16.65% | 18.18% |
| jlvdoorn Whisper large v3 | Offline | ATCO2-ASR via WhisperATC; project describes an 80/20 train/validation split | 498 exact train-source matches; 496 within first 30 seconds; exact job membership unavailable | 16.20% | 16.91% | 19.82% |
| Jacktol Whisper large v3 | Offline | Jacktol ATC-ASR dataset; exact training amount/revision not documented in reviewed card | Publisher-reported; known Jacktol/ATCO2 recording links | 16.30% | 17.57% | 20.29% |
| Our Jacktol-trained P2 | Offline | 5.896 h Jacktol + 10.534 h UWB + 100.344 h LibriSpeech across its curriculum; final stages use Jacktol/English | Internal records; inherits UWB; known Jacktol overlap | 18.38% | 18.90% | 20.11% |
| Our P4 Parakeet | Offline | 314.716 h ATCO2 Silver + 314.721 h LibriSpeech; then 0.418 h human-accepted Gold + English replay | Internal records; Silver uses supplied machine transcripts; no Jacktol/UWB in documented adaptation | 20.00% | 19.89% | 20.16% |
| Our N4 Nemotron | Native streaming | 314.716 h ATCO2 Silver + 314.721 h LibriSpeech; then 0.418 h human-accepted Gold + English replay | Internal records; Silver uses supplied machine transcripts; no Jacktol/UWB in documented adaptation | 22.46% | 22.33% | 22.32% |
| YouTube ATC FastConformer | Offline | 800+ h reported YouTube VATSIM/IVAO corpus; human simulator speech with ASR/LLM-generated labels | Publisher-reported corpus size; unique training hours not independently counted | 26.75% | 27.10% | 28.50% |
| STEAK synthetic Whisper | Offline | 20k synthetic STEAK v2 ATC utterances; one pass; no real ATC audio reported in fine-tuning | Publisher-reported; text/TTS/noise lineage incomplete | 29.01% | 29.07% | 30.50% |
| US ATC Whisper Turbo v3 | Offline | 1,216 human-transcribed Minneapolis-area radio clips; augmentation expands these to 6,080 files | Publisher-reported; augmented files are not distinct recordings | 66.70% | 66.91% | 69.25% |
| XLS-R ATCOSIM | Offline | ATCOSIM human speech from ATC simulations (Jzuluaga/atcosim_corpus) | Publisher-reported; this is the ATCOSIM-specific checkpoint | 88.64% | 88.91% | 90.39% |

Training-data entries summarize **adaptation and inherited fine-tuning**, not complete base-model pretraining. Hours are source-pool sizes unless noted; augmented files and sampling weights are not additional unique audio. Source links and full lineage are in the [training-data map](model-training-data.md). [Download this combined comparison as CSV](../reports/evaluation-comparison-2026-11-06.csv).

Full test: **1,908 utterances / 21,930 words / 2.000441 hours**. The supplementary view contains **1,695 utterances / 19,077 words / 1.7494 hours**, excluding known Jacktol-linked source recordings. The new audited-source-excluded view retains **1,269 utterances / 13,851 words / 1.262691 hours**, removing the union of known Jacktol links and audited ATCO2 train/validation matches. All three columns use the same lexical normalizer and corpus WER formula. See the [ATCO2 source audit](atco2-training-overlap-audit-2026-11-06.md) for matching methods, two new common masks, intervals and limitations. Neither the full package being unpublished nor the Jacktol exclusion proves independence from every candidate's training history.

RASR v1 has the lowest observed WER on all three common views. Direct ATCO2 source overlap is confirmed; exact checkpoint membership and full inherited ancestry remain unresolved. N4 remains the selected streaming release path. P2 has inherited UWB training; it does not satisfy the current requirement to exclude UWB throughout training. US Whisper's digit formatting affects primary WER; its secondary standard-English-normalized full WER is 51.78%, versus 66.70% primary. See the benchmark for that normalization applied uniformly to all 11 systems in the original group. P2 has only the primary metric in this update.

The [training-data map](model-training-data.md) lists the source datasets, label origin, inherited training and evidence gaps for all 12 checkpoints.

## Where training and test audio overlap

Overlap means that a model's training material and our evaluation material share audio segments or source recordings. Our two-hour package is unpublished, but it is selected from existing ATCO2 recordings. Other datasets can contain those underlying recordings without their authors ever having access to our package.

| Training source compared with our current test | Evidence | Affected checkpoints and interpretation |
|---|---|---|
| **Jacktol dataset ↔ ATCO2 test** | **Confirmed source-recording links:** 213 of the 1,908 test utterances belong to recordings linked to the audited Jacktol dataset. Of those, 197 are linked specifically to its training split. | Applies to our Jacktol-trained P2. Also relevant to qenneth Parakeet and Jacktol Whisper, which report Jacktol training; their exact training dataset revisions and example membership are not independently established. This does not prove that all 213 test utterances were identical training clips for every model. |
| **Older Gold-additive training ↔ current ATCO2 test** | **Confirmed exact segments:** 554 current test segments match that variant's training IDs, resolved source audio paths, offsets and durations. | Affects the older Gold-additive P2 variant, **not the original P2 in the table**. The additive variant is excluded from this comparison. Its historical 15.44% WER used a different 1,394-utterance / 1.48-hour test. |
| **Public ATCO2 train sources ↔ current ATCO2 test** | **Confirmed exact audio:** 498 test segments are sample-identical crops of 316 complete public training recordings. The two public datasets duplicate this ATCO2 component. **347** matches remain after the old Jacktol exclusion. | RASR v1/v2: 476 segments eligible under published duration/text filters. jlvdoorn Whisper: 496 wholly within the first 30 seconds. Exact optimizer consumption is not attested by available manifests. |
| **Public ATCO2 validation sources ↔ current ATCO2 test** | **Confirmed exact audio:** 140 further segments in 83 public validation recordings; 79 remain after the Jacktol exclusion. | These are validation-source matches, not proof of gradient training. They may affect model selection/evaluation independence. RASR recipe admits 126; Whisper windows cover all 140. |

The Jacktol audit first generated candidates from transcript agreement and then confirmed links using acoustic correlation. A **shared source recording** can contain several transmissions: it is a broader criterion than an **identical audio segment**. We conservatively exclude every test utterance associated with a known linked recording, even where its precise segment has not been shown to be a training duplicate. The 197 training-linked utterances are included within the 213 total exclusions; the counts must not be added. The separate 554-segment finding concerns another training recipe and must not be added to the Jacktol count to claim a combined overlap total.

The [ATCO2 source audit](atco2-training-overlap-audit-2026-11-06.md) independently reproduced all **638 unique ATCO2 train/validation matches**. Combining these with the 213 Jacktol exclusions removes **639**, because 212 occur in both groups. This is the explicit union; overlap counts must not simply be added. The wider ancestry review distinguishes direct audio reuse from teacher, synthetic and base-model inheritance.

### What the supplementary columns measure

**Known-Jacktol-disjoint WER** means **WER on the test subset excluding known Jacktol recording overlap**:

- Full-test WER uses **all 1,908 utterances** (2.0004 hours; 21,930 reference words).
- The supplementary score uses the same saved predictions for the **remaining 1,695 utterances**, after excluding 213 (1.7494 hours; 19,077 reference words).
- The same exclusion mask and WER calculation are applied to every model. No retraining or additional inference is involved in producing this second score.
- It addresses the audited Jacktol recording links only. It does **not** establish a universally clean test for RASR, jlvdoorn, or every possible base/teacher training source.

**Audited-source-excluded WER** additionally removes the newly confirmed public ATCO2 train and validation matches, using the same saved predictions and fixed 1,269-utterance mask for every model. This is a post-hoc diagnostic, not a blind benchmark or proof of universal training independence. Differences between columns also reflect changes in test difficulty.

### How this applies to our release candidates

**P4 and N4 follow a separate ATCO2 adaptation path.** Their documented Gold train/development/test contract separates airport-date groups, audio paths, record IDs and source-record IDs. They do not inherit the historical Jacktol/UWB curriculum or the older Gold-additive variant. The overlap findings above therefore must not be attributed to P4/N4 merely because those other experiments were ours. This describes the documented adaptation and Gold split; it is not certification of complete base-model or teacher pretraining independence.

Supporting evidence: [Jacktol recording-overlap audit](atco2-jacktol-comparison.md#cross-dataset-overlap), [exact Gold-additive overlap and P2 rerun](jacktol-p2-current-test-2026-11-06.md#historical-score-correction), [data and split contract](experiments/00-data-and-evaluation.md), and [P4/N4 training lineage](checkpoint-comparison-n4-p4.md#training-lineage-and-selection-rationale).

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
| Direct public ATCO2 audio audit and common-mask rescoring | Complete: 498 train + 140 validation matches; independent WAV verification; all 12 checkpoints rescored | [ATCO2 source audit](atco2-training-overlap-audit-2026-11-06.md), [JSON](../reports/atco2-training-overlap-audit-2026-11-06.json), [CSV](../reports/atco2-training-overlap-audit-2026-11-06.csv) |

## Remaining work

| Item | Status / next evidence needed |
|---|---|
| Full training ancestry for RASR, jlvdoorn and other ATCO2 descendants | Direct public ATCO2 source audit complete. Still need immutable per-checkpoint manifests/revisions, unscanned ATCOSIM/Tartan/synthetic audio comparisons, and unavailable base/teacher histories; see the audit coverage table. |
| Broader community candidates | Tier B/C and other inventory entries are not yet inferred; verify usable weights, speech task and provenance before expanding |
| ATC-specific callsign and number accuracy | Not measured; define a shared scoring contract beyond general English normalization |
| Comparable streaming and latency measurements | Not established for community candidates; current community runs are offline and elapsed times are not a controlled latency ranking |
| Release rights and approvals | Benchmark results do not resolve dataset restrictions, teacher/synthetic terms, privacy review or model-release approvals; follow the release review separately |

## How to maintain this record

For each new run, add a dated report with checkpoint identity/hash, dataset manifest hash, decoding settings, complete utterance and reference-word counts, errors and WER, and overlap limitations. Link it here and in the reports index. Keep historical snapshots intact and explicitly document corrections; never substitute a different split without labeling it. Model-card claims and our measured scores remain separate.

Only aggregate results and reproducibility metadata belong in this public repository. Source audio, transcripts, per-utterance predictions, model weights and private storage manifests remain outside it.
