# Nemotron N4 and Parakeet P4 checkpoint comparison

Prepared November 2, 2026; updated November 3 with the N4 UWB evaluation. Both checkpoints are uploaded to private repositories in the NVIDIA Hugging Face organization. Their remote file sizes and SHA-256 hashes match the saved upload records. Nemotron N4 is the chosen streaming release direction; Parakeet P4 is the offline alternative. This selection does not establish final technical validation or release approval.

## Checkpoint identities

| Detail | Nemotron N4 | Parakeet P4 |
| --- | --- | --- |
| Inference mode | Native cache-aware streaming RNN-T | Offline TDT |
| Base model | [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Fine-tuned repository | [`nvidia/nv-atc02-finetune-n4`](https://huggingface.co/nvidia/nv-atc02-finetune-n4) | [`nvidia/nv-atc02-finetune-p4`](https://huggingface.co/nvidia/nv-atc02-finetune-p4) |
| Artifact | `nemotron-asr-atc-n4.nemo` | `parakeet-tdt-0.6b-v3-p4.nemo` |
| Size | 2,553,098,240 bytes | 2,509,322,240 bytes |
| Refinement stage | `g3_silver_gold_refine_s42` in the Nemotron campaign | `g3_silver_gold_refine_s42` in the Parakeet campaign |
| Seed / selected refinement step | 42 / 2,000 | 42 / 2,000 |
| Checkpoint selection | Gold development-set WER | Gold development-set WER |
| Model class | `EncDecRNNTBPEModelWithPrompt` | `EncDecRNNTBPEModel` |
| Repository visibility | Private | Private |

The old `nvidia/nv-atc02-finetune-p0` URL redirects to P4. P0 was the pretrained baseline, not the fine-tuned release candidate; its artifact was replaced in that repository.

### Immutable artifact references

| Identifier | Nemotron N4 | Parakeet P4 |
| --- | --- | --- |
| Hugging Face revision | `bb06688439dd09aca9a072a5c0c52e7f2aef07d1` | `4b17d465eb441cc6ddb332a27a340b62aa42d051` |
| Checkpoint SHA-256 | `5f88e503be0985a95b40580159b7465a7b9244e3acf079d2f95c5d246a0274d4` | `974333ba371a4b3a9e2f306edb07bd4881312b49675665d0176410dbde5a42ca` |
| Base artifact pin | Pretrained archive SHA-256: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74` | Upstream repository revision: `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` |

Pinned checkpoint links: [Nemotron N4](https://huggingface.co/nvidia/nv-atc02-finetune-n4/blob/bb06688439dd09aca9a072a5c0c52e7f2aef07d1/nemotron-asr-atc-n4.nemo) and [Parakeet P4](https://huggingface.co/nvidia/nv-atc02-finetune-p4/blob/4b17d465eb441cc6ddb332a27a340b62aa42d051/parakeet-tdt-0.6b-v3-p4.nemo). Access requires permission to the private repositories.

## Evaluation metrics

All values are normalized corpus-level word error rate (WER) percentages; lower is better. These are individual seed-42 checkpoints, not averages across seeds or checkpoints. No external language model was used.

| Evaluation set | Utterances | Nemotron N4 WER | Parakeet P4 WER | Measurement context |
| --- | ---: | ---: | ---: | --- |
| Full ATCO2 Gold test | 1,908 | 22.46% | 20.00% | Historical Gold-refinement comparison; approximately 2.00044 hours |
| LibriSpeech test-clean | 2,620 | 3.63% | 2.31% | General-English retention in the same study comparison |
| Jacktol test | 813 | 28.16% | 21.36% | October 13, 2026 rerun of these exact checkpoint hashes; identical audio and references verified |
| UWB ATC test | 2,822 | 39.01% | 32.23% | N4 evaluated November 3, 2026; P4 historical result. Identical source audio/references verified against the Parakeet test manifest; evaluation use is distinct from training use |

Nemotron N4 additionally scored **21.30%** on the 100-utterance Gold development split (236 word errors / 1,108 reference words). This is a checkpoint-selection metric, not a held-out test result.

### Exact scoring counts where recorded

| Checkpoint | Evaluation | Word errors | Reference words | WER before rounding |
| --- | --- | ---: | ---: | ---: |
| N4 | Full ATCO2 Gold test | 4,925 | 21,930 | 22.457820% |
| N4 | LibriSpeech test-clean | 1,910 | 52,576 | 3.632836% |
| N4 | Jacktol test, October 13 | 2,396 | 8,510 | 28.155112% |
| N4 | UWB test, November 3 | 10,439 | 26,760 | 39.009716% |
| P4 | Jacktol test, October 13 | 1,818 | 8,510 | 21.363102% |

The P4 release evaluation snapshot supplies WER fractions for ATCO2, LibriSpeech, and UWB, but not their edit counts; counts are not inferred here.

### Evaluation conditions and limits

- WER is total substitutions, deletions, and insertions divided by total reference words, rather than an average of utterance WERs. Normalization uses Unicode NFKD, lowercasing, removal of combining marks, symbol replacement while retaining alphanumeric characters and straight apostrophes, and whitespace collapse.
- N4 uses native cache-aware greedy streaming, float32, attention context `[56,3]`, `target_lang=en-US`, and a maximum of 10 symbols per step. Language tags are stripped. The documented configuration uses 320 ms chunks and 240 ms right context; these are configuration values, not measured end-to-end latency.
- The November 3 UWB run verified the uploaded N4 checkpoint hash and reused the pinned evaluator and scorer from the earlier N4 evaluations. All 2,822 utterances (2.433739 hours) were scored; an independent rescore of saved predictions reproduced 10,439 / 26,760 word errors. UWB source audio is 8 kHz mono; the pinned NeMo loader resamples to 16 kHz during inference without changing source files. No training, external LM, or boosting was used.
- P4 uses greedy TDT decoding. The October 13 Jacktol rerun used bfloat16, batch size 16, and CUDA graph decoding. The ATCO2 and LibriSpeech figures above are the historical acoustic-model results, not a new November rerun.
- The models share evaluation targets and lexical scoring rules, but use architecture-specific streaming/offline inference. This is not a matched latency, throughput, or serving-cost comparison.
- The full ATCO2 test includes 213 segments linked to known Jacktol source recordings. It differs from the 1,695-segment recording-disjoint view used in later experiments. Do not substitute those later results or claim universal absence of training overlap from this table. These benchmarks have been examined repeatedly.

## Training lineage and selection rationale

Both candidates follow broad ATCO2 adaptation and then human-Gold refinement:

| Stage | Nemotron N4 | Parakeet P4 |
| --- | --- | --- |
| Broad adaptation | N0 → N3: ATCO2 Silver + English replay, sampled 80/20 | P0 → P3: ATCO2 Silver + English replay, sampled 80/20; parent selected at step 9,001 |
| Gold refinement | N3 → N4: Gold + English replay, sampled 85/15 | P3 → P4: Gold + English replay, sampled 85/15 |
| Refinement settings | 2,000-step budget, seed 42, peak learning rate `1e-5` | 2,000-step budget, seed 42, peak learning rate `1e-5` |
| Available training pools | 314.716 h Silver; 0.418 h Gold; 314.721 h English | 314.716 h Silver; 0.418 h Gold; 314.721 h English |

Sampling weights govern source draws, not percentages of unique audio hours. Training pools repeat as needed. Silver references are supplied machine/CNET transcripts; Gold references are human-accepted. These models should therefore not be described as trained exclusively on human-transcribed data.

The documented adaptation paths exclude the separate Jacktol/UWB curriculum and do not include an external n-gram language model. The extra approximately 5.9 hours of Jacktol training belongs to that separate curriculum, not these N4/P4 recipes. This is a statement about the documented adaptation stages, not certification of the base models' complete pretraining provenance or all dataset-use rights.

For Nemotron, the documented pretrained control scored 77.00% ATCO2 WER / 3.50% English WER, and the Silver parent scored 36.52% / 3.48%. N4 improved ATCO2 to 22.46% while retaining English at 3.63%; it also had the lowest Gold development WER among the tabulated Nemotron recipes. It is the preferred streaming release direction. P4 has lower WER on all four shared test sets above and remains the offline alternative. The preference for N4 reflects the streaming objective and release path, not superior WER to P4.

## Release status

Both Hugging Face repositories contain a checkpoint, checksum file, model card, and evaluation metadata. Private staging is complete. Final engineering/Myungjong validation, full lineage and license review, and public-release approvals remain to be recorded. Repository license metadata must not be treated as final approval of the adapted model's release terms.

The study documentation repository is [fciannella/atc-asr-finetuning-study](https://github.com/fciannella/atc-asr-finetuning-study), a public repository under the personal account, not the NVIDIA organization. Publication of this comparison does not make the private model artifacts publicly accessible.

## Evidence

- [Gold-refinement experiment comparison](experiments/02-gold-refinement.md).
- [Nemotron campaign results](../reports/nemotron-comparisons-2026-10-07.json), campaign `nemotron-gold-community-ablation-v1`, stage `g3_silver_gold_refine_s42`.
- [Parakeet experiment snapshot](../reports/experiment-study.json), `gold_experiments`, stage `g3_silver_gold_refine_s42`.
- Exact-checkpoint Jacktol rerun, October 13, 2026, job `36523941`: local supporting report `release-p4-n4-jacktol-rerun-2026-10-13.json`. Its aggregate results are reproduced above; the underlying report is not published in this repository.
- [N4 UWB evaluation and verification](../reports/release-n4-uwb-evaluation-2026-11-03.json), November 3, 2026, job `37896018`: aggregate metrics, checkpoint/manifest/evaluator/prediction hashes, inference settings, and independent-rescore status. Raw audio and prediction transcripts are not included.
- Pinned Hugging Face evaluation metadata: [N4](https://huggingface.co/nvidia/nv-atc02-finetune-n4/blob/bb06688439dd09aca9a072a5c0c52e7f2aef07d1/evaluation.json), [P4](https://huggingface.co/nvidia/nv-atc02-finetune-p4/blob/4b17d465eb441cc6ddb332a27a340b62aa42d051/evaluation.json).
