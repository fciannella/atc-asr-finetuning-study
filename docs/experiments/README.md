# ASR adaptation experiments: a reading guide

This is the results appendix for the fine-tuning blog. It organizes the recorded study by the question each experiment answered, rather than by job submission order. Evidence is frozen through October 7, 2026; the guide was assembled on October 9. No new training or evaluation was performed to write it.

## Read in this order

| Chapter | Question | What it contains |
| --- | --- | --- |
| [Data and evaluation contracts](00-data-and-evaluation.md) | What are we actually comparing? | Silver, Gold, Jacktol, source hours, scoring, and known overlap |
| [1. Silver adaptation](01-silver-adaptation.md) | Are more hours enough? | Scale, learning rate, normalization, CTC, replay, and common-test rescoring |
| [2. Gold refinement](02-gold-refinement.md) | Where should scarce trusted labels enter training? | Parakeet and Nemotron G1-G4, parents, mixtures, and seed repetitions |
| [3. Jacktol and transfer](03-jacktol-and-transfer.md) | Can we approach public results, and do they transfer? | Public checkpoints, curricula, Nemotron extensions, and Gold-only diagnostics |
| [4. Decoding and augmentation](04-decoding-and-augmentation.md) | Can we improve an already adapted model? | Beam search, averaging, seed checks, hard examples, noise, and phrase boosting |
| [5. N-gram language models](05-ngram-language-models.md) | Can training text help without changing acoustic weights? | LM construction, historical beam results, and fresh greedy controls |

Each chapter identifies the starting model, data, optimization or decoding change, measured result, and limits of the conclusion. These are the recorded experimental families, including negative results, not an inventory of every infrastructure retry or unfinished proposal. Detailed stage identifiers remain in the linked [frozen reports](../../reports/README.md).

## Three distinctions that prevent misleading comparisons

1. **Same label, different experiment.** Parakeet CTC 1.1B is not Parakeet TDT 0.6B v3. A Silver-to-Gold checkpoint is not a Jacktol-curriculum checkpoint. An average is not an individual checkpoint.
2. **Same model, different evaluation.** Historical ATC development, the full two-hour ATCO2 Gold test, and the 1.749-hour known-Jacktol-disjoint view are separate metrics. Development WER is not test WER.
3. **Same source hours, different exposure.** Sampling weights control training exposure. A 314.721-hour English pool sampled at 15% is not a 47-hour dataset. LM text weights describe word counts, not audio hours.

All tables use WER percentages; lower is better. "Not measured" is intentionally different from zero or a job waiting to run. Public model-card claims are labeled as reported, not silently treated as local measurements.

## What the study supports

- Broad Silver adaptation helped, but more Silver hours alone produced diminishing returns under the tested low-learning-rate recipe.
- English replay reduced forgetting. The domain score alone was an insufficient selection criterion.
- In both model families, a short Gold-refinement stage after broad adaptation offered a stronger balance than Gold-only adaptation from the pretrained model.
- Jacktol curricula answered a separate matched-data question. Their low Jacktol WER is not interchangeable with ATCO2 performance.
- Decoding and n-gram fusion added measurable, checkpoint-specific gains. The recent Nemotron greedy LM gain was small; significance and serving latency were not established.

These are findings about particular recipes, not proofs that one architecture or data type is universally superior. The small Gold-only arms were historical controlled ablations, not a blanket recommendation to fine-tune production ASR on a few minutes of data.

## Public documentation, not a data release

This repository provides aggregate evidence and selected scripts. It does not distribute licensed recordings, transcripts, model weights, credentials, or private manifests. The planned two-hour evaluation-data release still requires its approved download location and license. The blog's reviewer correspondence is not part of this appendix.
