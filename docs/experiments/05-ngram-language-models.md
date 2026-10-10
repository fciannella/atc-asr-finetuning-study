# 5. N-gram language models: text-only adaptation

[Experiment index](README.md)

**Question:** With an acoustic checkpoint frozen, can domain text improve its decoder while preserving general English?

An n-gram LM is a statistical model of token history, not a neural LLM. The NeMo path used here is **NGPU-LM**, described in NVIDIA's [GPU n-gram training and deployment tutorial](https://github.com/nvidia-riva/tutorials/blob/main/asr-train-and-deploy-NGPU-LM-for-parakeet-rnnt.ipynb). Our experiments measure inference-time fusion; they do not update acoustic weights or establish serving latency.

## Build the LM from training text

| Text recipe | Unique source material | Normalized words, including repetition |
| --- | --- | ---: |
| Gold | 393 Gold training transcripts, from 0.418 h audio | 4,761 |
| Gold + English | Gold repeated 64 times + 2,192 sampled LibriSpeech training transcripts | 380,880 |
| Jacktol | 6,495 usable training transcripts from the 5.896 h pool; two unknown-marker records excluded | 65,807 |
| Gold + Jacktol | Gold repeated 14 times + Jacktol training text | 132,461 |

Gold + English is 80/20 by normalized **word count**; Gold + Jacktol is 50.32/49.68. Repetition changes statistical weight, not the number of unique sentences or audio hours. No Silver text was added. Designated development/test references were not intentionally included, but known cross-dataset recording overlap remains relevant.

Preparation folds case and diacritics, removes punctuation/symbols, and normalizes whitespace. The historical LM text normalizer removes apostrophes too, unlike the WER scorer's retained ASCII apostrophes; that distinction was kept fixed across the LM comparison.

Each ASR checkpoint's own tokenizer maps text to subword IDs. KenLM estimates 3-gram and 4-gram models; an n-gram here counts **subword tokens**, not necessarily whole words. ARPA and compiled KenLM artifacts are retained, and the decoder loads a GPU-compatible `.bin.nemo` archive. It must match the acoustic tokenizer; Parakeet and Nemotron LM artifacts are not interchangeable.

Conceptually, shallow fusion adds `alpha × LM log-score` to the acoustic-model log-score before a token choice. The compatibility smoke test confirmed that alpha zero reproduced no-LM transcripts exactly on the tested clips. A model-class adapter was needed to build Nemotron's prompted-model subword LMs correctly; failed intermediate artifacts were not used.

## Historical beam-search experiment

This used the **top-three averaged Parakeet G3 checkpoint**, not individual G3 and not a Jacktol-curriculum checkpoint. The following rounded values are retained from the [earlier results narrative](../experiments-and-results.md#phase-6-n-gram-language-model-fusion):

| Beam decoder / text | Full ATCO2 test | Jacktol test | English |
| --- | ---: | ---: | ---: |
| No external LM, beam 4 | 19.21% | 20.36% | 2.32% |
| Gold-only 4-gram | 18.12% | Not recorded in summary | Guardrail passed; exact value not in summary |
| Gold + Jacktol 4-gram, alpha 0.1 | 17.61% | 18.70% | 2.43% |

The text recipe/order/weight screen and English guardrail preceded final evaluation. Full ATCO2 results involving Jacktol text are overlap-confounded; 17.61% must not be presented as a clean disjoint-transfer score. These historical beam results also do not establish what beam would achieve on the later individual-checkpoint greedy controls.

## Current blog comparison: fresh greedy controls

We used two already documented starting points:

- **Parakeet individual Gold-refined G3, seed 42:** the checkpoint from [Gold refinement](02-gold-refinement.md), SHA-256 `974333ba371a4b3a9e2f306edb07bd4881312b49675665d0176410dbde5a42ca`.
- **Nemotron Jacktol-curriculum P3:** the endpoint from [the curriculum chapter](03-jacktol-and-transfer.md), SHA-256 `1c8bcf234013a5248cd10be972c8ff15269fd4c50516c2cc8ae00313ea52fa15`.

Both used genuine `greedy_batch`, not beam width one, batch size eight, and maximum ten symbols per step. Nemotron remained native cache-aware streaming with context `[56,3]` in float32. Parakeet used offline TDT in BF16. Acoustic weights were fixed within each comparison.

### Development selection

For each checkpoint, we ran **44 LM settings plus one no-LM control** on the 100-clip Gold development set:

- Gold family: Gold or Gold + English; orders 3/4; alpha 0.025, 0.05, 0.1, 0.2, 0.4, 0.8 (24 settings).
- Jacktol family: Jacktol or Gold + Jacktol; orders 3/4; alpha 0.025, 0.05, 0.1, 0.2, 0.4 (20 settings).

One winner per family was frozen using development WER and deterministic tie handling. English regression had to stay within **0.20 percentage points** of that checkpoint's fresh no-LM control. All four candidates passed; test scores did not replace the selected candidates. English is a repeatedly used retention guardrail, not an untouched final set.

| Checkpoint | Selected text / order / alpha | Development WER |
| --- | --- | ---: |
| Nemotron P3 | No LM | 17.24% |
| Nemotron P3 | Gold + English / 3 / 0.025 | 17.06% |
| Nemotron P3 | Gold + Jacktol / 3 / 0.025 | 17.06% |
| Individual Parakeet G3 | No LM | 20.31% |
| Individual Parakeet G3 | Gold / 3 / 0.1 | 19.04% |
| Individual Parakeet G3 | Gold + Jacktol / 4 / 0.1 | 18.95% |

### Final greedy results

All entries are WER percentages. Full ATCO2 contains 1,908 segments; disjoint ATCO2 contains 1,695. Compare each LM with its own no-LM row.

| Checkpoint / decoder text | Full ATCO2 | Disjoint ATCO2 | Jacktol test | English |
| --- | ---: | ---: | ---: | ---: |
| Nemotron P3, no LM | 19.22 | 19.96 | 7.33 | 5.30 |
| Nemotron P3, Gold + English | 19.16 | 19.88 | 7.30 | 5.33 |
| Nemotron P3, Gold + Jacktol | 19.15 | 19.86 | 7.29 | 5.33 |
| Individual Parakeet G3, no LM | 19.91 | 19.72 | 21.39 | 2.30 |
| Individual Parakeet G3, Gold | 19.52 | 19.32 | 21.15 | 2.35 |
| Individual Parakeet G3, Gold + Jacktol | 19.31 | 19.14 | 20.56 | 2.33 |

On the disjoint set, the selected Gold + Jacktol LM reduces Parakeet errors from 3,762 to 3,652 out of 19,077 reference words. Nemotron drops from 3,807 to 3,789. This is a modest gain, especially for Nemotron. No significance test or latency benchmark was added in this campaign.

Fresh controls are essential. Parakeet's full-test baseline is 19.91%, versus the historical 20.00%; Nemotron's Jacktol baseline is 7.33%, versus historical 7.31%. We do not assign an unverified cause or subtract new LM results from old controls.

### Earlier greedy campaign, different averaged Parakeet artifact

For completeness, the first greedy campaign used the **averaged** G3 checkpoint and selected different text/weight settings:

| Averaged Parakeet decoder | Full ATCO2 | Disjoint ATCO2 | Jacktol | English |
| --- | ---: | ---: | ---: | ---: |
| No LM | 20.06 | 20.00 | 21.33 | 2.35 |
| Gold 3-gram, alpha 0.2 | 19.44 | 19.28 | 21.22 | 2.53 |
| Jacktol 4-gram, alpha 0.2 | 19.30 | 19.24 | 19.54 | 2.49 |

These remain valid measurements of a different artifact. They were not relabeled as individual seed-42 results. Nemotron P3 measurements were reused from the completed campaign, not presented as an independent repeat.

## Interpretation and next questions

The test establishes compatible greedy n-gram fusion in our evaluated streaming Nemotron pipeline. It does not show that the acoustic models are equally adapted, that one architecture inherently benefits more from LMs, or that every serving stack accepts the artifact unchanged.

Gold-only LM text is particularly small: only 4,761 normalized words. Carefully constrained synthetic **training text** could be a future coverage experiment, using the orchestration skill's Data Designer path. It would need deduplication, factual/domain checks, a separately weighted corpus, and development-only selection. **That experiment was not run here, and no synthetic-text gain is claimed.** More realistic channel augmentation, uncertainty estimates, and matched greedy/beam latency tests are likewise future work, not completed results.

## Evidence and reproducibility references

- [Current blog checkpoint results](../../reports/greedy-ngram-blog-checkpoints-2026-10-07.json): selections, exact errors/denominators, hashes, guardrails, and caveats.
- [Earlier averaged-checkpoint greedy summary](../../reports/greedy-ngram-summary-2026-10-07.json): separate artifact, percentages rather than fractional WER.
- [Historical beam narrative](../experiments-and-results.md#phase-6-n-gram-language-model-fusion): rounded historical evidence; detailed beam sweep is not included in these JSON snapshots.
- [Scoring implementation](../../scripts/score_wer.py) and [NVIDIA NGPU-LM tutorial](https://github.com/nvidia-riva/tutorials/blob/main/asr-train-and-deploy-NGPU-LM-for-parakeet-rnnt.ipynb).

This appendix documents the tested construction and selection rules. Licensed transcripts, LM binaries, and acoustic weights are not distributed by this repository.
