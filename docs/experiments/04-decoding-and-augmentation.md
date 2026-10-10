# 4. Decoding, averaging, and augmentation

[Experiment index](README.md) · [Next: n-gram language models](05-ngram-language-models.md)

**Question:** After a successful Gold recipe, can we improve recognition without simply adding more Silver hours?

The reference is Parakeet TDT 0.6B v3, [Gold-refined G3 seed 42](02-gold-refinement.md). It started from the Silver/English parent and used 0.418 h Gold plus a 314.721 h English pool at 85/15 sampling, 2,000 steps, and peak LR `1e-5`.

## Change decoding, keep weights fixed

| Decoder / checkpoint | Full ATCO2 test WER | English WER | New training |
| --- | ---: | ---: | --- |
| Individual G3, greedy | 20.00% | 2.31% | None |
| Same individual G3, beam width 4 | 19.08% | 2.28% | None |
| Average of top three G3 validation checkpoints, beam 4 | 19.21% | 2.32% | None; weight averaging only |

Beam-4 reduced errors by 0.92 percentage points on this frozen test. A paired 5,000-sample bootstrap gave a 95% interval of **0.68 to 1.15 points improvement**, with no resamples favoring greedy. This uncertainty estimate belongs to this specific decoder comparison; it is not evidence of significance for later LM experiments.

The averaged model was selected on development at 18.77% WER before its final test pass. It scored 20.36% Jacktol, 31.23% UWB, and 2.32% English, but did not beat the individual model on ATCO2. This illustrates why development selection does not guarantee the lowest observed final-test value.

## Reproduction and data refinements

All four new training arms below started from the **Silver parent**, not from the completed G3 checkpoint, and used 2k steps at LR `1e-5`. All used beam-4 for this development comparison.

| Arm | Available pools and sampling | Gold development WER | English WER |
| --- | --- | ---: | ---: |
| G3 seed 17 | Gold 0.418 h / English 314.721 h; 85/15 | 19.58% | 2.41% |
| G3 seed 73 | Same pools and weights | 19.04% | 2.28% |
| Hard-slice reweighting, seed 42 | Hard Gold 0.181 h / remaining Gold 0.237 h / English 314.721 h; 55/30/15 | 20.40% | 2.32% |
| Radio-proxy augmentation, seed 42 | Gold 0.418 h / English 314.721 h; 85/15; mild gain and white noise | 19.68% | 2.30% |
| Averaged G3 reference | No new training | 18.77% | 2.32% |

These are **development results**, not additional locked-test scores. The extra seeds support repeatability on this development set, not a population-level variance estimate. The hard-slice and simple noise recipes did not displace the reference. We did not establish that all augmentation is ineffective, or that white noise faithfully models ATC radio distortion.

## Phrase boosting pilot

We kept the averaged checkpoint fixed and used 200 phrases derived from training text, without adding development/test references. The tested MALSD decoder configurations did not beat ordinary beam-4 on Gold development:

| Decoder | Boost alpha | Development WER |
| --- | ---: | ---: |
| Ordinary beam-4 reference | 0 | 18.77% |
| MALSD | 0 | 19.40% |
| MALSD | 0.1 | 19.40% |
| MALSD | 0.2 | 19.86% |
| MALSD | 0.4 | 20.94% |
| MALSD | 0.8 | 24.73% |

The alpha-zero MALSD control matters: part of the difference comes from changing the decoder, not boosting alone. Positive weights did not justify promotion, so the no-boost beam reference was retained.

**What we learned:** decoder changes can help without acoustic retraining, but plausible refinements still need controls. Averaging, extra seeds, difficulty weighting, noise, and phrase boosting answer different questions; they should not be pooled into a single "augmentation improvement" claim.

## Evidence

[experiment-study.json](../../reports/experiment-study.json), `refinement_experiments` (eight entries) and `refinement_summary`, including final scores, the complete phrase-weight sweep, and paired-bootstrap statistics.
