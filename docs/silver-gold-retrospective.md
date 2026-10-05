# Silver-era checkpoints on the locked ATCO2 gold test

Generated 2026-09-30 from evaluation campaign `silver-gold-retrospective-v1`.

This is a retrospective, evaluation-only comparison. Every checkpoint was frozen before scoring; no retraining, checkpoint selection, or test-driven tuning was performed. WER is normalized by lowercasing, Unicode diacritic folding, punctuation and symbol removal, and whitespace normalization.

## Evaluation sets

| View | Records | Hours | Purpose |
| --- | ---: | ---: | --- |
| ATCO2 locked community test | 1,908 | 2.000 | Primary shared gold benchmark |
| Jacktol-disjoint ATCO2 view | 1,695 | 1.749 | Sensitivity view after removing 213 records / 0.251 h with confirmed Jacktol overlap |

The primary manifest SHA-256 is `4af85f51d32051ee84c841c92c8e5e1eb69ee07201149197f905a7edc9c13082`.

## Results

| Frozen artifact | Historical ATC dev WER | ATCO2 gold 2 h WER | Jacktol-disjoint WER | LibriSpeech test-clean WER |
| --- | ---: | ---: | ---: | ---: |
| Nemotron base | 75.75% | 75.57% | 76.26% | 3.52% |
| Nemotron · 10 h silver | 45.41% | 44.02% | 44.61% | 5.46% |
| Nemotron · 50 h silver | 44.21% | 42.71% | 43.40% | 5.77% |
| Nemotron · 100 h / 10k | 46.18% | 44.77% | 45.38% | 5.41% |
| Nemotron · 100 h / 20k | 44.82% | 43.63% | 44.32% | 5.74% |
| Nemotron · Full HQ / 3e-5 | 43.10% | 41.79% | 42.45% | 6.09% |
| Nemotron · Full HQ / 1e-4 | 32.85% | 32.12% | 32.44% | 16.88% |
| Nemotron · replay 1:1 | 37.85% | 36.55% | 36.96% | 3.39% |
| Nemotron · replay 2:1 | 37.38% | 35.96% | 36.33% | 3.39% |
| Parakeet CTC base | 65.22% | 63.51% | 63.97% | 2.04% |
| Parakeet CTC · Full HQ | 26.92% | **26.78%** | **27.04%** | 34.63% |
| Parakeet CTC · replay 1:1 | 33.34% | 32.81% | 33.47% | 2.15% |
| Parakeet TDT · silver control | 33.15% | 30.28% | 30.71% | 2.18% |

Historical ATC-development values are retained for provenance only: those scores came from earlier development contracts and are not interchangeable with the locked community test.

## Reading the matrix

- Parakeet CTC Full HQ is the strongest silver-only ATC specialist at 26.78%, but its 34.63% LibriSpeech WER makes it unsuitable as a balanced model.
- Parakeet TDT silver control is the strongest balanced silver artifact: 30.28% ATCO2 WER with 2.18% LibriSpeech WER. It is the parent checkpoint used by the later gold-refinement study.
- Within Nemotron, raising the learning rate to `1e-4` improved ATC WER to 32.12% but caused severe forgetting. English replay recovered general-domain performance, with the 2:1 recipe reaching 35.96% ATCO2 and 3.39% LibriSpeech WER.
- Removing confirmed Jacktol overlap changes rankings very little and raises WER modestly for every adapted artifact. The reported ATCO2 gains are therefore not explained by those 213 records.

The normalization preflight is absent because it was not a trained checkpoint. The Nemotron `6e-5` screen is absent because no standalone artifact was available for retrospective scoring; its historical training-validation value remains labeled as such in the main study report.
