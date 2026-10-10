# Data and evaluation contracts

[Experiment index](README.md)

## Source pools and their roles

| Source | Hours | Records | Intended use |
| --- | ---: | ---: | --- |
| ATCO2 Silver, original Full HQ | 314.721 | 396,461 | Confidence-filtered machine transcripts for broad adaptation |
| ATCO2 Silver, later audited pool | 314.716 | See frozen manifests | Broad adaptation after leakage exclusions |
| ATCO2 Gold training | 0.418 | 393 | Human-accepted supervision |
| ATCO2 Gold development | 0.100 | 100 | Checkpoint and decoder selection |
| ATCO2 Gold community test | 2.000 | 1,908 | Common final ATC comparison |
| Jacktol training | 5.896 | 6,497 | Matched public-domain training |
| Jacktol validation | 0.748 | 812 | Jacktol checkpoint selection |
| Jacktol test | 0.762 | 813 | Public-domain test |
| LibriSpeech large training pool | 314.721 | See frozen manifests | Replay in Silver/Gold studies |
| LibriSpeech clean-100 training pool | 100.344 | See frozen manifests | Replay in Jacktol curricula |
| UWB ATC training pool | 10.534 | See frozen manifests | Additional ATC source in curricula |

UWB means the University of West Bohemia ATC corpus. It is not a model component. Hours describe available source pools, not unique hours consumed in a fixed-step run; duration filters can further restrict eligible examples. The original Full HQ and later audited Silver releases should not be silently treated as byte-identical.

Gold denotes per-record human acceptance in this study, not permission to train on evaluation data. The frozen Gold inventory totals 2.518 hours, with 0.418 hours assigned to training. Older exploratory pools of 0.678 or 1.043 hours predate this contract and are not additional training hours under it.

The Gold train/development/test split was isolated by airport-date, audio path, record ID, and source-record ID. Some historical exploratory runs deliberately broke that boundary; they remain [diagnostics](03-jacktol-and-transfer.md), not eligible benchmark checkpoints.

## ATCO2 and Jacktol are related, not interchangeable

Both contain air traffic control speech. This does not establish that their recording channels, speakers, airports, annotation conventions, or difficulty distributions match. Our [dataset comparison](../atco2-jacktol-comparison.md) records these observable differences:

- Jacktol uses utterance WAV files; ATCO2 uses segments with offsets into recordings. Both evaluated representations are 16 kHz mono.
- Median/P90 duration is 3.05/5.70 seconds for Jacktol versus 3.22/6.43 for ATCO2 Gold.
- The frozen Jacktol manifests do not carry the airport/channel metadata available for ATCO2 Gold, which covers seven airports and nine channel labels.
- Jacktol uses upstream cleaned/manually filtered text without our per-record acceptance flag. ATCO2 Gold retains human re-checker acceptance. Neither fact alone ranks transcript accuracy.

We have **not** established a controlled acoustic comparison of SNR, accents, channel distortion, or callsign distributions. It would be premature to attribute a WER gap to any one of those factors, or to describe Jacktol as fully independent of ATCO2.

The pinned Jacktol revision is `075e736bf8aed80579d829092f74355486b10bc7` of [jacktol/ATC-ASR-Dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset). We do not assume the current upstream contents are identical to this snapshot.

## Evaluation sets

| Evaluation | Records | Hours | Reference words | Role |
| --- | ---: | ---: | ---: | --- |
| ATCO2 Gold development | 100 | 0.100 | 1,108 | Selection, never a final-test score |
| ATCO2 Gold full test | 1,908 | 2.0004 | 21,930 | Primary fixed comparison |
| ATCO2 known-Jacktol-disjoint test | 1,695 | 1.7494 | 19,077 | Transfer view excluding known linked recordings |
| Jacktol test | 813 | 0.7616 | 8,510 | Separate ATC benchmark |
| UWB test | 2,822 | 2.4337 | 26,760 | Additional ATC check, not proven fully cross-corpus isolated |
| LibriSpeech test-clean | 2,620 | 5.4035 | 52,576 | General-English retention |

Transcript candidates followed by acoustic correlation identified shared recording material. Removing 213 ATCO2 test segments connected to known Jacktol recordings leaves the disjoint view above. Of the full test, 197 segments were linked specifically to Jacktol training. Full-test results for Jacktol-trained acoustic models or LMs are consequently overlap-confounded.

This audit removes **known** links; it is not exhaustive acoustic-fingerprint deduplication. Full and disjoint scores have different denominators and should never be compared as if they were a before/after experiment.

## Scoring and selection

Standalone corpus WER is total substitutions, deletions, and insertions divided by total reference words, not an average of per-utterance WER. The [scorer](../../scripts/score_wer.py) lowercases, folds Unicode diacritics, normalizes symbols while retaining ASCII apostrophes, and collapses whitespace. The same operation applies to predictions and references. LM corpus preparation has its own historical punctuation policy.

Gold or Jacktol development selected checkpoints, according to the campaign. Later LM work selected text/order/weight on Gold development and used LibriSpeech as a retention guardrail. Because that guardrail participates in decisions, it is not an untouched general-domain final test.

These benchmarks have been viewed repeatedly. "Locked" describes the within-campaign rule of freezing artifacts before final scoring, not a claim that the study has never seen these test results. Likewise, older and newer evaluator controls must remain separate: the Silver retrospective Nemotron baseline is 75.57% ATCO2 WER, while the later native-streaming campaign baseline is 77.00%. Do not combine one campaign's baseline with another's endpoint.

Evidence: [dataset comparison](../atco2-jacktol-comparison.md), [Gold policy](../atco2-gold-policy.md), [retrospective](../silver-gold-retrospective.md), and [frozen reports](../../reports/README.md).
