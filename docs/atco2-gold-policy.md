# ATCO2 gold-only experiment policy

Effective 2026-09-30, all new ATCO2 model-development work uses human-gold ATCO2 supervision only.

## Active data contract

| Split | Inventory | Allowed use |
| --- | --- | --- |
| Gold train | 0.418 h · 393 segments · 19 airport-day groups | Training and training-time augmentation |
| Gold development | 0.100 h · 100 segments | Checkpoint, decoding, and recipe selection |
| Gold community test | 2.000 h · 1,908 segments | Final confirmation after the candidate is frozen |
| General English | 314.721 h source pool; LibriSpeech test-clean evaluation | Optional replay and mandatory forgetting guardrail |

Train, development, and community-test records remain isolated by airport-date, audio path, record ID, and source-record ID.

## Rules

1. Silver ATCO2 labels must not be used in new training, tuning, checkpoint selection, or model-selection decisions.
2. Historical silver releases, checkpoints, W&B runs, and reports remain available read-only for reproducibility.
3. The community test must not be used for training, augmentation design, hyperparameter tuning, checkpoint selection, decoder selection, or repeated iterative feedback.
4. Candidate selection uses gold development WER, LibriSpeech retention, and replication across seeds.
5. General English is not ATCO2 supervision. It may be sampled during training solely to limit catastrophic forgetting.
6. Additional ATCO2 training data must pass human verification and enter through a new, versioned gold release with a fresh leakage audit.

## Current reference point

The strongest locked ATCO2 result in the completed study is 19.08% WER from the seed-42 G3 refinement with beam-4 decoding. Its LibriSpeech test-clean WER is 2.28%.

The development-selected top-three G3 average reached 18.77% on gold development and 19.21% on its single locked-test pass. Because the community test did not select it, that result remains the audited finalist while seed-42 beam-4 remains the best measured locked score.
