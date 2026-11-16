# Next ATC experiments: close the gap to the public checkpoints

Planning record, November 16, 2026. No new training or inference jobs are launched by this plan. The active Nemotron Transcribe Silver recovery continues under its existing contract. Gold remains held until a new experiment is explicitly started. The user has accepted allowing shared airport/date groups while retaining source-identity and audio-overlap exclusions.

## Measured starting point

The November 6 common-mask comparison contains nine public community checkpoints and three internal checkpoints, all scored on the same 1,269 clips / 75.7615 minutes / 13,851 reference words. These are our measured standings for the pinned checkpoints, not a claim about all current public models. Lower WER is better.

| Rank | Checkpoint | WER | Mode |
|---|---|---:|---|
| 1 | RASR Parakeet v1 | 15.68% | Offline |
| 2 | RASR Parakeet v2 | 17.33% | Offline |
| 3 | qenneth Parakeet | 18.18% | Offline |
| 4 | jlvdoorn Whisper large v3 | 19.82% | Offline |
| 5 | Our Jacktol-trained P2 | 20.11% | Offline; inherits UWB |
| 6 | Our P4 Parakeet | 20.16% | Offline |
| 7 | Jacktol Whisper large v3 | 20.29% | Offline |
| 8 | Our N4 Nemotron | 22.32% | Native streaming |
| 9 | YouTube ATC FastConformer | 28.50% | Offline |
| 10 | STEAK synthetic Whisper | 30.50% | Offline |
| 11 | US ATC Whisper Turbo v3 | 69.25% | Offline |
| 12 | XLS-R ATCOSIM | 90.39% | Offline |

RASR v1 has 2,172 word errors versus P4's 2,792: a gap of 620 errors, 4.48 WER percentage points, or about 22.2% of P4's errors. Beating its observed count requires at most 2,171 errors on this frozen denominator. A numerical lead alone is not a statistically established improvement; compare paired errors with source-recording-clustered bootstrap intervals. The published marginal intervals are 14.70–16.66% for RASR v1 and 19.11–21.25% for P4.

[Standings and counts](../reports/evaluation-comparison-2026-11-06.csv) · [Benchmark protocol](community-atc-benchmark-2026-11-06.md).

## Benchmark and data contract

1. Keep the 1,269-clip common test as the primary historical comparison. All new runs report it unchanged.
2. Establish a separate 1,503-clip / 90.6411-minute comparison for eligible model lineages. Score the 234-clip extension for the nine pinned community checkpoints, reuse original-test predictions only after verifying the matching checkpoint, inference settings and PCM, and recompute combined corpus WER from summed edit/reference counts. Never average the two split WER percentages. Report the extension separately too. A new model can have different rankings on the two views.
3. Existing P4/N4 cannot enter the combined comparison: all 234 extension clips were in their Gold training. For new runs, audit the full adaptation lineage, including the pre-Gold Silver parent, against evaluation and development sources. Unknown base/teacher ancestry remains a limitation for both public and internal models.
4. Prepare a versioned candidate Gold training pool from the 639 excluded original-test clips plus 159 excluded former-Gold clips: 798 clips / 54.4580 minutes before final training-format checks. Source identity, complete-file SHA256 and exact crop-PCM checks against the 1,503 evaluation clips and 100 development clips passed. Permit shared airport/date groups under the newly accepted policy. Finish full candidate-pool deduplication, crop/label validation, duration filtering and source/version recording before training; publish any resulting count changes. Do not use the frozen evaluation recordings or references for augmentation, language-model fitting, checkpoint selection or decoder tuning.
5. Use the existing 100-row Gold development set and English guard for model selection. Record exact checkpoint hashes, manifests, decoder settings and error counts. Freeze each experiment's shortlist before testing; do not select the best checkpoint or seed using the test results. This benchmark has already been inspected repeatedly and is not a new blind test. Any broader generalization claim needs an additional independently collected holdout.

[Reclaimed Gold audit and accepted policy](gold-training-reclaim-audit-2026-11-16.md) · [Frozen evaluation package](atco2-overlap-removed-release-2026-11-13.md).

## Proposed first experiments

These are hypotheses and proposed budgets, not approved jobs or guaranteed gains. Prioritize the offline WER objective; retain N4 as a separate streaming reference.

| Order / ID | Experiment | Starting point | Proposed treatment | What it establishes |
|---|---|---|---|---|
| 0 / B0 | Complete baseline views and verify training eligibility | Nine pinned public checkpoints; audited base and pre-Gold internal parents | Existing fixed decoding; no tuning on test | Leader and gap on both the historical test and the expanded eligible test |
| 1 / P-G3 | Parakeet: Silver then reclaimed Gold | Original P3 Silver parent selected at step 9,001, conditional on full protected-set audit | New eligible Gold 85% / English replay 15%; LR `1e-5`; up to 2,000 steps; seed 42 | Highest-priority reproduction of our historically strongest Gold recipe with the revised training pool |
| 2 / P-G2 | Parakeet: direct Gold control | Pinned pretrained `nvidia/parakeet-tdt-0.6b-v3` | Same data, source weights, LR, step budget and seed as P-G3 | Whether Silver initialization helps with this Gold pool |
| 3 / T-S | Finish current Transcribe Silver experiment | Current step-9,000 resume | Existing 20,000-step schedule; select among 9,000 / 15,000 / 20,000 on development WER plus its frozen English guard | Whether the larger Transcribe architecture is competitive after broad adaptation |
| 4 / T-G3 | Transcribe: reclaimed Gold refinement | Development-selected, verified merged T-S checkpoint | Fresh optimizer/adapters; same proposed 85/15 sources, LR `1e-5`, up to 2,000 updates; retain architecture-specific batching | Whether trusted labels improve Transcribe; do not infer equivalence of compute from matching update counts |
| 5 / R1 | Replicate the strongest development-selected recipe | Same audited initialization as the winning recipe | Second seed, 1234; predeclare how seeds are reported or selected before test | Check that a gain is reproducible rather than a favorable seed |

For proposed Gold runs, save candidates at 250, 500, 1,000, 1,500 and 2,000 updates and select on development WER, with ties favoring earlier steps. This adds early candidates because the revised pool and exposure differ from the original 25-minute Gold experiment. Retain the established per-family precision and batching for the first comparison. Preserve the current Transcribe guard and schedule unchanged; explicitly freeze the Parakeet English guard threshold and reference before launching its new run. No target WER is promised.

Do not initialize an expanded-test experiment from P4/N4 or another checkpoint that already consumed the extension. Removing those clips from a later fine-tuning stage does not undo inherited exposure. If the archived P3 parent fails the full protection audit, use an audited replacement Silver parent or the pretrained control rather than silently accepting leakage.

## Follow-up only if the first experiments leave a gap

- **Radio-domain augmentation / synthetic-data ablation:** RASR v1 uses the same Parakeet v3 base, up to 200,000 RadioTalk examples and upweighted real ATCO2/ATCOSIM. Its pinned recipe sets both real sources to weight 10 and disables its separate noise-augmentation switch. This suggests a testable data/curriculum direction; it does not establish why it wins. Compare the best verified recipe with one controlled augmentation or audited synthetic-data addition, holding the other settings fixed. Do not use evaluation audio as a noise source or evaluation text as generation prompts. Added data/teacher ancestry must be recorded and checked.
- **Decoding ablation:** After acoustic training is selected, compare fixed greedy decoding with one development-tuned language-model or contextual-decoding variant. Build text resources only from permitted training sources. Our previous n-gram experiments showed modest gains on different historical test views, so they are supporting evidence, not an established solution to the current 4.48-point gap.
- **RASR-initialized adaptation:** If the goal explicitly permits adapting a community checkpoint, fine-tuning the strongest public model is a useful additional arm. Keep its inherited provenance and release-review questions separate from the NVIDIA-base experiment; do not assume its entire ancestry is cleared.

RASR recipe source, verified November 16: [pinned v1 mixed training configuration](https://github.com/twangodev/rasr/blob/ddff2ba501ab78f2820d159325ca3ff26699a94e/configs/train/rtx6kpro/v1-mixed.yaml). Historical evidence: [Gold G1–G4 comparison](experiments/02-gold-refinement.md), [P4/N4 lineage](checkpoint-comparison-n4-p4.md), and [Transcribe experiment](nemotron-transcribe-p4-curriculum-2026-11-07.md).

## Success criteria and reporting

The initial target is to beat the measured RASR v1 result on the unchanged common test while meeting the frozen English-retention requirement. Report both lowest observed WER and the paired uncertainty against RASR; do not equate the former with a statistically established lead. On the expanded set, first establish the public baselines before setting a numerical target. Keep streaming results and latency claims separate from this offline accuracy ranking. Callsign/number accuracy is a useful secondary measure once a shared scoring contract is defined; it must not replace the primary WER denominator post hoc.

Track one row per experiment with parent/hash, training-manifest version, seed, data mixture, update budget, selected step, development WER, English guard, final common-test WER, eligible combined-test WER, insertion/deletion/substitution counts and compute cost. No model publishing or release approval is implied by a benchmark win.
