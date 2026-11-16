# Gold training candidates after freezing the 90.64-minute evaluation set

**Protocol update, November 16:** The user accepted relaxing airport/date grouping. Shared airport/date groups alone no longer exclude candidate Gold data; source identity and audio-overlap checks against the frozen evaluation and development sets remain mandatory. This makes the 798 clips / 54.46 minutes the candidate pool under the accepted policy. It does not change the running Silver data, restart Gold training, or remove the prior hold on Gold. The stricter counts below are retained as a comparison.

Read-only audit completed November 16, 2026. The frozen evaluation set contains 1,503 clips / 90.6411 minutes. The existing development set contains 100 clips / 6.0019 minutes. Neither is included in the candidate training pool. Current Nemotron training remains Silver-only; Gold is held by user decision. No training manifest or job was changed by this audit.

| Candidate source | Clips | Minutes | Passing the current airport/date group rule | Minutes after that rule |
|---|---:|---:|---:|---:|
| Removed from the original 1,908-clip test by public-overlap exclusions | 639 | 44.2650 | 34 | 2.3192 |
| Removed from the former 393-clip Gold pool by public-overlap exclusions | 159 | 10.1930 | 47 | 2.7226 |
| **Total** | **798** | **54.4580** | **81** | **5.0419** |

All 798 candidate clips pass the source-ID, resolved-path/basename, complete source-file SHA-256 and exact 16-kHz mono signed-int16 crop-PCM checks against both the protected evaluation set and development set. The retained 81 clips also have unique PCM hashes within that retained pool. The earlier 47-clip count considered only the former Gold-training pool; reclaiming excluded original-test clips adds 34 under the same strict policy.

The additional exclusions are 605 original-test candidates and 112 former-Gold candidates that share an airport/date or corresponding Gold group with protected data. These group matches are a conservative independence rule, **not evidence that the clips contain identical audio**. Consequently, 54.46 minutes is the source/exact-audio-disjoint candidate budget, while 5.04 minutes is the budget under the current stricter group contract. The protocol change was accepted on November 16: training and evaluation may share airport/date groups, while source/audio separation remains required. No universal guarantee about transformed copies, partial audio overlap or inaccessible model ancestry follows from negative exact-match checks.

Public-training overlap is a reason to exclude material from an independent test; it does not itself disqualify that material from training. Public availability is not public-domain status; the source data's applicable terms remain relevant. Existing P4/N4 exposure to the 234-clip evaluation extension remains unchanged.

The removed original-test duration is measured from its frozen manifest: 44.2650 minutes. The original package was approximately two hours, so this differs slightly from simply subtracting 75.7615 from exactly 120 minutes.

[Aggregate audit](../reports/gold-training-reclaim-audit-2026-11-16.json) · [Audit source](../scripts/audit_gold_training_reclaim.py) · [Frozen evaluation release](atco2-overlap-removed-release-2026-11-13.md)
