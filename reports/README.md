# Frozen reports

- [nemotron-transcribe-atc-pilot-2026-11-06.json](nemotron-transcribe-atc-pilot-2026-11-06.json): submitted bounded training experiment, frozen data counts and job IDs. Data preparation passed; GPU phases are pending. Contains no pilot accuracy claims.

- [atco2-training-overlap-audit-2026-11-06.json](atco2-training-overlap-audit-2026-11-06.json) and [CSV](atco2-training-overlap-audit-2026-11-06.csv): exact public ATCO2 audio matches, pinned ancestry evidence, preprocessing eligibility and all 12 checkpoints rescored on two common exclusion masks. Private per-utterance evidence is not included.

- [evaluation-comparison-2026-11-06.csv](evaluation-comparison-2026-11-06.csv): current 12-checkpoint comparison on only the common 1,269-utterance subset with known overlap removed; includes WER, confidence intervals, error counts, mode, adaptation training data and evidence caveats. Mirrors the evaluation tracker; does not replace the detailed frozen result JSON.
- [community-atc-benchmark-2026-11-06.json](community-atc-benchmark-2026-11-06.json) and [CSV](community-atc-benchmark-2026-11-06.csv): historical inference snapshot for eight community evaluations plus three existing baselines. Its full-test and Jacktol-only metrics retain identified overlap; use the current comparison CSV for results. Preserves revisions, hashes and decoding evidence.
- [jacktol-p2-current-test-2026-11-06.json](jacktol-p2-current-test-2026-11-06.json): fresh evaluation of our Jacktol P2 on the current two-hour test, plus the historical split and Gold-additive training-overlap audit.
- [community-atc-model-inventory-2026-11-06.json](community-atc-model-inventory-2026-11-06.json) and [CSV](community-atc-model-inventory-2026-11-06.csv): 287 screened repositories; only explicitly populated measured-result fields represent our evaluations.
- [experiment-study.json](experiment-study.json): Silver experiments, their common-test retrospective, Parakeet Gold ablations, decoding, and augmentation.
- [jacktol-gold-study.json](jacktol-gold-study.json): public references, Parakeet curricula, transfer, and diagnostic results. Reported and locally measured values are distinguished.
- [nemotron-comparisons-2026-10-07.json](nemotron-comparisons-2026-10-07.json): Nemotron Gold ablations, Jacktol-only extensions, matched curriculum, recipes, and score identities.
- [greedy-ngram-blog-checkpoints-2026-10-07.json](greedy-ngram-blog-checkpoints-2026-10-07.json): fresh greedy controls and selected LMs for the individual Parakeet G3 seed-42 checkpoint and Nemotron curriculum P3. Use this for the current blog's LM comparison.
- [greedy-ngram-summary-2026-10-07.json](greedy-ngram-summary-2026-10-07.json): earlier greedy campaign with the **averaged** Parakeet checkpoint. This is a different acoustic artifact, not an alternative score for individual G3.

These files contain aggregate result evidence. Dataset manifests, predictions, audio paths, and model artifacts are not included.

Read the [ordered experiment appendix](../docs/experiments/README.md) before comparing rows. Most JSON WER fields are fractions, but `greedy-ngram-summary-2026-10-07.json` uses percentages. A missing standalone score does not mean a job is currently pending. These are frozen snapshots, not a live status feed.

The older `jacktol-gold-study.json` is preserved as a historical snapshot. Its P2/Gold-additive ATCO2 figures use the older 1,394-utterance split. The November 6 P2 report establishes the current full-test result and identifies 554 Gold-additive training segments in the current test; use that audit when interpreting older “leakage-safe” labels.
