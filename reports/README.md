# Frozen reports

- [community-atc-benchmark-2026-11-06.json](community-atc-benchmark-2026-11-06.json) and [CSV](community-atc-benchmark-2026-11-06.csv): eight new community evaluations plus three verified existing baselines; primary and secondary normalized metrics, revisions, hashes, intervals, and decoding conditions.
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
