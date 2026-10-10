# Frozen reports

- [experiment-study.json](experiment-study.json): Silver experiments, their common-test retrospective, Parakeet Gold ablations, decoding, and augmentation.
- [jacktol-gold-study.json](jacktol-gold-study.json): public references, Parakeet curricula, transfer, and diagnostic results. Reported and locally measured values are distinguished.
- [nemotron-comparisons-2026-10-07.json](nemotron-comparisons-2026-10-07.json): Nemotron Gold ablations, Jacktol-only extensions, matched curriculum, recipes, and score identities.
- [greedy-ngram-blog-checkpoints-2026-10-07.json](greedy-ngram-blog-checkpoints-2026-10-07.json): fresh greedy controls and selected LMs for the individual Parakeet G3 seed-42 checkpoint and Nemotron curriculum P3. Use this for the current blog's LM comparison.
- [greedy-ngram-summary-2026-10-07.json](greedy-ngram-summary-2026-10-07.json): earlier greedy campaign with the **averaged** Parakeet checkpoint. This is a different acoustic artifact, not an alternative score for individual G3.

These files contain aggregate result evidence. Dataset manifests, predictions, audio paths, and model artifacts are not included.

Read the [ordered experiment appendix](../docs/experiments/README.md) before comparing rows. Most JSON WER fields are fractions, but `greedy-ngram-summary-2026-10-07.json` uses percentages. A missing standalone score does not mean a job is currently pending. These are frozen snapshots, not a live status feed.
