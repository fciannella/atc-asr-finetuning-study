# Jacktol-trained Parakeet P2 on the current two-hour test

Evaluated **2026-11-06**, job **38096604**. The original Jacktol-trained P2 checkpoint has now been run on the exact current test: **1,908 utterances / 21,930 reference words / 2.000441 hours**. This is a new evaluation, not reuse of its older 1.48-hour result.

| Checkpoint | Full 2h WER | Known-Jacktol-disjoint WER |
|---|---:|---:|
| Our Jacktol-trained P2 | 18.38% | 18.90% |
| Current P4 release candidate | 20.00% | 19.89% |
| Community qenneth Parakeet | 15.91% | 16.65% |
| Community RASR Parakeet v1 | 12.71% | 13.36% |

The supplementary view contains **1,695 utterances / 19,077 words** and excludes known Jacktol-linked source recordings. It is not a guarantee of independence from all training histories. All four models here were evaluated offline. Other rows come from the [community benchmark](community-atc-benchmark-2026-11-06.md).

## Which model was trained on Jacktol?

P2 belongs to the historical A1 → A2 → P1 → P2 curriculum initialized from `nvidia/parakeet-tdt-0.6b-v3`. Early stages used **5.896 hours Jacktol, 10.534 hours UWB and 100.344 hours English**. P1/P2 used Jacktol plus English, but inherited UWB-adapted weights. P2 achieved **5.93% WER on the separate Jacktol test**. The current P4 release candidate follows a separate ATCO2 adaptation path and does not include this Jacktol/UWB curriculum.

P2's checkpoint SHA-256 is `3627bf6b74ea8cf4a041ba3c93eb3a0a89745b3a827c6dcd901018927b80f516`. It was verified before inference. UWB ancestry prevents treating this checkpoint as satisfying the current requirement to exclude UWB throughout the training history.

## Historical score correction

The older ledger's **17.79% P2** and **15.44% Gold-additive** ATCO2 scores refer to **1,394 utterances, 15,981 words and 1.475811 hours**, not today's full two-hour test. The original remote `wer.json` files and frozen training contract confirm this. Their test manifest SHA-256 is `faaad18713e6fc5400e7d3133c7b6983b2bab13633b88b869062c271b2d2b046`.

The Gold-additive arm started from P1 and added 0.678 hours of ATCO2 Gold. **554 of its training segments appear in the current two-hour test**, matching IDs, resolved source paths, offsets and durations. A full-test score for that variant would therefore include direct training examples. It was not run or added to this comparison. Its historical “leakage-safe” description applies to its original split, not the later two-hour split.

## Execution and validation

The same exact PCM crops, normalizer and standalone NeMo inference setup as the community benchmark were reused: BF16 without AMP, batch 8, greedy TDT, maximum 10 symbols per step and CUDA graph decoding. No LM or contextual hints were used. All 1,908 rows returned in order with matching references. Eight development utterances were checked before test inference. The checkpoint was fixed before testing; no fine-tuning or test-based selection occurred.

- Full: **4,030 errors** = 2,823 substitutions + 783 deletions + 424 insertions; WER **18.3767%**.
- Supplementary: **3,605 errors / 19,077 words**; WER **18.8971%**.
- Full 95% recording-cluster bootstrap interval: **17.57–19.26%**; 2,000 draws, seed 20261106.
- Test manifest SHA-256: `4af85f51d32051ee84c841c92c8e5e1eb69ee07201149197f905a7edc9c13082`.
- Primary scorer SHA-256: `cc05898d6e8dc243d6a12995799b8bc042216e4e2bf0bccf42f2f1d67e146b96`.

The [JSON report](../reports/jacktol-p2-current-test-2026-11-06.json) records exact errors, intervals, checkpoint and prediction hashes, software versions, and the historical split audit. Raw audio and predictions remain on the cluster.
