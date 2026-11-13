# Can the 25-minute Gold training pool extend the evaluation set?

Audit completed **November 13, 2026** on OCI. **The full Gold training pool is not independent of the audited public datasets.** Of 393 clips (25.0726 minutes), **159 clips / 10.1930 minutes** have identified overlap. **234 clips / 14.8796 minutes** remain after conservative exclusion.

These remaining clips are candidates for a **future** evaluation split only after removing them and their linked recordings from all training and checkpoint-selection ancestry, including Silver exposure. **P4 and N4 already trained on this Gold pool, so none of its clips can become held-out evaluation for those existing checkpoints.**

## Gold audit result

The outcome rows below are disjoint and sum to the 393-clip input.

| Outcome | Gold clips | Minutes |
|---|---:|---:|
| Exact complete Gold-crop duplicates in audited public audio | 158 | 10.1240 |
| Additional strong acoustic match to Jacktol training | 1 | 0.0690 |
| No overlap identified by the completed checks | **234** | **14.8796** |
| Total Gold training pool | **393** | **25.0726** |

The additional acoustic match lasts **4.14 seconds**. An independent native 16-kHz comparison using Python's WAV reader measured **0.97414 zero-lag waveform correlation** and **98.904% identical samples** across all 66,240 samples. It is excluded conservatively, but is not counted as a completely sample-identical crop.

The recording-level check covered all **253 Gold source recordings**. It searched for Jacktol clips anywhere in those recordings and reconciled previously confirmed acoustic links only after verifying identical complete source PCM. Combining direct and recording-level evidence excludes **106 source recordings / 159 Gold clips**. It adds no further Gold clips beyond the 158 exact matches and the one acoustic match above. The remainder spans **147 source recordings**.

## Where the exact duplicates occur

| Public dataset | Public split | Rows compared at 16 kHz | Complete Gold crops found | Gold minutes |
|---|---|---:|---:|---:|
| `jlvdoorn/atco2-asr` | Train | 446 | 130 | 8.7335 |
| `jlvdoorn/atco2-asr` | Validation | 113 | 28 | 1.3905 |
| `jlvdoorn/atco2-asr-atcosim` | Train, ATCO2 component | 446 | Same 130 | Same 8.7335 |
| `jlvdoorn/atco2-asr-atcosim` | Validation, ATCO2 component | 113 | Same 28 | Same 1.3905 |
| `jacktol/ATC-ASR-Dataset` | Train | 6,497 | 113 | 6.9350 |
| `jacktol/ATC-ASR-Dataset` | Validation | 812 | 13 | 0.9413 |
| `jacktol/ATC-ASR-Dataset` | Test | 813 | 17 | 1.0540 |

**Do not add these rows.** The combined ATCO2/ATCOSIM repository duplicates the first repository's ATCO2 component. All exact Gold matches found in Jacktol are already among the same 158 Gold clips matched to public ATCO2. Public split assignments differ across repositories: the union matching any audited **training** split is **149 clips / 9.6875 minutes**, before the additional 4.14-second acoustic training match. Validation matches concern evaluation/selection exposure and do not alone establish gradient training.

These are dataset-level matches. Mapping them to a particular community checkpoint still requires its actual data, preprocessing and ancestry records; see the [earlier ancestry audit](atco2-training-overlap-audit-2026-11-06.md).

## Consequence for a two-hour evaluation set

| Component | Minutes |
|---|---:|
| Existing common subset with known overlap removed | 75.7615 |
| Gold remainder with no overlap identified in this audit | 14.8796 |
| Conditional combined duration for future eligible checkpoints | **90.6411** |
| Additional eligible audio needed to reach 120 minutes | **29.3589** |

Thus the proposed swap would reach approximately **1 hour 30 minutes 38 seconds**, still **29 minutes 22 seconds short**. This is a planning budget, not a new released evaluation set or a declaration of complete training independence.

The frozen Gold training and original 1,908-clip test manifests have zero shared clip IDs, source-record IDs, audio paths or airport/date groups. The [existing split contract](experiments/00-data-and-evaluation.md) separately documents their isolation. Moving Gold into evaluation would require a new versioned split and eligible model lineage; removing it only from a final fine-tuning stage would not remove earlier exposure.

For **existing P4/N4**, eligible added duration from this Gold pool is **zero**. Their current 75.7615-minute common subset still needs **44.2385 additional minutes** to reach two hours.

## Method, verification and limits

- Frozen Gold manifest: 393 clips, 253 recordings, SHA-256 `e73737113cea302971ac1e5f46e7800b5863d9e4cf9d3a109b8cf6730408c106`.
- Public snapshots: [ATCO2](https://huggingface.co/datasets/jlvdoorn/atco2-asr/tree/3b96f5791582cf6f6f348ca46fd502a981918d85), [ATCO2/ATCOSIM](https://huggingface.co/datasets/jlvdoorn/atco2-asr-atcosim/tree/21820a50f2f4d0a8f90838feb4efe88c3db31f95), and [Jacktol](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset/tree/075e736bf8aed80579d829092f74355486b10bc7). Exact pins and asset hashes are preserved in the report.
- All nine public Parquet assets and all 8,122 Jacktol WAV files passed their recorded file-hash checks. The scan decoded all 10,677 Parquet rows, comparing the 1,118 eligible 16-kHz ATCO2 rows, including duplicated repository content. The **9,559 32-kHz ATCOSIM rows were not acoustically compared**.
- Exact matching searched for whole Gold crops inside public recordings without filename or transcript filtering, and for complete public clips of at least 0.5 seconds inside Gold crops. Every positive passed an independent WAV-decoder check. Both prior train/validation positive controls were recovered. All 158 positive Gold crops were fully contained; no additional partial-only crop was found by this method.
- The separate Jacktol candidate check tested 263 transcript/duration candidates and accepted 139 under the existing acoustic heuristic. Only the independently checked 4.14-second clip added a new Gold exclusion beyond exact matching.
- CPU jobs **38487111** and **38487185** completed with exit code `0:0`, in 8m23s and 5m52s respectively. Script hashes and private-evidence hashes are recorded in the [aggregate JSON](../reports/gold-public-overlap-audit-2026-11-13.json); [execution scripts and requirements](../scripts/gold-public-overlap-audit-2026-11-13/README.md) are preserved.

**“No identified overlap” is limited to these checks.** This does not establish absence from all public datasets, inaccessible model pretraining, arbitrary transformed audio, or partial overlaps where neither complete clip contains the other. The transcript-candidate acoustic pass is not an exhaustive transformed-audio search. The existing 75.76-minute subset was also defined after test inspection and is not a fresh blind benchmark.

Audio, transcripts, row-level matches and exclusion masks remain in restricted storage. The published artifacts contain aggregate findings and hashes.
