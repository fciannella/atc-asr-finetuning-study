# ATCO2 S0484 dataset profile

Generated: `2026-08-10T15:55:18.872088+00:00`

## Corpus layers

| Layer | Recordings | WAV hours | Speech/transcript hours | Labels |
|---|---:|---:|---:|---|
| Raw | 3,088,603 | 6,038.4 | 4,281.9 SNR/VAD signal | Automatic CNET |
| Human annotated v1.1 | 2,238 unique (2,261 files) | 4.60 | 4.21 segmented | Human XML |
| ASR evaluation v1.0 | 2,252 unique (2,275 files) | 4.63 | 4.21 segmented | Human XML |

These layers overlap and must not be added together as independent hours.

The extracted raw tree contains 15 monthly packages; the documented `2020_12` package is absent.

## Raw pool by language decision

| Language decision | Recordings | WAV hours | SNR/VAD signal hours |
|---|---:|---:|---:|
| English | 2,609,290 | 5,007.2 | 3,631.8 |
| non-English | 479,313 | 1,031.2 | 650.2 |

## Candidate automatic-label quality pools

| Filter | Recordings | WAV hours | SNR/VAD signal hours |
|---|---:|---:|---:|
| english_all_snr_ge_10 | 1,602,842 | 3,185.7 | 2,445.8 |
| english_score_0.9 | 2,001,694 | 4,065.0 | 3,013.2 |
| english_score_0.9_snr_ge_0 | 1,853,003 | 3,830.1 | 2,886.1 |
| english_score_0.9_snr_ge_10 | 1,265,425 | 2,663.9 | 2,068.3 |
| english_score_0.9_snr_ge_20 | 293,986 | 572.3 | 446.1 |
| english_score_ge_0.8_snr_ge_10 | 1,411,897 | 2,902.1 | 2,245.3 |

## Raw pool by month

| Month | Recordings | WAV hours | SNR/VAD signal hours |
|---|---:|---:|---:|
| 2020_10 | 65,692 | 131.7 | 87.0 |
| 2020_11 | 84,051 | 198.1 | 132.0 |
| 2021_01 | 57,813 | 148.0 | 91.8 |
| 2021_02 | 47,035 | 114.8 | 79.1 |
| 2021_03 | 53,766 | 98.1 | 72.8 |
| 2021_04 | 118,888 | 219.7 | 160.8 |
| 2021_05 | 228,787 | 428.5 | 304.8 |
| 2021_06 | 293,319 | 540.2 | 393.3 |
| 2021_07 | 328,771 | 615.6 | 450.9 |
| 2021_08 | 392,172 | 712.7 | 525.5 |
| 2021_09 | 354,237 | 650.5 | 475.1 |
| 2021_10 | 358,945 | 821.6 | 531.1 |
| 2021_11 | 223,577 | 474.8 | 318.0 |
| 2021_12 | 251,770 | 459.7 | 345.9 |
| 2022_01 | 229,780 | 424.5 | 313.8 |

## Raw pool by airport

| Airport | Location | Recordings | SNR/VAD signal hours |
|---|---|---:|---:|
| EETN | Tallinn | 81,738 | 106.3 |
| EGPF |  | 0 | 0.6 |
| EPLB | Lublin | 145 | 0.3 |
| Emergency |  | 0 | 0.2 |
| LKPR | Prague | 1,104,791 | 1,522.8 |
| LKTB | Brno | 615,145 | 897.4 |
| LSGS | Sion | 225,148 | 301.4 |
| LSZB | Bern | 367,268 | 535.2 |
| LSZH | Zurich | 506,366 | 702.7 |
| LZIB | Bratislava | 20,770 | 35.1 |
| SEA |  | 0 | 0.0 |
| UNKNOWN |  | 975 | 0.0 |
| YBBN | Brisbane | 113,666 | 126.6 |
| YSSY | Sydney | 52,591 | 53.1 |

## Human-annotated package

Accepted by human re-checker: **1,500 recordings / 2.78 WAV hours**.

| Airport | Recordings | WAV hours |
|---|---:|---:|
| LKPR | 89 | 0.33 |
| LKTB | 47 | 0.12 |
| LSGS | 619 | 1.37 |
| LSZB | 338 | 0.68 |
| LSZH | 424 | 0.85 |
| LZIB | 88 | 0.26 |
| YSSY | 656 | 0.99 |

## Human transcript quality tags

| Tag/value | Segments | Hours |
|---|---:|---:|
| correct_tagging_0 | 3,468 | 3.70 |
| correct_tagging_1 | 426 | 0.51 |
| correct_transcript_0 | 104 | 0.10 |
| correct_transcript_1 | 3,790 | 4.10 |
| non_english_0 | 3,609 | 3.94 |
| non_english_1 | 285 | 0.27 |
| total | 3,894 | 4.21 |

## Overlap

- Human v1.1 and ASR evaluation basenames in common: **2,238**.
- Exact normalized clip identities found in the raw lists: **0**. Different segmentation may still represent overlapping source sessions.

## Training-relevant metadata

| Dimension | Availability | Suggested use |
|---|---|---|
| Transcript provenance | Human XML vs automatic CNET | Core ablation axis |
| Human quality | accepted list; correct_transcript; correct_tagging; non_english | Filtering |
| Language | English decision, exact detector score, explicit non-English tags | Threshold ablation |
| Acoustic quality | Wada-SNR raw score and bins | Noise curriculum/ablation |
| Geography | 10 airport codes | Leakage-safe splits and leave-airport-out evaluation |
| Radio conditions | Channel and frequency | Coverage analysis |
| Time | UTC date/time and monthly packages | Session/day-disjoint splits |
| Speakers | Diarization speaker ID; ATCO labels; callsigns | Role-balanced evaluation |
| Entities | Callsign, command, value, unnamed phrase, anonymize tags | Error slices |
| Context | Nearby callsigns, waypoints, airport documentation | Contextual-biasing experiments |

## License constraint

The included COPYRIGHT.txt states that ATCO2 Project Data may not be used or distributed without written ELDA/ELRA consent. Keep W&B artifacts metadata-only; do not upload audio or transcripts.

