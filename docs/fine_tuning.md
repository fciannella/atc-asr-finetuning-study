| Article overview | Working title: Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills Format: Hands-on developer tutorial Draft deadline: October 2, 2026 (requested by Michelle Horton) Length: Under 1,400 words Target publication: October 13, 2026, alongside the livestream, pending editorial confirmation |
| :---- | :---- |
| **Submitter’s notes** | **Author / submitter:** Francesco Ciannella **Jira:** [TECHBLOG-5795](https://nvidia.atlassian.net/browse/TECHBLOG-5795) **Scope:** Use the Nemotron ASR fine-tuning skills with a coding agent to adapt a pretrained ASR model to domain-specific audio, using air traffic control (ATC) as the worked example. Cover data preparation, NeMo training, and baseline versus fine-tuned evaluation. **Community contribution:** NVIDIA plans to release a two-hour, human-annotated ATC dataset alongside the blog and livestream, giving developers and researchers data and guidance to build on. **Release links:** Dataset, skills, example code, and staging/publication links pending. **Results:** Exact model, dataset splits, training setup, and WER results to be verified against experiment artifacts. |
| **Editorial and technical review** | **Pitch reviewer:** Michelle Horton (confirmed by Elizabeth Goodman) **Editorial contact:** Elizabeth Goodman (cw) **Proposed reviewers / contacts:** [Maryam Motamedi US](mailto:maryamm@nvidia.com), [Adi Margolin US](mailto:amargolin@nvidia.com), @Maryam Motamedi, Adi Margolin, Jayda Ritchie — specific responsibilities to be confirmed. **Review status:** Michelle requested a draft under 1,400 words by October 2\. Publication timing remains subject to review. |
| **Campaign / livestream coordination** | **Livestream:** October 13, 2026 **Livestream details:** Maryam Motamedi and Rebecca Kao **PMM lead and campaign reviewers:** To be confirmed |
| **Awareness / outstanding details** | Confirm dataset documentation, license, download location, and how the released two-hour dataset relates to the experimental training, validation, and test splits. Verify measured results and reproducible example links before publication. |
| **Social copy — draft for review** | **LinkedIn:** Adapt speech recognition to your domain with NVIDIA Nemotron Speech skills. This hands-on ATC example walks through preparing data, fine-tuning with NeMo, and evaluating recognition quality. Alongside the blog, NVIDIA plans to release two hours of human-annotated ATC data to help the community experiment and build on the work. Join the October 13 livestream for more. \[Blog, dataset, and livestream links pending\] **X:** Fine-tune ASR for your domain with NVIDIA Nemotron Speech skills. Explore the ATC workflow and a planned community release of two hours of human-annotated ATC data. Join the October 13 livestream. \[Links pending\] |

| Feedback Deadline: 10/09/26 Target Publish Date: 10/13/26 |  |
| ----- | :---- |
| **Submitters Notes:** | Please see above |
| **Formal Reviewers** | **Formal Reviewers: [Maryam Motamedi US](mailto:maryamm@nvidia.com) [Adi Margolin US](mailto:amargolin@nvidia.com)[Yitagessu Gebremedhin US](mailto:ygebremedhin@nvidia.com)[Myungjong Kim US](mailto:myungjongk@nvidia.com)[Jinhan Wang US](mailto:jinhanw@nvidia.com)[Rajpreet Thethy (cw) US](mailto:rthethy@nvidia.com)**  |
| **Awareness (CC)** | [Chintan Patel US](mailto:cpatel@nvidia.com)[Udi Karpas IL](mailto:ekarpas@nvidia.com)[Davide Onofrio US](mailto:donofrio@nvidia.com)[Will Jennings US](mailto:wjennings@nvidia.com) |

# **Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills**

Suppose a speech recognition model transcribes everyday conversation accurately, yet struggles with a pilot reading back a clearance. The model already knows English. What it needs to learn is how English sounds over a radio and how people use it in air traffic control. Fine-tuning lets us teach those differences by updating a pretrained model with examples from the new domain.

The challenge is deciding what to teach, how much to change, and how to check that existing capabilities survive. We explored those questions using air traffic control (ATC) speech and two NVIDIA agent skills. Here we follow Nemotron through Gold refinement, longer domain-only training, and a staged curriculum with general-English replay. We measure recognition quality with word error rate (WER), which counts substitutions, deletions, and insertions relative to the reference transcript. Lower is better.

# **How the skills guide the work**

Think of a skill as a reusable set of instructions for a coding agent. It helps the agent ask the right questions, carry out the work in a sensible order, and collect the evidence needed to judge the result.

You can use the two skills together to move from an adaptation goal to a trained and evaluated model:

| [`nemotron-asr-finetune`](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) — plan and coordinate adaptation | [`nemo-speech-asr-finetune`](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune) — train and evaluate with NeMo |
| :---- | :---- |
| **Define success:** Turn a domain or language request into accuracy targets, latency constraints, and a baseline evaluation plan. | **Inspect your model:** Identify its architecture, tokenizer, and preprocessing, then choose a compatible NeMo recipe and environment. |
| **Choose an approach:** Decide whether word boosting, vocabulary support, an n-gram language model, or acoustic fine-tuning fits the measured errors. | **Prepare training data:** Check manifests and transcript conventions, configure Lhotse loaders, and mix domain audio with general-speech replay. |
| **Plan resources:** Assess available audio, label quality, GPU capacity, and time/cost trade-offs; identify missing inputs before expensive work. | **Configure and run training:** Set learning rate, batch sizing, step budget, validation, and checkpoint retention for the chosen model. |
| **Protect existing capabilities:** Define domain and general-speech checks so better domain accuracy does not hide catastrophic forgetting. | **Measure the result:** Evaluate exported checkpoints with normalized WER on domain and general speech, independently of training logs. |
| **Choose the next experiment:** Use measured errors and constraints to prioritize data, training, or decoding changes and route specialist work. | **Refine the model:** Adjust replay or curriculum and compare individual and averaged checkpoints, keeping changes only when evaluation supports them. |
| **Coordinate delivery:** Hand off serving and endpoint evaluation to the deployment skill when the selected model meets the objective. | **Produce reviewable artifacts:** Retain training configurations, checkpoints, exported `.nemo` models, and standalone evaluation results. |

The orchestration skill coordinates specialist execution; serving through Riva/NIM requires the `nemotron-speech` skill. Both skills run through your coding agent with access to your data, software environment, and hardware.

Together, the skills connect experiment planning with execution. Each training run has a reason, a defined dataset, and a test that can tell us whether the change helped.

## **Prerequisites**

You need a coding agent supporting Agent Skills, Node.js/npm, Git, and a Linux GPU host with NVIDIA drivers, NVIDIA Container Toolkit, and a compatible NeMo ASR environment. Prepare a licensed `.nemo` checkpoint, readable audio, separate training/development/test manifests, and storage for checkpoints. JSONL rows need `audio_filepath`, `duration`, and `text`.

The linked historical run used eight GPUs, BF16, and NeMo `2.8.0rc0`; that is its recorded setup, not a minimum for every pilot. Pin the software and model versions and check available GPU memory before training. Skills guide the work; they do not provision hardware or install the training environment.

## **Install and activate the skills**

Follow the [NVIDIA skills installation flow](https://github.com/NVIDIA/skills#quickstart):

```sh
npx skills@latest add NVIDIA/skills --skill nemotron-asr-finetune
npx skills@latest add NVIDIA-NeMo/Speech --skill nemo-speech-asr-finetune
npx skills@latest list
```

Choose your agent and installation scope, then start a fresh agent session and confirm both skills are available.

Explicitly name both in a prompt. This sample is illustrative; replace the paths with your files:

```textproto
Use nemotron-asr-finetune and nemo-speech-asr-finetune to adapt /models/base.nemo to ATC. Human-verified training and development manifests are /data/atc/train.jsonl and /data/atc/dev.jsonl; keep /data/atc/test.jsonl locked until selection is complete. English replay is /data/english/train.jsonl; the general evaluation is /data/librispeech/test-clean.jsonl. Audit transcript style and overlap, inspect the checkpoint and GPUs, and measure baseline WER. If acoustic training is justified, propose a conservative pilot with replay. Preserve tokenizer and preprocessing. Save the effective configuration, manifest hashes, exported model, predictions, and baseline-versus-adapted WER report in /exp/atc. Report missing inputs before launching training.
```

## **Choose the architecture as part of the product decision**

Before choosing a recipe, inspect the checkpoint. Its encoder, decoder, tokenizer, feature normalization, and loss function were designed and trained to work together.

| Architecture | Basic idea | Practical implication |
| :---- | :---- | :---- |
| CTC | Predict frame-level tokens, then collapse blanks and repetitions | Parallelizable decoding and convenient external-LM integration |
| RNN-T | Combine an acoustic encoder with a prediction and joint network | Streaming-friendly and conditioned on previously emitted tokens |
| TDT | Extend the transducer family with token-and-duration outputs | Efficient sequence modeling with architecture-specific duration settings |

Our experiments used Nemotron 3.5 ASR Streaming 0.6B, Parakeet CTC 1.1B, and Parakeet TDT 0.6B v3. Each required a compatible training recipe. CTC can be attractive for offline decoding and language-model integration, while a streaming transducer may suit an application that must respond as someone speaks. Choose with the intended application in mind, then compare accuracy. Parakeet is a separate model family in this comparison. Preserve the CTC head and tokenizer, RNN-T loss and streaming configuration, or TDT duration vocabulary and decoder settings, as appropriate for the checkpoint.

When adapting within the same language, keeping the pretrained tokenizer is usually the safest starting point. If a replacement is needed, train it only on training text. Input normalization deserves similar care: changing our completed Nemotron checkpoint from global to per-feature normalization without retraining sharply increased WER on both ATC and general speech. The lesson was to preserve the preprocessing the model had learned to expect.

# **Why ATC is a demanding adaptation problem**

ATC speech concentrates several ASR challenges in one domain:

* narrow-band radio, interference, clipping, and variable gain;
* short, context-dependent transmissions;
* accents and non-native English;
* callsigns, runways, headings, altitudes, and frequencies;
* specialized, compressed phraseology;
* number or callsign errors that are more consequential than ordinary conversational substitutions.

Both the sound and the language differ from ordinary conversation. An adapted model may nevertheless need to recognize general English too, so we measured those two capabilities throughout the study.

## **Know which data is being released**

The documented community-release plan covers **two hours of human-annotated ATCO2 evaluation data**, containing 1,908 segments. It does not currently include the study's 0.418-hour Gold training set, 0.100-hour Gold development set, or licensed 314.7-hour Silver training corpus. The download URL, license, and final packaging remain pending release-owner confirmation. Keep the community evaluation data out of training and model selection.

For a public starting point, the [Jacktol ATC-ASR dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) provides official training, validation, and test splits under its dataset-card terms. Alternatively, supply your own licensed recordings. Audit overlap before comparing Jacktol with ATCO2. New data means a new experiment, not reproduction of the historical scores.

## **Historical training and evaluation example**

The linked 2:1 recipe illustrates the mechanics of training and scoring. The newer Gold, Jacktol-only, and curriculum campaigns below use their own configurations; this archived recipe does not reproduce those results.

## **Inspect and run the training configuration**

The [Nemotron 2:1 replay example](https://github.com/fciannella/atc-asr-finetuning-experiments/tree/main/experiments/nemotron-mixed-2to1) contains an [effective training configuration](https://github.com/fciannella/atc-asr-finetuning-experiments/blob/main/experiments/nemotron-mixed-2to1/config.yaml), executable `recipe.sh`, sampling weights, logs, and results. **This repository is currently private; public release is pending.** The container digest and base-checkpoint download link still need to be pinned for release.

The archived run mixed 314.721 hours of Silver ATCO2 with 314.721 hours of English, sampled 2:1. It used 40,000 optimizer steps, learning rate `1e-4`, and 400 warmup steps. Validation retained the top five checkpoints plus last. The published comparison evaluated the final exported 40,000-step model, separately from the best in-training checkpoint.

Authorized readers set `NEMO_ROOT`, `BASE_MODEL`, `ATC_TRAIN_MANIFEST`, `GENERAL_TRAIN_MANIFEST`, `ATC_DEV_MANIFEST`, and `OUTPUT_DIR`, then run `bash recipe.sh`. The YAML describes the settings; the shell script executes them. The example README supplies the path-setting commands and checkpoint-resume procedure.

This is a historical Silver-data recipe. New ATCO2 development follows the Gold-only policy. A small Gold pilot needs its own conservative learning rate, batch sizing, and step budget; copying a 40,000-step recipe unchanged would be inappropriate.

## **Evaluate both models and inspect the outputs**

Run standalone inference with the untouched and exported fine-tuned checkpoints on identical domain and general-English manifests. Development selects checkpoints and decoder settings; the locked test is evaluated only after selection. The historical table below uses development data, not the community test.

For the Nemotron streaming checkpoint, use the [NeMo inference script](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py). In the GPU container, set `NEMO_ROOT`, `BASE_MODEL`, `FINETUNED_MODEL`, `ATC_EVAL_MANIFEST`, and `GENERAL_EVAL_MANIFEST` to real paths. Use the historical ATC development manifest for the linked comparison and LibriSpeech test-clean for general evaluation. Set `STUDY_ROOT` to this study checkout and `EVAL_DIR` to a new output directory. Run in Bash:

```sh
set -euo pipefail
mkdir -p "$EVAL_DIR"
for variant in baseline finetuned; do
 model="$BASE_MODEL"
 if [ "$variant" = finetuned ]; then model="$FINETUNED_MODEL"; fi
 for split in atc general; do
 manifest="$ATC_EVAL_MANIFEST"
 if [ "$split" = general ]; then manifest="$GENERAL_EVAL_MANIFEST"; fi
 out="$EVAL_DIR/$variant-$split"
 mkdir "$out"
 python "$NEMO_ROOT/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py" \
 model_path="$model" dataset_manifest="$manifest" output_path="$out" \
 target_lang=en-US att_context_size='[56,3]' decoder_type=rnnt \
 pad_and_drop_preencoded=true batch_size=8 cuda=0 strip_lang_tags=true \
 amp=false compute_dtype=float32
 done
done

```

Materialize segmented audio as clips first. These settings are specific to the historical Nemotron recipe; Parakeet needs its own inference path. The newer campaigns' pinned streaming entrypoint requires `float32`; record the precision and software revision. Missing historical environment pins prevent a guarantee of exact numerical reproduction.

For each output directory, score its prediction JSONL (`text` and `pred_text`) using the actual generated filename:

```py
python "$STUDY_ROOT/scripts/score_wer.py" /actual/predictions.jsonl \
 --output /actual/wer.json
```

The scorer folds case and diacritics, normalizes symbols while retaining apostrophes, and divides total word edits by total reference words. Use the same implementation for both models and verify matching sample counts.

Expected artifacts are configuration and manifest hashes, training logs, retained checkpoints, an exported `.nemo`, and four prediction/score pairs. Each `wer.json` includes utterances, reference words, word errors, and `normalized_wer` as a fraction; multiply by 100 for percentages. Save decoder settings and checkpoint hashes with the report.

## **Historical reference: the original 2:1 replay recipe**

| Evaluation | Untouched Nemotron | Fine-tuned Nemotron |
| :---- | :---- | :---- |
| Historical ATC development: 1,007 utterances | 75.75% WER | **37.38% WER** |
| LibriSpeech test-clean: 2,620 utterances | 3.52% WER | **3.39% WER** |

The [unrounded results](https://github.com/fciannella/atc-asr-finetuning-experiments/blob/main/experiments/nemotron-mixed-2to1/results.yaml) show a **38.37-percentage-point reduction**, or **50.65% relative reduction**, in ATC development WER. These are observed results, not promised outcomes. This development comparison belongs to the archived recipe. It is separate from the newer Nemotron comparisons below and is not their baseline.

## **ATCO2 and Jacktol provide different evidence**

The ATCO2 delivery contained 3,088,603 recordings and approximately 4,281.9 hours. Filtering for language, duration, confidence, text, and audio quality produced a 314.721-hour English Silver release with 396,461 segments. "Silver" means high-confidence CNET hypotheses rather than human-verified references. The smaller human-Gold pool was assigned to non-overlapping roles:

| Split | Audio | Segments | Role |
| :---- | :---- | :---- | :---- |
| Gold training | 0.418 h | 393 | Final supervised refinement |
| Gold development | 0.100 h | 100 | Checkpoint and recipe selection |
| Gold community test | 2.000 h | 1,908 | Held out from training/selection; reused comparison benchmark |

The splits had zero overlap by airport-date, audio path, record ID, and source recording. Reserving two hours for evaluation left about 25 minutes for Gold training. That was a deliberate trade-off: we wanted enough held-out speech to assess what the model had learned.

The public [Jacktol ATC-ASR dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) contains 7.405 hours, including about 5.9 training hours and official validation and test splits. It supplied an external comparison and training-only domain text. Because it includes material derived from public ATCO2 recordings, cross-corpus claims required an acoustic overlap audit.

# **A reproducible fine-tuning workflow**

1. **Set the objective.** Define the target domain WER and an acceptable general-English regression. We used ATCO2 Gold and LibriSpeech test-clean to measure both.

2. **Prepare and version the data.** Validate audio and transcripts, record provenance, keep related recordings in the same split, and hash the manifests. Document replay data volumes and sampling ratios.

3. **Freeze evaluation.** Use development data to select checkpoints and decoder settings; reserve the test set for final evaluation. Score exported models with the same text normalizer.

4. **Train and diagnose.** Start conservatively, change one factor at a time, and retain both the best validation checkpoints and the final checkpoint.

5. **Save the recipe.** Package the `.nemo` model with its starting revision, tokenizer, preprocessing, manifest hashes, training and decoder settings, selection rule, normalizer, and domain/general-English scores.

# **Three questions answered by the newer Nemotron experiments**

The newer work lets us follow one model, Nemotron 3.5 ASR Streaming 0.6B, through three questions: does Gold refinement need replay, how far does Jacktol-only training go, and does a staged curriculum improve the trade-off? All scores below come from completed standalone evaluations, captured in the [October 7 result snapshot](../reports/nemotron-comparisons-2026-10-07.json).

Every new row uses native cache-aware streaming, attention context `[56,3]`, `en-US`, greedy decoding with at most 10 symbols per step, and float32 inference. No beam search or external language model contributes to these results. The model retains its native tokenizer, RNN-T loss, and feature normalization. Checkpoint selection uses the relevant development/validation split; the tables score the actual exported artifacts.

The existing test sets have been evaluated in previous experiments. Treat these as repeated benchmark comparisons, not fresh blind tests. The new baseline is **77.00% on ATCO2**, **72.22% on Jacktol**, and **3.50% on LibriSpeech**. Older figures such as 75.57% ATCO2 and 3.52% LibriSpeech belong to historical evaluator runs and should not be substituted into these comparisons. Matching the model name alone is insufficient to establish an identical evaluation contract.

## **Comparison 1: What do Gold labels and English replay each contribute?**

We repeated the G1–G4 design with Nemotron. The parent trained for 20,000 steps using 80% Silver ATCO2 and 20% English sampling. Each Gold run then used 2,000 steps at peak learning rate `1e-5`, with 393 Gold training segments, about 25 minutes of audio. The two-hour ATCO2 benchmark and LibriSpeech test-clean remained evaluation-only. Gold development selected checkpoints.

| Model / strategy | Initialization and sampling | Seed | ATCO2 WER | LibriSpeech WER |
| --- | --- | ---: | ---: | ---: |
| Untouched base | Pretrained model | — | 77.00% | 3.50% |
| Silver/English parent | Base; 80% Silver / 20% English | 42 | 36.52% | 3.48% |
| G1: Gold only | Base; 100% Gold | 1234 | 26.09% | 5.16% |
| G2: Gold + replay | Base; 85% Gold / 15% English | 1234 | 26.89% | 3.72% |
| G3: Gold refinement | Parent; 85% Gold / 15% English | 1234 | 22.97% | 3.68% |
| G3: repeat seed | Parent; 85% Gold / 15% English | 42 | 22.46% | 3.63% |
| G4: retain Silver | Parent; 70% Silver / 15% Gold / 15% English | 1234 | 30.08% | 3.54% |
| G4: repeat seed | Parent; 70% Silver / 15% Gold / 15% English | 42 | 30.69% | 3.56% |

G1 demonstrates substantial adaptation from scarce human labels, but English WER rises from 3.50% to 5.16%. G2 trades a small amount of ATCO2 accuracy for much better English retention: 26.89% ATCO2 and 3.72% LibriSpeech. Replay matters even when the domain training set is small.

G3 starts from the broadly adapted parent and improves ATCO2 from 36.52% to 22.97%, with English at 3.68%. Seed 42 reaches 22.46% and 3.63%, supporting the direction of the result across two seeds without establishing a confidence interval. Both G3 seeds outperform the corresponding G4 runs, where continued Silver sampling weakens the Gold correction. This is evidence for separating broad adaptation from trusted-label refinement under this recipe.

These are scoped experimental comparisons, including approved historical Silver ablations. They do not make the study's licensed training data part of the evaluation-data release or establish a general recommendation to fine-tune every model on 25 minutes of speech.

## **Comparison 2: Does more Jacktol-only training keep helping?**

We next started from the untouched Nemotron model and trained only on Jacktol's official training split: 6,497 clips, 5.896 hours. No ATCO2, UWB, English replay, or synthetic audio was used. Each phase selected its checkpoint using Jacktol validation, with the same held-out manifests and normalized scorer across phases.

| Completed training budget | Jacktol validation WER | Jacktol test WER | ATCO2 recording-disjoint WER | LibriSpeech WER |
| --- | ---: | ---: | ---: | ---: |
| Untouched base | 72.33% | 72.22% | 77.79% | 3.50% |
| 5k steps | 14.26% | 14.11% | 29.41% | 6.30% |
| 10k cumulative steps | 11.43% | 11.01% | 25.27% | 6.86% |
| 20k cumulative steps | 9.10% | 9.34% | 22.12% | 7.52% |
| 30k cumulative steps | 8.17% | 8.05% | 20.44% | 7.76% |

Jacktol test WER improves at every reported budget, reaching **8.05%** at 30k. English moves in the opposite direction, reaching **7.76%**, versus 3.50% for the base model. More domain optimization helped specialization but did not solve retention.

The budgets describe completed training phases; the validation-selected checkpoint may precede the endpoint. Each extension initializes from the prior phase's final weights, restarts AdamW, and rewarms the learning-rate schedule. This is not an uninterrupted 30k-step cosine run, so the comparison cannot isolate step count from the restart schedule.

The ATCO2 column uses the existing **1,695-segment, 1.749-hour recording-disjoint view**, excluding 213 segments linked to any Jacktol split. It is distinct from the full 1,908-segment, two-hour benchmark in the Gold table. The acoustic audit reduces known overlap; it does not prove that all possible overlap is absent.

## **Comparison 3: Can a staged curriculum improve the balance?**

The next campaign returned to the untouched base and matched the data pools and curriculum of our earlier Parakeet study: 5.896 hours of Jacktol, 10.534 hours of UWB ATC, and 100.344 hours of LibriSpeech training audio. A1 and A2 blend both ATC sources with English; P1–P3 progressively emphasize Jacktol while retaining English replay. Each stage starts from the previous stage's selected export and uses a fresh optimizer and schedule.

| Stage | Jacktol / UWB / English sampling | Stage step budget | Jacktol validation WER | Jacktol test WER | LibriSpeech WER |
| --- | --- | ---: | ---: | ---: | ---: |
| Untouched base | — | 0 | 72.33% | 72.22% | 3.50% |
| A1 | 30 / 30 / 40 | 14,465 | 10.51% | 11.03% | 5.18% |
| A2 | 40 / 35 / 25 | 23,144 | 7.30% | 7.90% | 5.20% |
| P1 | 80 / 0 / 20 | 5,000 | 6.86% | 7.57% | 5.19% |
| P2 | 85 / 0 / 15 | 4,000 | 6.74% | 7.53% | 5.27% |
| P3 | 90 / 0 / 10 | 3,000 | 6.61% | 7.31% | 5.30% |

P3 has the lowest Jacktol validation WER among these exports and reaches **7.31% Jacktol test WER**, an **89.88% relative reduction** from the 72.22% base. Compared with the 30k Jacktol-only model, it improves Jacktol by **0.74 percentage points** and reduces English WER by **2.46 points**, to 5.30%. Retention is better, but still worse than the original 3.50%; replay has not eliminated forgetting.

A manifest audit confirmed identical evaluation audio paths, offsets, durations, and normalized references between the 30k and curriculum campaigns, despite different raw Jacktol and English manifest hashes. This comparison changes training data, order, budget, and label conventions together. It demonstrates the combined recipe's outcome, not a controlled estimate of replay's individual contribution. The five budgets total 49,609 steps, but that is not the exact training lineage of P3 because intermediate stages hand off selected checkpoints rather than necessarily their final weights.

For recipe fidelity, this campaign preserved the historical uppercase Jacktol and lowercase UWB/English training labels, unlike the normalized-label Jacktol-only runs. It also preserved the historical checkpoint-weight choices: A1 exports ordinary weights despite selection by exponential-moving-average (EMA) validation scores; A2 exports EMA weights. Standalone evaluation measures the exported models, not their in-training proxies.

P3 also scores **20.68% on UWB test** and **19.96% on the ATCO2 recording-disjoint view**. The latter is a qualified transfer measurement: the subset removes known Jacktol-linked recordings, but the additional UWB training source has not received a fresh cross-corpus acoustic fingerprint audit. The full ATCO2 score, 19.22%, remains overlap-confounded and should not headline an unseen-transfer claim.

## **Where Parakeet fits in the comparison**

The historical Parakeet curriculum reached 5.93% Jacktol test WER, compared with Nemotron P3's 7.31%. That gives useful context for the matched-data/curriculum study, but it is not an architecture-only contest: the checkpoints differ in pretraining, tokenizer, streaming/offline behavior, selection details, and execution topology. Neither the earlier Parakeet beam-search result nor its 17.61% ATCO2 result with an external language model should be attributed to Nemotron. The [earlier experiment narrative](fine_tuning_blog_2800.md) retains those separate results.

## **What users can learn from these comparisons**

The skill table maps directly to the experiments. The orchestration skill frames the question and the domain/general evaluation contract; the NeMo skill preserves model-specific settings, configures the data mixture, and evaluates exported checkpoints. The useful output is a comparison that explains the trade-off, not just the lowest domain WER.

For ATCO2, the tested Silver-to-Gold sequence with replay provided the strongest balance among the Gold ablations. For Jacktol, longer domain-only training kept improving specialization while increasing forgetting. The staged curriculum improved on that domain-only model on both measured objectives, although it still regressed on general English. Those are three distinct conclusions, each backed by its own baseline and evaluation set.

## **ATC02 Dataset**

We are releasing a small subset of the larger ATC02 dataset for evaluation purposes. This dataset can be used to evaluate models that the community further fine-tunes.

NVIDIA gratefully acknowledges ELDA (Evaluations and Language resources Distribution Agency) for making the ATCO2 Project Data available to train and test this model checkpoint. The ATCO2 corpus supports research and development in air-traffic-control speech technologies, including automatic speech recognition, speaker diarization, language detection, and related speech and language understanding tasks. The dataset includes air-traffic-control voice communications and associated metadata collected and processed through the ATCO2 project. A 2-hour subset of the larger 5K hour dataset is linked here for evaluation purposes only. To learn more about this dataset, visit [https\://catalog.elra.info/](https://catalog.elra.info/) (Catalogue Reference: ELRA-S0484, ISLRN: 589-403-577-685-7).

## **Try the workflow**

Install the [orchestration skill](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) and [training skill](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune), inspect the [example](https://github.com/fciannella/atc-asr-finetuning-experiments/tree/main/experiments/nemotron-mixed-2to1), and start with licensed domain data or [Jacktol](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset). Measure both domain gains and general-English retention.
