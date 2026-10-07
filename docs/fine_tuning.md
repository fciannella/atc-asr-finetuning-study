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

The challenge is deciding what to teach, how much to change, and how to check that existing capabilities survive. We explored those questions using air traffic control (ATC) speech and two NVIDIA agent skills. Here we follow the experiments from automatically transcribed audio to human annotations and, finally, an n-gram language model. We measure recognition quality with word error rate (WER), which counts substitutions, deletions, and insertions relative to the reference transcript. Lower is better.

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

Materialize segmented audio as clips first. These settings are specific to the historical Nemotron recipe; Parakeet needs its own inference path. `float32` makes the current script's default precision explicit; record it and the software revision. Missing historical environment pins prevent a guarantee of exact numerical reproduction.

For each output directory, score its prediction JSONL (`text` and `pred_text`) using the actual generated filename:

```py
python "$STUDY_ROOT/scripts/score_wer.py" /actual/predictions.jsonl \
 --output /actual/wer.json
```

The scorer folds case and diacritics, normalizes symbols while retaining apostrophes, and divides total word edits by total reference words. Use the same implementation for both models and verify matching sample counts.

Expected artifacts are configuration and manifest hashes, training logs, retained checkpoints, an exported `.nemo`, and four prediction/score pairs. Each `wer.json` includes utterances, reference words, word errors, and `normalized_wer` as a fraction; multiply by 100 for percentages. Save decoder settings and checkpoint hashes with the report.

## **What fine-tuning changed**

| Evaluation | Untouched Nemotron | Fine-tuned Nemotron |
| :---- | :---- | :---- |
| Historical ATC development: 1,007 utterances | 75.75% WER | **37.38% WER** |
| LibriSpeech test-clean: 2,620 utterances | 3.52% WER | **3.39% WER** |

The [unrounded results](https://github.com/fciannella/atc-asr-finetuning-experiments/blob/main/experiments/nemotron-mixed-2to1/results.yaml) show a **38.37-percentage-point reduction**, or **50.65% relative reduction**, in ATC development WER. These are observed results, not promised outcomes. This development comparison is distinct from the later two-hour locked ATCO2 test.

## **ATCO2 and Jacktol provide different evidence**

The ATCO2 delivery contained 3,088,603 recordings and approximately 4,281.9 hours. Filtering for language, duration, confidence, text, and audio quality produced a 314.721-hour English Silver release with 396,461 segments. "Silver" means high-confidence CNET hypotheses rather than human-verified references. The smaller human-Gold pool was assigned to non-overlapping roles:

| Split | Audio | Segments | Role |
| :---- | :---- | :---- | :---- |
| Gold training | 0.418 h | 393 | Final supervised refinement |
| Gold development | 0.100 h | 100 | Checkpoint and recipe selection |
| Gold community test | 2.000 h | 1,908 | Locked final evaluation |

The splits had zero overlap by airport-date, audio path, record ID, and source recording. Reserving two hours for evaluation left about 25 minutes for Gold training. That was a deliberate trade-off: we wanted enough held-out speech to assess what the model had learned.

The public [Jacktol ATC-ASR dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) contains 7.405 hours, including about 5.9 training hours and official validation and test splits. It supplied an external comparison and training-only domain text. Because it includes material derived from public ATCO2 recordings, cross-corpus claims required an acoustic overlap audit.

# **A reproducible fine-tuning workflow**

1. **Set the objective.** Define the target domain WER and an acceptable general-English regression. We used ATCO2 Gold and LibriSpeech test-clean to measure both.

2. **Prepare and version the data.** Validate audio and transcripts, record provenance, keep related recordings in the same split, and hash the manifests. Document replay data volumes and sampling ratios.

3. **Freeze evaluation.** Use development data to select checkpoints and decoder settings; reserve the test set for final evaluation. Score exported models with the same text normalizer.

4. **Train and diagnose.** Start conservatively, change one factor at a time, and retain both the best validation checkpoints and the final checkpoint.

5. **Save the recipe.** Package the `.nemo` model with its starting revision, tokenizer, preprocessing, manifest hashes, training and decoder settings, selection rule, normalizer, and domain/general-English scores.

# **Experiment 1: What did additional Silver data teach the model?**

We began with nemotron-3.5-asr-streaming-0.6b.nemo and a peak learning rate of 3e-5. The Silver releases were nested. The following retrospective results evaluate surviving checkpoints on the same locked Gold test, separate from the historical development comparison above.

| Training data | Steps | ATCO2 Gold WER | LibriSpeech WER |
| :---- | :---- | :---- | :---- |
| Untouched model | 0 | 75.57% | 3.52% |
| 10 h Silver | 9,791 | 44.02% | 5.46% |
| 50 h Silver | 24,001 | 42.71% | 5.77% |
| 100 h Silver | 20,000 | 43.63% | 5.74% |
| 314.7 h Silver | 62,944 | 41.79% | 6.09% |

The first ten hours produced most of the initial improvement. Expanding to the full 314.7 hours reduced WER by another 2.23 points under this recipe. These runs used different step budgets, so they are not a pure test of data volume. They did, however, give us a practical reason to investigate the training recipe before adding still more audio.

# **Experiment 2: Better ATC accuracy came with a cost**

Raising Nemotron’s learning rate to `1e-4` reduced ATCO2 WER to 32.12%, but LibriSpeech deteriorated to 16.88%. Looking at ATC alone would have made this seem like an unqualified success. The English evaluation showed the cost of that specialization.

Parakeet CTC 1.1B trained on the same Silver pool reached 26.78% ATCO2. Changing the model and its recipe clearly deserved attention, although this comparison cannot isolate architecture from model size and pretraining. Its LibriSpeech WER rose to 34.63%, again exposing substantial forgetting.

General-English replay addressed that trade-off. Nemotron replay runs held LibriSpeech near 3.39% while reaching approximately 36% ATCO2. Parakeet CTC with equal ATC and English sampling reached 32.81% ATCO2 and 2.15% LibriSpeech.

Parakeet TDT gave us the most useful balance for the next stage. It sampled 80% from a 314.716-hour Silver pool audited for leakage and 20% from a 314.721-hour English pool. It reached 30.28% on ATCO2 and 2.18% on LibriSpeech. We used this checkpoint as the starting point for Gold refinement.

# **Experiment 3: Gold worked best as a second-stage correction**

We compared four strategies using Parakeet TDT, 2,000 steps, and peak LR `1e-5`.

| Strategy | Starting point and data | ATCO2 WER | LibriSpeech WER |
| :---- | :---- | :---- | :---- |
| G1 | Untouched model; 0.418 h Gold only | 26.44% | 10.91% |
| G2 | Untouched model; Gold \+ English replay | 24.30% | 2.67% |
| G3 | Silver-adapted model; Gold \+ English replay | **20.00%** | 2.31% |
| G4 | Silver-adapted model; Silver \+ Gold \+ English | 27.38% | **2.24%** |

G1 showed that even 25 minutes of matching human data could adapt the untouched model, but it also caused severe forgetting. G2 restored general English but left cross-ATC performance weak.

G3 combined what the earlier runs had taught us. We first adapted to the broad Silver corpus, then refined that model on Gold examples at a lower learning rate while retaining English replay. ATCO2 WER fell from 30.28% to 20.00%, with LibriSpeech remaining close to the 2.18% starting value.

G4 continued sampling the large Silver pool during Gold refinement and performed much worse. This suggests that continued exposure to automatic labels weakened the correction provided by human annotations. In these experiments, Silver helped establish broad domain knowledge, while the final stage benefited from greater emphasis on Gold.

Repeating G3 with another random seed produced a similar improvement, giving us more confidence that the recipe was responsible.

# **Experiment 4: The final gains came from decoding**

Beam width four reduced the G3 seed-42 result from 20.00% to 19.08% without changing model weights. A paired 5,000-sample bootstrap placed the improvement between 0.68 and 1.15 WER points at 95% confidence; no resample favored greedy decoding.

We also averaged the three checkpoints with the best validation scores. The average improved Gold development WER, so we selected it for the LM experiments; its test WER was 19.21%, slightly above the individual checkpoint's 19.08%. Giving difficult examples more training weight, adding gain variation and white noise, and boosting 200 ATC phrases did not improve on the selected approach.

We next tested whether a language model could help resolve the remaining transcription ambiguities.

## **Why an n-gram model after acoustic fine-tuning?**

The ASR model scores candidate transcriptions using the audio and its learned context. An external n-gram language model adds a preference based on short token sequences observed in text. During shallow fusion, the decoder combines the ASR and LM scores. A weight called alpha controls how strongly the external LM influences the choice.

Consider the words following a runway clearance. ATC uses a relatively small set of recurring phrases, so some continuations are much more likely than others. An LM can use those patterns to help choose between acoustically similar candidates. It provides additional evidence, although a strong preference for familiar phrases can also steer the decoder away from what was actually said.

This is an economical experiment because the ASR checkpoint stays fixed and LM training requires only text. It lets us test an additional source of improvement without running another acoustic fine-tuning job.

## **Constructing the LM corpora without test leakage**

The first corpus contained only the 393 ATCO2 Gold training transcripts: 4,761 normalized tokens. We also constructed an 80% Gold / 20% general-English corpus by token count to test whether an LM replay mixture would preserve ordinary English.

The normalization matched our WER contract: case and diacritic folding, symbol normalization while retaining apostrophes, and whitespace collapse. Gold development and community-test transcripts were used only for identifier-level isolation checks; their text never entered LM training. Silver ATCO2 was also excluded.

For the second round, we added the official Jacktol training text. After removing unusable rows, it contributed 6,495 lines and 65,807 tokens. Because the ATCO2 Gold corpus was tiny, we repeated its training lines to create a token-balanced corpus rather than allowing Jacktol to dominate. The result contained 132,461 tokens: 50.32% ATCO2 Gold and 49.68% Jacktol. Jacktol validation and test transcripts remained excluded.

Repeating the Gold lines increased their contribution to the n-gram counts; it did not increase the amount of unique text. Recording that distinction matters when describing the corpus and reproducing its balance.

## **Selecting order and fusion weight**

We trained three-gram and four-gram KenLM models and converted them for offline NeMo TDT decoding. The acoustic model was the frozen, top-three-averaged G3 finalist. Ordinary beam-4 decoding scored 18.77% on Gold development. Switching to the MALSD decoder without an LM scored 19.40%, so every LM candidate first had to recover the cost of changing decoder.

The first sweep evaluated 24 combinations of corpus, n-gram order, and alpha. A Gold-only four-gram with alpha `0.1` scored best at 17.96% development WER, an improvement of 0.81 points over ordinary beam decoding. Before testing that configuration on held-out ATCO2, we checked English: LibriSpeech WER changed from 2.32% to 2.48%, below our predefined 2.52% maximum. The selected Gold-only LM then reached **18.12%** on ATCO2, compared with 19.21% for the frozen acoustic finalist.

With Jacktol text added, we evaluated 20 further configurations against the Gold-only LM. A balanced four-gram at alpha `0.2` produced the lowest ATCO2 development WER, 17.15%, but raised LibriSpeech to 2.80%. That exceeded our accepted regression limit, so we rejected it before final testing.

We stepped down to alpha `0.1`. On Jacktol validation, it improved the Gold-only LM from 21.15% to 19.49%, while LibriSpeech reached 2.43% and passed the guardrail. Only then did we freeze the configuration and run the final tests.

| Decoder configuration | ATCO2 locked test | Jacktol test | LibriSpeech test-clean |
| :---- | :---- | :---- | :---- |
| Frozen G3 acoustic finalist, beam 4 | 19.21% | 20.36% | 2.32% |
| Gold-only four-gram, alpha 0.1 | 18.12% | 20.05% | 2.48% |
| Gold \+ Jacktol balanced four-gram, alpha 0.1 | **17.61%** | **18.70%** | 2.43% |

Using the unrounded scores, the balanced LM reduced WER by 1.61 absolute points on ATCO2 and 1.67 points on Jacktol without updating the ASR weights. The choice of alpha illustrates a useful principle: select decoder settings against the full objective. The lowest domain development score was insufficient if ordinary English suffered too much.

These measurements came from an offline NeMo pilot. To deploy the approach through Riva, we would rebuild the approved corpus in its supported word-level LM format and evaluate the running service. The orchestration skill distinguishes these stages because serving can introduce a different decoder and a different accuracy profile.

# **External reference: the Jacktol result**

We also verified that our setup could reproduce the public Jacktol benchmark. Our selected checkpoint reached **5.93% Jacktol WER**, closely matching the published 5.99% result; the public checkpoint scored 6.06% with our local normalizer. This result provides useful external context, but the main study remained focused on improving ATCO2 under its locked Gold and general-English evaluation contract.

# **Learning from the experiments that did not help**

Several plausible changes fell short in this study:

* post-hoc feature-normalization changes violated the checkpoint contract;
* increasing Silver scale without changing the recipe plateaued;
* high-learning-rate and ATC-only runs hid catastrophic forgetting;
* continuing Silver replay diluted final Gold correction;
* generic noise did not reproduce the remaining radio errors;
* hard-example oversampling reduced overall development accuracy;
* phrase boosting did not help the selected TDT checkpoint;
* the LM weight giving the lowest development WER exceeded the permitted English regression.

These outcomes helped us decide where to spend the next training or evaluation budget. They also kept the conclusions specific: a technique that failed with this data and checkpoint might still help elsewhere.

# **The practical recipe that emerged**

For a domain with abundant weak labels and scarce trusted references, our evidence supports six steps:

1. Freeze leakage-safe Gold development and test sets, then measure the untouched checkpoint on domain and general speech.
2. Use Silver data for broad channel adaptation, with general-domain replay when existing capability must be preserved.
3. Refine the balanced checkpoint on scarce Gold data at a lower learning rate and repeat the recipe across seeds.
4. Select and export checkpoints before opening the locked test.
5. Evaluate beam search, averaging, and n-gram fusion as separate controlled stages.
6. Report domain WER, transfer results, and the forgetting guardrail together.

The judgment in fine-tuning lies in choosing what to investigate next. A plateau may reflect noisy labels, an unsuitable learning rate, or a decoder that needs more domain context. The measurements help distinguish these explanations, and the next experiment should make that distinction clearer.

The skills help carry those decisions through the project. The orchestration skill keeps the objective and evaluation criteria in view; the NeMo skill guides the training and scoring needed to test each idea. That leaves the next engineer with a useful starting point: a model, the evidence behind it, and a clear account of what remains to be learned.

## **Apply the lessons to your own domain**

The historical sequence was broad Silver adaptation, English replay, low-learning-rate Gold refinement, then decoder experiments. New ATCO2 work uses the Gold-only policy; the historical Silver results explain the study rather than authorize new Silver training.

Use development data to choose the next experiment, retain a general-English guardrail, and evaluate the exported artifact. A plateau can reflect label quality, optimization, or decoding. The skills keep the configurations and measurements consistent while you test those explanations.

## **ATC02 Dataset**

We are releasing a small subset of the larger ATC02 dataset for evaluation purposes. This dataset can be used to evaluate models that the community further fine-tunes.

NVIDIA gratefully acknowledges ELDA (Evaluations and Language resources Distribution Agency) for making the ATCO2 Project Data available to train and test this model checkpoint. The ATCO2 corpus supports research and development in air-traffic-control speech technologies, including automatic speech recognition, speaker diarization, language detection, and related speech and language understanding tasks. The dataset includes air-traffic-control voice communications and associated metadata collected and processed through the ATCO2 project. A 2-hour subset of the larger 5K hour dataset is linked here for evaluation purposes only. To learn more about this dataset, visit [https\://catalog.elra.info/](https://catalog.elra.info/) (Catalogue Reference: ELRA-S0484, ISLRN: 589-403-577-685-7).

## **Try the workflow**

Install the [orchestration skill](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) and [training skill](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune), inspect the [example](https://github.com/fciannella/atc-asr-finetuning-experiments/tree/main/experiments/nemotron-mixed-2to1), and start with licensed domain data or [Jacktol](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset). Measure both domain gains and general-English retention.
