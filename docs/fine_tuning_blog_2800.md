# Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills

A speech recognition model can understand everyday English yet struggle with an air-traffic-control (ATC) clearance. Radio noise, accents, callsigns, and compressed phraseology change both sound and language. Fine-tuning adapts a pretrained model to those differences while testing whether existing capabilities survive.

Our study used two NVIDIA agent skills to guide data preparation, training, and evaluation. We follow the experiments from automatic labels to human annotations and language-model fusion. Recognition quality is measured with word error rate (WER): substitutions, deletions, and insertions divided by reference words. Lower is better.

## How the skills guide the work

A skill is a reusable instruction set for a coding agent. [`nemotron-asr-finetune`](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) establishes the objective and baseline, then considers word boosting, language-model fusion, or acoustic fine-tuning. [`nemo-speech-asr-finetune`](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune) prepares data, preserves checkpoint contracts, configures training, exports candidates, and evaluates them. Together, they connect each experiment to a measurable objective.

## Prerequisites

You need a coding agent supporting Agent Skills, Node.js/npm, Git, and a Linux GPU host with NVIDIA drivers, NVIDIA Container Toolkit, and a compatible NeMo ASR environment. Prepare a licensed `.nemo` checkpoint, readable audio, separate training/development/test manifests, and storage for checkpoints. JSONL rows need `audio_filepath`, `duration`, and `text`.

The linked historical run used eight GPUs, BF16, and NeMo `2.8.0rc0`; that is its recorded setup, not a minimum for every pilot. Pin the software and model versions and check available GPU memory before training. Skills guide the work; they do not provision hardware or install the training environment.

## Install and activate the skills

Follow the [NVIDIA skills installation flow](https://github.com/NVIDIA/skills#quickstart):

```bash
npx skills@latest add NVIDIA/skills --skill nemotron-asr-finetune
npx skills@latest add NVIDIA-NeMo/Speech --skill nemo-speech-asr-finetune
npx skills@latest list
```

Choose your agent and installation scope, then start a fresh agent session and confirm both skills are available.

Explicitly name both in a prompt. This sample is illustrative; replace the paths with your files:

> Use `nemotron-asr-finetune` and `nemo-speech-asr-finetune` to adapt `/models/base.nemo` to ATC. Human-verified training and development manifests are `/data/atc/train.jsonl` and `/data/atc/dev.jsonl`; keep `/data/atc/test.jsonl` locked until selection is complete. English replay is `/data/english/train.jsonl`; the general evaluation is `/data/librispeech/test-clean.jsonl`. Audit transcript style and overlap, inspect the checkpoint and GPUs, and measure baseline WER. If acoustic training is justified, propose a conservative pilot with replay. Preserve tokenizer and preprocessing. Save the effective configuration, manifest hashes, exported model, predictions, and baseline-versus-adapted WER report in `/exp/atc`. Report missing inputs before launching training.

## Choose the architecture as part of the product decision

| Architecture | Basic idea | Training constraint |
| --- | --- | --- |
| CTC | Frame-level tokens followed by blank/repetition collapse | Preserve head and tokenizer |
| RNN-T | Acoustic encoder plus prediction and joint networks | Preserve transducer loss and streaming configuration |
| TDT | Transducer with token-and-duration outputs | Preserve duration vocabulary and decoder settings |

We tested Nemotron 3.5 ASR Streaming 0.6B, Parakeet CTC 1.1B, and Parakeet TDT 0.6B v3. Parakeet is a separate model family in this comparison. Choose a compatible recipe and the latency/serving mode your application needs.

Keep the pretrained tokenizer and preprocessing for an initial same-language adaptation. Our post-hoc change from global to per-feature normalization sharply worsened Nemotron WER. If replacing a tokenizer, use training text only.

## Know which data is being released

The documented community-release plan covers **two hours of human-annotated ATCO2 evaluation data**, containing 1,908 segments. It does not currently include the study's 0.418-hour Gold training set, 0.100-hour Gold development set, or licensed 314.7-hour Silver training corpus. The download URL, license, and final packaging remain pending release-owner confirmation. Keep the community evaluation data out of training and model selection.

For a public starting point, the [Jacktol ATC-ASR dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) provides official training, validation, and test splits under its dataset-card terms. Alternatively, supply your own licensed recordings. Audit overlap before comparing Jacktol with ATCO2. New data means a new experiment, not reproduction of the historical scores.

## ATCO2 and Jacktol provide different evidence

Filtering the approximately 4,281.9-hour ATCO2 delivery produced 314.721 hours of English Silver data: 396,461 segments with selected automatic CNET hypotheses. Human-Gold data had separate roles:

| Split | Audio | Segments | Role |
| --- | ---: | ---: | --- |
| Gold training | 0.418 h | 393 | Supervised refinement |
| Gold development | 0.100 h | 100 | Checkpoint and decoder selection |
| Gold community test | 2.000 h | 1,908 | Locked evaluation |

The splits had no overlap by airport-date, audio path, record ID, or source recording. Public Jacktol supplied a separate comparison and training-only domain text. Its ATCO2-derived material required an acoustic overlap audit. These internal study splits do not expand the planned public release.

## Inspect and run the training configuration

The [Nemotron 2:1 replay example](https://github.com/fciannella/atc-asr-finetuning-experiments/tree/main/experiments/nemotron-mixed-2to1) contains an [effective training configuration](https://github.com/fciannella/atc-asr-finetuning-experiments/blob/main/experiments/nemotron-mixed-2to1/config.yaml), executable `recipe.sh`, sampling weights, logs, and results. **This repository is currently private; public release is pending.** The container digest and base-checkpoint download link still need to be pinned for release.

The archived run mixed 314.721 hours of Silver ATCO2 with 314.721 hours of English, sampled 2:1. It used 40,000 optimizer steps, learning rate `1e-4`, and 400 warmup steps. Validation retained the top five checkpoints plus last. The published comparison evaluated the final exported 40,000-step model, separately from the best in-training checkpoint.

Authorized readers set `NEMO_ROOT`, `BASE_MODEL`, `ATC_TRAIN_MANIFEST`, `GENERAL_TRAIN_MANIFEST`, `ATC_DEV_MANIFEST`, and `OUTPUT_DIR`, then run `bash recipe.sh`. The YAML describes the settings; the shell script executes them. The example README supplies the path-setting commands and checkpoint-resume procedure.

This is a historical Silver-data recipe. New ATCO2 development follows the [Gold-only policy](atco2-gold-policy.md). A small Gold pilot needs its own conservative learning rate, batch sizing, and step budget; copying a 40,000-step recipe unchanged would be inappropriate.

## Evaluate both models and inspect the outputs

Run standalone inference with the untouched and exported fine-tuned checkpoints on identical domain and general-English manifests. Development selects checkpoints and decoder settings; the locked test is evaluated only after selection. The historical table below uses development data, not the community test.

For the Nemotron streaming checkpoint, use the [NeMo inference script](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py). In the GPU container, set `NEMO_ROOT`, `BASE_MODEL`, `FINETUNED_MODEL`, `ATC_EVAL_MANIFEST`, and `GENERAL_EVAL_MANIFEST` to real paths. Use the historical ATC development manifest for the linked comparison and LibriSpeech test-clean for general evaluation. Set `STUDY_ROOT` to this study checkout and `EVAL_DIR` to a new output directory. Run in Bash:

```bash
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

```bash
python "$STUDY_ROOT/scripts/score_wer.py" /actual/predictions.jsonl \
  --output /actual/wer.json
```

The [scorer](../scripts/score_wer.py) folds case and diacritics, normalizes symbols while retaining apostrophes, and divides total word edits by total reference words. Use the same implementation for both models and verify matching sample counts.

Expected artifacts are configuration and manifest hashes, training logs, retained checkpoints, an exported `.nemo`, and four prediction/score pairs. Each `wer.json` includes utterances, reference words, word errors, and `normalized_wer` as a fraction; multiply by 100 for percentages. Save decoder settings and checkpoint hashes with the report.

## What fine-tuning changed

| Evaluation | Untouched Nemotron | Fine-tuned Nemotron |
| --- | ---: | ---: |
| Historical ATC development: 1,007 utterances | 75.75% WER | **37.38% WER** |
| LibriSpeech test-clean: 2,620 utterances | 3.52% WER | **3.39% WER** |

The [unrounded results](https://github.com/fciannella/atc-asr-finetuning-experiments/blob/main/experiments/nemotron-mixed-2to1/results.yaml) show a **38.37-percentage-point reduction**, or **50.65% relative reduction**, in ATC development WER. These are observed results, not promised outcomes. This development comparison is distinct from the later two-hour locked ATCO2 test.

## Experiment 1: What did additional Silver data teach the model?

We began with `nemotron-3.5-asr-streaming-0.6b.nemo` and a peak learning rate of `3e-5`. The Silver releases were nested. The following retrospective results evaluate surviving checkpoints on the same locked Gold test, separate from the historical development comparison above.

| Training data | Steps | ATCO2 locked Gold test WER | LibriSpeech WER |
| --- | ---: | ---: | ---: |
| Untouched model | 0 | 75.57% | 3.52% |
| 10 h Silver | 9,791 | 44.02% | 5.46% |
| 50 h Silver | 24,001 | 42.71% | 5.77% |
| 100 h Silver | 20,000 | 43.63% | 5.74% |
| 314.7 h Silver | 62,944 | 41.79% | 6.09% |

The first ten hours produced most of the initial improvement. Expanding to the full 314.7 hours reduced WER by another 2.23 points under this recipe. These runs used different step budgets, so they are not a pure test of data volume. They did, however, give us a practical reason to investigate the training recipe before adding still more audio.

## Experiment 2: Better ATC accuracy came with a cost

Raising Nemotron’s learning rate to `1e-4` reduced ATCO2 WER to 32.12%, but LibriSpeech deteriorated to 16.88%. Looking at ATC alone would have made this seem like an unqualified success. The English evaluation showed the cost of that specialization.

Parakeet CTC 1.1B trained on the same Silver pool reached 26.78% ATCO2. Changing the model and its recipe clearly deserved attention, although this comparison cannot isolate architecture from model size and pretraining. Its LibriSpeech WER rose to 34.63%, again exposing substantial forgetting.

General-English replay addressed that trade-off. Nemotron replay runs held LibriSpeech near 3.39% while reaching approximately 36% ATCO2. Parakeet CTC with equal ATC and English sampling reached 32.81% ATCO2 and 2.15% LibriSpeech.

Parakeet TDT gave us the most useful balance for the next stage. It sampled 80% from a 314.716-hour Silver pool audited for leakage and 20% from a 314.721-hour English pool. It reached 30.28% on ATCO2 and 2.18% on LibriSpeech. We used this checkpoint as the starting point for Gold refinement.

## Experiment 3: Gold worked best as a second-stage correction

We compared four strategies using Parakeet TDT, 2,000 steps, and peak LR `1e-5`.

| Strategy | Starting point and data | ATCO2 WER | LibriSpeech WER |
| --- | --- | ---: | ---: |
| G1 | Untouched model; 0.418 h Gold only | 26.44% | 10.91% |
| G2 | Untouched model; Gold + English replay | 24.30% | 2.67% |
| G3 | Silver-adapted model; Gold + English replay | **20.00%** | 2.31% |
| G4 | Silver-adapted model; Silver + Gold + English | 27.38% | **2.24%** |

G1 showed that even 25 minutes of matching human data could adapt the untouched model, but it also caused severe forgetting. G2 restored general English but left cross-ATC performance weak.

G3 combined what the earlier runs had taught us. We first adapted to the broad Silver corpus, then refined that model on Gold examples at a lower learning rate while retaining English replay. ATCO2 WER fell from 30.28% to 20.00%, with LibriSpeech remaining close to the 2.18% starting value.

G4 continued sampling the large Silver pool during Gold refinement and performed much worse. This suggests that continued exposure to automatic labels weakened the correction provided by human annotations. In these experiments, Silver helped establish broad domain knowledge, while the final stage benefited from greater emphasis on Gold.

Repeating G3 with another random seed produced a similar improvement, giving us more confidence that the recipe was responsible.

## Experiment 4: The final gains came from decoding

Beam width four reduced the G3 seed-42 result from 20.00% to 19.08% without changing model weights. A paired 5,000-sample bootstrap placed the improvement between 0.68 and 1.15 WER points at 95% confidence; no resample favored greedy decoding.

We also averaged the three checkpoints with the best validation scores. The average improved Gold development WER, so we selected it for the LM experiments; its test WER was 19.21%, slightly above the individual checkpoint's 19.08%. Giving difficult examples more training weight, adding gain variation and white noise, and boosting 200 ATC phrases did not improve on the selected approach.

We next tested whether a language model could help resolve the remaining transcription ambiguities.

### Add domain text without changing the acoustic model

An external n-gram language model scores short token sequences, helping the decoder choose between acoustically similar phrases. Shallow fusion combines its score with the ASR score; alpha controls its influence. The acoustic weights remain fixed.

We first trained three-gram and four-gram models using only the 393 Gold training transcripts, with an English-text blend as a control. Development and test transcripts never entered LM training. A 24-configuration sweep selected a Gold-only four-gram at alpha `0.1`: development WER improved from 18.77% for ordinary beam decoding to 17.96%. LibriSpeech reached 2.48%, passing our predefined 2.52% guardrail, before the selected decoder was tested on locked ATCO2.

Next we added official Jacktol training text, excluding its validation and test text. Repeating the scarce Gold lines produced a 132,461-token corpus balanced approximately 50:50 between Gold and Jacktol. This changed counts, not unique text. The overlap audit remained part of the data contract.

A second sweep found that alpha `0.2` improved ATCO2 development most, but worsened LibriSpeech to 2.80%. We rejected it before final testing. Alpha `0.1` passed the English guardrail at 2.43% and improved Jacktol validation. We then froze the decoder and evaluated the held-out sets:

| Decoder configuration | ATCO2 locked test | Jacktol test | LibriSpeech test-clean |
| --- | ---: | ---: | ---: |
| Frozen G3 acoustic finalist, beam 4 | 19.21% | 20.36% | 2.32% |
| Gold-only four-gram, alpha 0.1 | 18.12% | 20.05% | 2.48% |
| Gold + Jacktol balanced four-gram, alpha 0.1 | **17.61%** | **18.70%** | 2.43% |

Using the unrounded scores, the balanced LM reduced WER by 1.61 absolute points on ATCO2 and 1.67 points on Jacktol without updating the ASR weights. The choice of alpha illustrates a useful principle: select decoder settings against the full objective. The lowest domain development score was insufficient if ordinary English suffered too much.

These measurements came from an offline NeMo pilot. To deploy the approach through Riva, we would rebuild the approved corpus in its supported word-level LM format and evaluate the running service. The orchestration skill distinguishes these stages because serving can introduce a different decoder and a different accuracy profile.

## External reference: the Jacktol result

We also verified that our setup could reproduce the public Jacktol benchmark. Our selected checkpoint reached **5.93% Jacktol WER**, closely matching the published 5.99% result; the public checkpoint scored 6.06% with our local normalizer. This result provides useful external context, but the main study remained focused on improving ATCO2 under its locked Gold and general-English evaluation contract.

## Learning from the experiments that did not help

Post-hoc normalization changes, continued Silver mixing during Gold refinement, generic noise augmentation, hard-example oversampling, and phrase boosting failed to improve the selected approach. The strongest domain LM weight also failed the English guardrail. These outcomes narrowed the next experiment; they do not establish that those techniques fail for every dataset.

## Apply the lessons to your own domain

The historical sequence was broad Silver adaptation, English replay, low-learning-rate Gold refinement, then decoder experiments. New ATCO2 work uses the Gold-only policy; the historical Silver results explain the study rather than authorize new Silver training.

Use development data to choose the next experiment, retain a general-English guardrail, and evaluate the exported artifact. A plateau can reflect label quality, optimization, or decoding. The skills keep the configurations and measurements consistent while you test those explanations.

## Try the workflow

Install the [orchestration skill](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) and [training skill](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune), inspect the [example](https://github.com/fciannella/atc-asr-finetuning-experiments/tree/main/experiments/nemotron-mixed-2to1), and start with licensed domain data or [Jacktol](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset). Measure both domain gains and general-English retention.

<!-- Publication handoff: add the approved ATCO2 evaluation dataset URL/license and public example/scoring-code URLs. Confirm the planned evaluation-only scope with the release owner. Nemotron positioning remains with the messaging owner. Before livestream: add “Join the upcoming livestream to walk through the workflow” only with a confirmed event link. After livestream: replace with “Watch the walkthrough” and the recording URL. Do not publish this private-link draft as launch-ready. -->
