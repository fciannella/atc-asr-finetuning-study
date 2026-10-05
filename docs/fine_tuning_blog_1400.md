# Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills

Fine-tuning automatic speech recognition (ASR) is easy to summarize: start from a pretrained model, provide transcribed audio from a new domain, and update the model weights. Doing it well is harder. Data quality, model architecture, transcript conventions, learning rate, decoding, and evaluation all interact. A model can improve dramatically on specialized speech while quietly becoming worse at ordinary English.

This tutorial explains how two NVIDIA agent skills turn that open-ended process into a controlled workflow. We use air-traffic-control (ATC) radio as the example because it combines a difficult acoustic channel with specialized language: noise, clipping, accents, callsigns, runways, altitudes, frequencies, and terse phraseology.

## Two skills, two responsibilities

The [`nemotron-asr-finetune`](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) skill is the orchestration layer. It defines the objective, establishes a baseline, and chooses the least expensive customization likely to solve the measured problem. Its ladder starts with word boosting and custom vocabulary, advances to n-gram language-model fusion, and uses acoustic-model fine-tuning when the mismatch includes channel conditions, accents, or noise.

The [`nemo-speech-asr-finetune`](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune) skill executes the NeMo workflow. It identifies the checkpoint family, prepares Lhotse-compatible data, preserves tokenizer and architecture contracts, configures training, retains useful checkpoints, and runs standalone WER evaluation.

The distinction matters. The first skill asks what should change and how success will be measured. The second makes that change correctly for a particular NeMo model.

## The ATC data contract

Our ATCO2 delivery contained approximately 4,281.9 hours across more than three million recordings. Most of it was not equally suitable for supervised training. After filtering for language, duration, confidence, text, and audio quality, we produced a 314.721-hour English Silver set containing 396,461 segments. “Silver” means the transcripts were selected automatic CNET hypotheses, not human-verified references.

Human-Gold data was much scarcer. We isolated 0.418 hours for training, 0.100 hours for development, and 2.000 hours for a locked community test. These splits had no overlap by airport-date, audio path, record ID, or source recording.

We also used the public [Jacktol ATC-ASR dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset). Its training split contains about 5.9 hours and is useful as a compact supervised source and external comparison. An overlap audit was essential because Jacktol contains material derived from public ATCO2 recordings.

## A five-step workflow

### 1. Define “better” before training

We froze two objectives: reduce normalized WER on ATCO2 Gold while limiting regression on LibriSpeech test-clean. Domain WER measured specialization; LibriSpeech acted as the catastrophic-forgetting guardrail.

### 2. Version the data

Every sample retained its source, label class, split, and grouping identity. Related segments were kept together when splitting. We normalized transcript style consistently and fingerprinted the final manifests. Development and test text was excluded from tokenizer and language-model construction.

### 3. Preserve the model contract

We tested Nemotron 3.5 ASR Streaming 0.6B, Parakeet CTC 1.1B, and Parakeet TDT 0.6B v3. CTC, RNN-T, and TDT checkpoints are not interchangeable recipes: their decoders, losses, tokenizers, and feature processing must remain consistent with the pretrained model.

### 4. Train and retain candidates

We used maximum optimizer steps rather than assuming a fixed number of epochs. Validation WER selected candidate checkpoints, but final claims came from reloading exported artifacts and running standalone evaluation. General-English replay remained in the mixture when preservation mattered.

### 5. Change one interpretable factor at a time

Successive experiments varied data scale, learning rate, architecture, replay ratio, Gold refinement, and decoding separately. This made negative outcomes useful instead of mysterious.

## What the experiments showed

The untouched Nemotron model scored 75.57% WER on the locked ATCO2 Gold test. Fine-tuning on the first 10 Silver hours reduced it to 44.02%. Expanding to the complete 314.7-hour Silver set reached 41.79% under the same conservative learning rate. Most of the initial domain gain therefore arrived early; adding hours alone produced diminishing returns.

Optimization and architecture changed the frontier. A higher Nemotron learning rate reached 32.12% ATCO2 WER, but LibriSpeech deteriorated to 16.88%. Parakeet CTC trained only on Silver reached 26.78% ATCO2, but its LibriSpeech WER rose to 34.63%. Both looked impressive if we ignored forgetting.

English replay corrected that problem. A Parakeet TDT mixture sampled from 314.716 hours of audited Silver ATCO2 and 314.721 hours of English reached 30.28% ATCO2 while retaining 2.18% LibriSpeech WER. This balanced Silver model became the parent for Gold refinement.

We then refined it for 2,000 steps using only 0.418 hours of human-Gold ATCO2 training audio plus English replay. The result reached 20.00% on the locked ATCO2 test and 2.31% on LibriSpeech. Repeating the recipe with another seed produced a similar result. By contrast, mixing the large Silver pool back into this final stage weakened the Gold correction.

Decoder work delivered the final gains. Beam width four reduced the Gold-refined model from 20.00% to 19.08%. A four-gram language model built only from allowed training text reduced the development-selected acoustic model from 19.21% to 17.61% on the locked test while keeping LibriSpeech at 2.43%. No acoustic retraining was required.

We also reproduced the public Jacktol result. A staged curriculum using Jacktol, UWB ATC, and LibriSpeech reached 5.93% Jacktol WER, closely matching the published 5.99%. A flat, single-stage imitation failed, demonstrating that a model name, dataset, and learning rate do not constitute a complete recipe.

## Fine-tuning is experiment design

The ATC study produced four reusable lessons.

First, weakly labeled data and human-Gold data serve different purposes. Silver data taught broad acoustic and domain structure; scarce Gold labels worked best as a precise second-stage correction.

Second, domain accuracy needs a general-domain guardrail. Higher learning rates and ATC-only training created strong specialists, but replay was necessary when the model also had to recognize ordinary English.

Third, training order matters. Broad adaptation followed by low-learning-rate Gold refinement outperformed both Gold-only training and continued Silver mixing.

Finally, fine-tuning does not end with the last optimizer step. Checkpoint selection, beam search, overlap audits, and n-gram fusion all changed the final outcome.

That is why ASR adaptation feels partly like an art. The goal is not to search settings blindly. It is to choose the next experiment that distinguishes among competing explanations, while the skills preserve the contracts that make every result reproducible.
