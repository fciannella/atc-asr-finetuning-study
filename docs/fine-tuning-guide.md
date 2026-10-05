# How to Fine-Tune ASR for a New Domain

## A skill-guided case study in air-traffic-control speech

Fine-tuning an automatic speech recognition model is easy to describe: take a pretrained model, show it transcribed examples from a new domain, and update its weights. Doing it well is more complicated. The model architecture, audio conditions, transcript conventions, data split, tokenizer, optimizer, decoding strategy, and success criteria all interact. A model can improve dramatically on a domain test set while quietly becoming much worse at ordinary speech.

This article explains that complete process. We use air-traffic-control (ATC) radio as the running example and two reusable NVIDIA fine-tuning skills as the guide rails. The goal is not to present one magic configuration. It is to show how to turn an adaptation request into a controlled, reproducible experiment—and how to reason about the trade-offs that appear along the way.

The experimental results will follow in the next section of the article. First, we need to establish the models, data, workflow, and evaluation contract that make those results meaningful.

## What the fine-tuning skills are

The skills are not model checkpoints, training frameworks, or hidden services. They are reusable operating procedures for an AI coding agent. They capture the questions to ask, the order in which decisions should be made, the checks that should happen before expensive training, and the evidence required before a model can be called better.

We used two complementary skills:

- [`nemotron-asr-finetune`](https://github.com/nvidia-riva/Nemotron-speech-skills/tree/main/skills/nemotron-asr-finetune) is the orchestration layer. It scopes the adaptation objective, selects the least expensive technique likely to work, routes the task to the appropriate specialist, and keeps domain accuracy, general accuracy, latency, and deployment constraints visible throughout the project.
- `nemo-speech-asr-finetune` is the training specialist. It handles the concrete NeMo workflow: identifying the checkpoint family, preparing Lhotse-compatible data, preserving architecture and tokenizer contracts, constructing the training recipe, retaining useful checkpoints, and running standalone evaluation.

Locally, these skills are installed under `~/.agents/skills/nemotron-asr-finetune/` and `~/.agents/skills/nemo-speech-asr-finetune/`. Separating orchestration from execution is intentional. One skill asks, “What should we change, and how will we know it worked?” The other answers, “How do we train and evaluate this particular NeMo ASR model correctly?”

### Fine-tuning is not always the first rung

The orchestration skill treats ASR customization as a ladder:

1. Word boosting for a small list of names or terms.
2. Custom vocabulary and pronunciation support.
3. An external n-gram language model for broader phrase and spelling bias.
4. Acoustic-model fine-tuning for a true domain or acoustic mismatch.
5. Training from scratch only when the earlier rungs are insufficient.

These methods can also be combined. A fine-tuned acoustic model may still benefit from an n-gram language model. The important lesson is that a request to “fine-tune the model” should first be translated into a measurable failure mode. If the problem is only ten unusual proper nouns, retraining hundreds of millions of parameters may be unnecessary.

## The model families we considered

An ASR checkpoint is more than a collection of weights. Its encoder, decoder, tokenizer, feature normalization, and loss function form a contract. A recipe written for one family should not be applied blindly to another.

| Architecture | How it predicts text | Practical character | Important fine-tuning constraint |
| --- | --- | --- | --- |
| CTC | Produces frame-level token probabilities and collapses blanks and repetitions | Simple, parallelizable decoding; can work well with external language models | Preserve the CTC head and tokenizer; decoding settings can materially affect final WER |
| RNN-T | Combines an acoustic encoder, a prediction network, and a joint network | Streaming-friendly and conditioned on previous emitted tokens | Preserve the transducer loss and decoder/joint configuration |
| TDT | Extends the transducer family by predicting tokens and their durations | Efficient sequence modeling with an explicit duration vocabulary | Keep the TDT loss, duration bins, and model type consistent |
| Hybrid transducer/CTC | Shares an encoder between a transducer head and a CTC head | Supports multiple decoding paths and useful auxiliary training | Evaluate and report which head produced each score |
| AED / Canary | Uses an encoder-decoder attention model, often with task or language prompts | Flexible multilingual and multitask behavior | Preserve prompt fields, language/task metadata, and decoder conventions |

Our study used three representative pretrained models:

- **Nemotron 3.5 ASR Streaming 0.6B**, a streaming transducer-family model.
- **Parakeet CTC 1.1B**, a larger CTC model well suited to testing architecture and capacity effects.
- **Parakeet TDT 0.6B v3**, a token-and-duration transducer model with its own native feature-processing contract.

There is no universally best architecture. A CTC model may offer convenient offline decoding and language-model integration; a transducer may be the better product choice for low-latency streaming. Model selection must therefore include the intended serving mode, latency budget, memory budget, and deployment target—not only the lowest offline WER.

### The tokenizer is part of the model

For adaptation within the same language and writing system, preserving the pretrained tokenizer is usually the safest starting point. Replacing it changes the output space and can discard useful knowledge encoded in the decoder.

A new tokenizer becomes justified when the target introduces a new script, language, or symbol inventory that the original tokenizer cannot represent adequately. Even then, it must be trained only on training text. Development and test transcripts must never influence the vocabulary.

The same principle applies to audio normalization. A checkpoint trained with global feature normalization cannot safely be converted to per-feature normalization by changing one configuration field after training. Preprocessing is learned context, not a cosmetic switch.

## Why air-traffic-control speech is a useful case study

ATC radio concentrates several hard ASR problems into one domain:

- narrow-band radio, background noise, interference, clipping, and variable gain;
- short, context-dependent transmissions from pilots and controllers;
- accents and non-native English;
- callsigns, runway identifiers, headings, altitudes, frequencies, and spelled letters;
- a constrained but highly specialized phraseology;
- errors in numbers or callsigns that are more consequential than ordinary conversational substitutions.

This makes ATC a good teaching example. The acoustic channel differs from the speech on which a general model is typically trained, and the language distribution differs as well. At the same time, we may want the adapted model to retain ordinary English recognition. The experiment therefore needs two objectives: improve ATC accuracy and limit general-domain regression.

## The ATCO2 and Jacktol data play different roles

Our primary source was the ATCO2 delivery. It is a broad, multi-airport corpus rather than a single clean benchmark. The complete delivery contains 3,088,603 recordings and approximately 4,281.9 hours of audio. Most of that material is not equally suitable for supervised training.

After language, duration, confidence, text, and audio-quality filtering, we created a high-quality English **Silver** training release containing 396,461 segments and 314.721 hours. “Silver” is important: these are selected CNET top-hypothesis transcripts, not human-verified references. The corpus also contains a much smaller **Gold** pool whose transcripts were accepted by human annotators. Gold data is scarce and is therefore reserved carefully among training, development, and the locked community test.

The publicly available [`jacktol/ATC-ASR-Dataset`](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) serves a different purpose. Its official training split is compact—about 5.9 hours—and it provides published train, validation, and test partitions. It is useful as an external ATC benchmark, a small supervised source, and a source of in-domain training text. We pin a specific dataset revision whenever it is used so that the experiment remains reproducible.

| Property | ATCO2 | Jacktol ATC-ASR |
| --- | --- | --- |
| Availability | Project/vendor delivery | Public Hugging Face dataset |
| Scale | Thousands of raw hours; 314.721-hour filtered Silver release plus scarce Gold data | Compact; about 5.9 hours in the official training split |
| Coverage | Multiple airports with an uneven geographic distribution | A separate, narrower released corpus |
| Primary value | Large-scale domain adaptation and controlled Gold evaluation | External transfer test, compact supervised source, and domain text |
| Main caution | Confidence-filtered pseudo-labels are not equivalent to human Gold | Official validation and test data must remain outside training and language-model construction |

The two datasets are complementary, not interchangeable. Mixing them without provenance labels would hide what the experiment is actually measuring. We retain the source, label class, airport, split, and dataset revision for every sample. We also audit acoustic overlap before calling a result “external.”

## The end-to-end fine-tuning workflow

The skills turn the project into a sequence of explicit decisions.

### 1. State the objective

Start with the product outcome, not the training command. A useful objective might be:

> Reduce normalized WER on the locked ATCO2 Gold test set while keeping LibriSpeech test-clean WER within an agreed regression budget, using a model that can meet the target serving constraints.

This statement identifies the domain metric, general-domain guardrail, evaluation split, and operational constraint. Without it, “better” can change meaning after the results arrive.

### 2. Inventory the evidence

Before training, determine:

- how many hours have reliable transcripts;
- which labels are Gold, Silver, or synthetic;
- whether source recordings cross split boundaries;
- what domain-only text is available;
- which development and test sets are genuinely independent;
- whether the intended model is streaming or offline;
- the acceptable latency, memory, and general-domain regression.

This stage often changes the experiment more than any optimizer setting. One hour of trusted Gold data can be more informative than many hours of noisy pseudo-labels, but using that Gold data for training means it can no longer provide an unbiased test.

### 3. Freeze the evaluation contract

We use three separate evaluation roles:

| Split | Purpose | May influence training? |
| --- | --- | --- |
| Training | Updates model parameters | Yes |
| Development | Selects checkpoints and tunes decoding or LM weights | Indirectly, yes |
| Locked test | Produces the final ATC claim | No |
| General-domain guardrail | Measures catastrophic forgetting | No |

WER normalization must also be frozen. Our default comparison lowercases text, removes punctuation and non-lexical symbols, folds Unicode diacritics, and normalizes whitespace before scoring. The same normalizer must be applied to references and hypotheses.

Training-time validation WER is useful for selecting candidate checkpoints, but it is not the final number. The selected saved model artifact is reloaded and scored in a standalone evaluation. That verifies that the checkpoint can actually be exported and reproduced.

### 4. Prepare the data as a versioned artifact

Audio paths, durations, transcripts, sources, and split assignments are expressed as validated manifests. Lhotse provides a convenient representation for NeMo training, filtering, and weighted multi-source sampling.

Data preparation includes more than converting file formats:

- reject missing, unreadable, or zero-duration audio;
- normalize transcript conventions consistently;
- inspect duration and token distributions;
- preserve callsigns and numbers according to the scoring policy;
- group related segments by source recording before splitting;
- record every exclusion and its reason;
- fingerprint the final manifests so the exact population can be reconstructed.

When general-English replay is used, source weights determine how often ATC and general speech are sampled. This is preferable to physically duplicating audio, and it makes a ratio such as 80% ATC / 20% general English an explicit part of the recipe.

### 5. Adapt the pretrained checkpoint

The specialist skill first confirms the model family and its configuration. It then makes only deliberate changes: training manifests, batch strategy, learning rate, schedule, maximum steps, validation cadence, checkpoint retention, and logging.

For same-language domain adaptation, a conservative recipe normally starts with the original tokenizer, mixed-precision training, and a lower learning rate than pretraining. We use BF16 where supported and avoid automatic mixed-precision policies that silently change precision behavior.

Checkpointing is part of the experiment design. Keeping only the final step assumes that the best model occurs at the end; that is often false. We retain the strongest validation checkpoints, the final checkpoint, and enough metadata to reproduce the selection rule. Checkpoint averaging is evaluated as a candidate technique rather than assumed to help.

### 6. Measure both specialization and forgetting

Every serious run answers at least two questions:

1. Did ATC recognition improve?
2. What capability did the model lose to obtain that improvement?

A domain-only model can achieve a striking ATC score while suffering catastrophic forgetting on ordinary English. Conversely, a heavily replay-balanced model may preserve general speech but leave useful ATC accuracy on the table. Neither is inherently wrong; they are different operating points. The product objective decides which one is acceptable.

### 7. Diagnose before launching the next run

The next experiment should change one interpretable factor whenever possible:

- data scale or label quality;
- ATC/general replay ratio;
- architecture;
- peak learning rate or schedule;
- number of updates;
- checkpoint selection or averaging;
- decoding beam or n-gram LM weight;
- acoustic augmentation.

Changing all of them at once may produce a better number, but it teaches us almost nothing about why.

### 8. Export the complete model contract

A useful output is not just a `.nemo` file. It includes the starting checkpoint and revision, tokenizer, feature processing, manifests and their hashes, training configuration, logs, checkpoint-selection rule, decoder settings, text normalizer, domain and guardrail scores, and the intended deployment profile.

That package is what allows another engineer to reproduce the result or safely continue the work.

## Why fine-tuning is partly an art

Fine-tuning is scientific when the data, hypotheses, and evaluation are controlled. It feels like an art because several valid objectives compete and the best settings are not independent.

A larger learning rate may unlock rapid domain learning but accelerate forgetting. More Silver data may add acoustic variety while also reinforcing transcript errors. Gold data may be the most valuable training material, yet spending it on training reduces the evidence available for unbiased evaluation. A language model may fix domain phraseology without improving acoustic confusions. A checkpoint with the best development WER may fail on the locked test or violate the general-English guardrail.

The craft lies in choosing the next experiment that separates these explanations. The skills help by making that reasoning explicit: start with the least expensive credible intervention, preserve the model’s architectural contract, freeze the evaluation before optimization, keep every dataset role visible, and promote a model only when both its gain and its cost have been measured.

## Where the results begin

With this foundation in place, the results section can be read as a sequence of answered questions rather than a leaderboard: How much did the first ten hours teach us? Did more Silver data continue to help? What changed when we increased the learning rate or switched architecture? Could general-English replay prevent forgetting? How much value came from scarce Gold data, decoding, and an external n-gram language model?

Those experiments—and the lessons they changed—are presented in [What We Learned Fine-Tuning ASR for Air-Traffic Control](experiments-and-results.md).
