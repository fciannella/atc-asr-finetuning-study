# Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills

Fine-tuning automatic speech recognition (ASR) is simple to describe: begin with a pretrained model, show it transcribed examples from a new domain, and update its weights. Doing it well is more complicated. The architecture, audio conditions, transcript conventions, tokenizer, optimizer, decoding strategy, and evaluation contract all interact. A model can improve dramatically on specialized speech while quietly becoming worse at the speech it already understood.

This article explains how to approach that problem with two reusable NVIDIA agent skills. We use air-traffic-control (ATC) radio as a worked example, following the project from a broad weakly labeled corpus through human-Gold refinement and n-gram language-model fusion. The purpose is not to present one magic configuration. It is to show how an adaptation request becomes a controlled, reproducible sequence of experiments.

## Skills are operating procedures, not model checkpoints

The skills used in this study do not contain hidden models or training services. They encode the questions to ask, the order in which decisions should be made, and the evidence required before calling a model better.

The [`nemotron-asr-finetune`](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) skill is the orchestration layer. It scopes the target domain, available data, quality objective, hardware, latency, and deployment constraints. It establishes a measured baseline and selects the least expensive customization likely to solve the observed errors.

Its customization ladder starts with inexpensive runtime controls such as word boosting and custom vocabulary, advances to n-gram language models, and uses acoustic fine-tuning for real channel, accent, or noise mismatches. Training from scratch is the last resort.

The [`nemo-speech-asr-finetune`](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune) skill is the execution specialist. It selects and inspects the checkpoint, prepares Lhotse data inputs, checks transcript style, preserves architecture and tokenizer contracts, configures training, retains checkpoints, and runs standalone evaluation.

The separation is useful. The orchestration skill asks, “What should change, and how will we know it worked?” The NeMo skill answers, “How do we train and evaluate this model family correctly?”

## Choose the architecture as part of the product decision

An ASR checkpoint is more than a weight file. Its encoder, decoder, tokenizer, feature normalization, and loss function form a learned contract.

| Architecture | Basic idea | Practical implication |
| --- | --- | --- |
| CTC | Predict frame-level tokens, then collapse blanks and repetitions | Parallelizable decoding and convenient external-LM integration |
| RNN-T | Combine an acoustic encoder with a prediction and joint network | Streaming-friendly and conditioned on previously emitted tokens |
| TDT | Extend the transducer family with token-and-duration outputs | Efficient sequence modeling with architecture-specific duration settings |
Our experiments used Nemotron 3.5 ASR Streaming 0.6B, Parakeet CTC 1.1B, and Parakeet TDT 0.6B v3. We did not treat their recipes as interchangeable. A CTC model can be attractive for offline decoding and language-model integration; a transducer can be the better serving choice for low-latency streaming. The intended product matters as much as offline WER.

The tokenizer and input normalization are also part of the contract. For same-language adaptation, preserving the pretrained tokenizer is normally safest; development and test text must never train a replacement. Likewise, changing a completed checkpoint from global to per-feature normalization without retraining caused both ATC and general WER to collapse. Preprocessing cannot be mutated safely after training.

## Why ATC is a demanding adaptation problem

ATC speech concentrates several ASR challenges in one domain:

- narrow-band radio, interference, clipping, and variable gain;
- short, context-dependent transmissions;
- accents and non-native English;
- callsigns, runways, headings, altitudes, and frequencies;
- specialized, compressed phraseology;
- number or callsign errors that are more consequential than ordinary conversational substitutions.

The acoustic channel differs from general speech, and so does the language distribution. Yet an adapted model may still need to recognize ordinary English. That produces two objectives: improve ATC recognition and limit general-domain regression.

## ATCO2 and Jacktol provide different evidence

The ATCO2 delivery contained 3,088,603 recordings and approximately 4,281.9 hours. Filtering for language, duration, confidence, text, and audio quality produced a 314.721-hour English Silver release with 396,461 segments. “Silver” means high-confidence CNET hypotheses rather than human-verified references. The smaller human-Gold pool was assigned to non-overlapping roles:

| Split | Audio | Segments | Role |
| --- | ---: | ---: | --- |
| Gold training | 0.418 h | 393 | Final supervised refinement |
| Gold development | 0.100 h | 100 | Checkpoint and recipe selection |
| Gold community test | 2.000 h | 1,908 | Locked final evaluation |

The splits had zero overlap by airport-date, audio path, record ID, and source recording. Reserving two hours for final evaluation left only about 25 minutes for Gold training, but protected the result.

The public [Jacktol ATC-ASR dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) contains 7.405 hours, including about 5.9 training hours and official validation and test splits. It supplied an external comparison and training-only domain text. Because it includes material derived from public ATCO2 recordings, cross-corpus claims required an acoustic overlap audit.

## A reproducible fine-tuning workflow

### 1. State the objective

Begin with the product outcome rather than a training command. Our objective was to reduce normalized WER on the locked ATCO2 Gold test while keeping LibriSpeech test-clean within an acceptable regression budget. This defined both the domain metric and the catastrophic-forgetting guardrail before optimization began.

### 2. Inventory and version the data

Every sample retained its source, label class, split, and grouping identity. We rejected missing audio and empty transcripts, inspected duration and token distributions, normalized transcript conventions, and grouped related segments before splitting. Final manifests were fingerprinted so that the exact population could be reconstructed.

When we used general-English replay, Lhotse sampling weights determined how often each source appeared. The ratio was part of the experiment contract; the audio was not physically copied to manufacture more hours.

### 3. Freeze evaluation

Training data updated model weights. Development data selected checkpoints and decoder settings. The locked test produced the final claim and did not influence selection. LibriSpeech measured forgetting.

All reported WERs used the same normalizer: lowercase, Unicode diacritic folding, punctuation and symbol removal, and whitespace normalization. In-training validation selected candidates, but final quality came from reloading the exported artifact and running standalone evaluation.

### 4. Train conservatively, then diagnose

For same-language adaptation, we preserved the tokenizer and began with a lower learning rate than pretraining. We retained the best validation checkpoints and the final checkpoint rather than assuming the last step was best.

Successive experiments changed one interpretable factor whenever possible: data scale, learning rate, architecture, replay ratio, Gold refinement, checkpoint averaging, decoding, or language-model weight. This made failed experiments useful because they ruled out explanations.

### 5. Export the full model contract

A reproducible result includes more than a `.nemo` file. It records the starting checkpoint and revision, tokenizer, feature processing, manifest hashes, training configuration, checkpoint-selection rule, decoding parameters, text normalizer, domain WER, and general-domain guardrail.

## Experiment 1: Silver scale delivered an early jump

We began with `nemotron-3.5-asr-streaming-0.6b.nemo` and a peak learning rate of `3e-5`. The Silver releases were nested, allowing us to change scale without redefining earlier samples.

| Training data | Steps | ATCO2 Gold WER | LibriSpeech WER |
| --- | ---: | ---: | ---: |
| Untouched model | 0 | 75.57% | 3.52% |
| 10 h Silver | 9,791 | 44.02% | 5.46% |
| 50 h Silver | 24,001 | 42.71% | 5.77% |
| 100 h Silver | 20,000 | 43.63% | 5.74% |
| 314.7 h Silver | 62,944 | 41.79% | 6.09% |

The first ten hours produced the decisive domain jump. Adding another 304.7 hours improved ATCO2 by only 2.23 additional points under the same conservative recipe. More data was not useless, but scale alone was no longer the strongest lever.

## Experiment 2: Optimization and architecture changed the frontier

Raising Nemotron’s learning rate to `1e-4` reduced ATCO2 WER to 32.12%, but LibriSpeech deteriorated to 16.88%. The optimizer had unlocked stronger specialization and stronger forgetting at the same time.

Parakeet CTC 1.1B trained on the same Silver pool reached 26.78% ATCO2, demonstrating that architecture mattered. Its LibriSpeech WER rose to 34.63%, however, making it an ATC specialist rather than a balanced model.

General-English replay addressed that trade-off. Nemotron replay runs held LibriSpeech near 3.39% while reaching approximately 36% ATCO2. Parakeet CTC with equal ATC and English sampling reached 32.81% ATCO2 and 2.15% LibriSpeech.

The strongest balanced Silver parent was Parakeet TDT. It sampled 80% from a 314.716-hour leakage-audited Silver pool and 20% from a 314.721-hour English pool. It reached 30.28% on locked ATCO2 and 2.18% on LibriSpeech. This became the control checkpoint for Gold refinement.

## Experiment 3: Gold worked best as a second-stage correction

We compared four strategies using Parakeet TDT, 2,000 steps, and peak LR `1e-5`.

| Strategy | Starting point and data | ATCO2 WER | LibriSpeech WER |
| --- | --- | ---: | ---: |
| G1 | Untouched model; 0.418 h Gold only | 26.44% | 10.91% |
| G2 | Untouched model; Gold + English replay | 24.30% | 2.67% |
| G3 | Silver-adapted model; Gold + English replay | **20.00%** | 2.31% |
| G4 | Silver-adapted model; Silver + Gold + English | 27.38% | **2.24%** |

G1 showed that even 25 minutes of matching human data could adapt the untouched model, but it also caused severe forgetting. G2 restored general English but left cross-ATC performance weak.

G3 was the key result: broad Silver adaptation first, then a low-learning-rate Gold correction with English replay. It reduced the balanced Silver control from 30.28% to 20.00% while keeping LibriSpeech close to the original 2.18% control.

G4 continued sampling the large Silver pool during Gold refinement and performed much worse. The abundant pseudo-labels diluted the small human-Gold signal. Silver was valuable as an earlier domain-learning stage, not as the dominant source during final correction.

The G3 result also reproduced across seeds, supporting a recipe effect rather than a lucky initialization.

## Experiment 4: The final gains came from decoding

Beam width four reduced the G3 seed-42 result from 20.00% to 19.08% without changing model weights. A paired 5,000-sample bootstrap placed the improvement between 0.68 and 1.15 WER points at 95% confidence; no resample favored greedy decoding.

Checkpoint averaging improved Gold development but did not beat the best individual checkpoint on the locked test. Hard-slice oversampling and generic gain-plus-white-noise augmentation were also safe but weaker than the unmodified G3 recipe. A 200-phrase boosting sweep lost to ordinary beam search.

The more productive decoder experiment was an n-gram language model.

### Why an n-gram model after acoustic fine-tuning?

The acoustic model estimates which token sequences are supported by the audio. An n-gram model contributes a separate estimate of how likely a short token sequence is in the target domain. During shallow fusion, the decoder combines the acoustic score with the LM score. The fusion weight, usually called alpha, controls the strength of this preference.

This is useful for ATC because its phrase patterns are unusually repetitive: callsigns are followed by a limited family of instructions, and words describing headings, altitudes, runways, and clearances appear in predictable local sequences. The LM cannot repair audio that the encoder did not hear, but it can help the decoder choose the domain-plausible sequence among acoustically similar candidates.

It is also a cheap customization rung. The acoustic checkpoint stays frozen, and the LM requires text rather than additional transcribed audio. That made it a natural experiment after Gold refinement had established a strong acoustic model.

### Constructing the LM corpora without test leakage

The first corpus contained only the 393 ATCO2 Gold training transcripts: 4,761 normalized tokens. We also constructed an 80% Gold / 20% general-English corpus by token count to test whether an LM replay mixture would preserve ordinary English.

The normalization matched our WER contract: case folding, Unicode diacritic folding, punctuation and symbol removal, and whitespace collapse. Gold development and community-test transcripts were used only for identifier-level isolation checks; their text never entered LM training. Silver ATCO2 was also excluded.

For the second round, we added the official Jacktol training text. After removing unusable rows, it contributed 6,495 lines and 65,807 tokens. Because the ATCO2 Gold corpus was tiny, we repeated its training lines to create a token-balanced corpus rather than allowing Jacktol to dominate. The result contained 132,461 tokens: 50.32% ATCO2 Gold and 49.68% Jacktol. Jacktol validation and test transcripts remained excluded.

Repetition does not invent new language. It changes the relative weight assigned to the Gold phrase distribution when the n-gram counts are estimated. This was an explicit modeling decision recorded in the corpus report, not hidden oversampling.

### Selecting order and fusion weight

We trained three-gram and four-gram KenLM models and converted them for offline NeMo TDT decoding. The acoustic model was the frozen, top-three-averaged G3 finalist. Ordinary beam-4 decoding scored 18.77% on Gold development. Switching to the MALSD decoder without an LM scored 19.40%, so every LM candidate first had to recover the cost of changing decoder.

The first sweep evaluated 24 combinations of corpus, n-gram order, and alpha. A Gold-only four-gram with alpha `0.1` won at 17.96% development WER, an improvement of 0.81 points over ordinary beam decoding. Before opening the locked test, it had to pass LibriSpeech: WER changed from 2.32% to 2.48%, remaining below the prespecified 2.52% maximum. On the locked ATCO2 test, this Gold-only LM reached **18.12%**, compared with 19.21% for the frozen acoustic finalist.

The Jacktol-text expansion then evaluated 20 additional arms against that Gold-only LM. A balanced four-gram at alpha `0.2` produced the lowest ATCO2 development WER, 17.15%, but raised LibriSpeech to 2.80%. It failed the guardrail and was not authorized for final testing.

We stepped down to alpha `0.1`. On Jacktol validation, it improved the Gold-only LM from 21.15% to 19.49%, while LibriSpeech reached 2.43% and passed the guardrail. Only then did we freeze the configuration and run the final tests.

| Decoder configuration | ATCO2 locked test | Jacktol test | LibriSpeech test-clean |
| --- | ---: | ---: | ---: |
| Frozen G3 acoustic finalist, beam 4 | 19.21% | 20.36% | 2.32% |
| Gold-only four-gram, alpha 0.1 | 18.12% | 20.05% | 2.48% |
| Gold + Jacktol balanced four-gram, alpha 0.1 | **17.61%** | **18.70%** | 2.43% |

The balanced LM removed 1.61 absolute points from the ATCO2 acoustic baseline and 1.66 points from the Jacktol acoustic baseline without another optimizer step. Just as importantly, the result demonstrates why the guardrail belongs inside decoder selection: the development winner was not the model we shipped to final evaluation.

This was an offline NeMo pilot, not yet a production Riva language-model artifact. Deployment would require rebuilding the approved text corpus in the Riva-supported word-level format and then measuring the served decoder again. The orchestration skill treats offline proof and deployable LM construction as separate stages precisely to avoid assuming they are interchangeable.

## External reference: the Jacktol result

We also verified that our setup could reproduce the public Jacktol benchmark. Our selected checkpoint reached **5.93% Jacktol WER**, closely matching the published 5.99% result; the public checkpoint scored 6.06% with our local normalizer. This result provides useful external context, but the main study remained focused on improving ATCO2 under its locked Gold and general-English evaluation contract.

## What did not work—and why it mattered

Negative experiments prevented several plausible but incorrect conclusions:

- post-hoc feature-normalization changes violated the checkpoint contract;
- increasing Silver scale without changing the recipe plateaued;
- high-learning-rate and ATC-only runs hid catastrophic forgetting;
- continuing Silver replay diluted final Gold correction;
- generic noise did not reproduce the remaining radio errors;
- hard-example oversampling reduced overall development accuracy;
- phrase boosting did not help the selected TDT checkpoint;
- the lowest development LM weight was not acceptable when it failed the English guardrail.

Each failure narrowed the next question. That is more valuable than an unexplained improvement from changing several variables at once.

## The practical recipe that emerged

For a domain with abundant weak labels and scarce trusted references, our evidence supports six steps:

1. Freeze leakage-safe Gold development and test sets, then measure the untouched checkpoint on domain and general speech.
2. Use Silver data for broad channel adaptation, with general-domain replay when existing capability must be preserved.
3. Refine the balanced checkpoint on scarce Gold data at a lower learning rate and repeat the recipe across seeds.
4. Select and export checkpoints before opening the locked test.
5. Evaluate beam search, averaging, and n-gram fusion as separate controlled stages.
6. Report domain WER, transfer results, and the forgetting guardrail together.

This is why fine-tuning remains partly an art. The art is not intuition without measurement. It is choosing the next experiment that distinguishes among competing explanations: insufficient domain exposure, noisy labels, weak optimization, architecture mismatch, forgetting, or decoding bias.

The skills make that reasoning durable. The orchestration skill keeps the objective, cheapest sufficient path, and guardrails visible. The NeMo fine-tuning skill preserves the model and evaluation contracts during execution. Together they turn ASR adaptation from a collection of commands into an evidence-driven workflow that another engineer can inspect, reproduce, and extend.
