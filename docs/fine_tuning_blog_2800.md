# Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills

Fine-tuning automatic speech recognition (ASR) is simple to describe: begin with a pretrained model, show it transcribed examples from a new domain, and update its weights. Doing it well is more complicated. The architecture, audio conditions, transcript conventions, tokenizer, optimizer, decoding strategy, and evaluation contract all interact. A model can improve dramatically on specialized speech while quietly becoming worse at the speech it already understood.

This article explains how to approach that problem with two reusable NVIDIA agent skills. We use air-traffic-control (ATC) radio as a worked example, following the project from a broad weakly labeled corpus through human-Gold refinement and n-gram language-model fusion. The purpose is not to present one magic configuration. It is to show how an adaptation request becomes a controlled, reproducible sequence of experiments.

## Skills are operating procedures, not model checkpoints

The skills used in this study do not contain hidden models or training services. They encode the questions to ask, the order in which decisions should be made, and the evidence required before calling a model better.

The [`nemotron-asr-finetune`](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) skill is the orchestration layer. It scopes the target domain, available data, quality objective, hardware, latency, and deployment constraints. It establishes a measured baseline and selects the least expensive customization likely to solve the observed errors.

Its customization ladder is deliberate:

1. Use word boosting for a small, known list of terms.
2. Add custom vocabulary or pronunciation support for repeatable lexical failures.
3. Test an n-gram language model when domain text is available but acoustic training data is scarce.
4. Fine-tune the acoustic model when the mismatch includes noise, accents, microphones, or channel conditions.
5. Train from scratch only when no suitable checkpoint exists.

The [`nemo-speech-asr-finetune`](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune) skill is the execution specialist. It selects and inspects the checkpoint, prepares Lhotse data inputs, checks transcript style, preserves architecture and tokenizer contracts, configures training, retains checkpoints, and runs standalone evaluation.

The separation is useful. The orchestration skill asks, “What should change, and how will we know it worked?” The NeMo skill answers, “How do we train and evaluate this model family correctly?”

## Choose the architecture as part of the product decision

An ASR checkpoint is more than a weight file. Its encoder, decoder, tokenizer, feature normalization, and loss function form a learned contract.

| Architecture | Basic idea | Practical implication |
| --- | --- | --- |
| CTC | Predict frame-level tokens, then collapse blanks and repetitions | Parallelizable decoding and convenient external-LM integration |
| RNN-T | Combine an acoustic encoder with a prediction and joint network | Streaming-friendly and conditioned on previously emitted tokens |
| TDT | Extend the transducer family with token-and-duration outputs | Efficient sequence modeling with architecture-specific duration settings |
| AED / Canary | Use an attention-based encoder-decoder, often with task prompts | Flexible multilingual and multitask behavior with different metadata requirements |

Our experiments used Nemotron 3.5 ASR Streaming 0.6B, Parakeet CTC 1.1B, and Parakeet TDT 0.6B v3. We did not treat their recipes as interchangeable. A CTC model can be attractive for offline decoding and language-model integration; a transducer can be the better serving choice for low-latency streaming. The intended product matters as much as offline WER.

The tokenizer and input normalization are also part of this contract. For same-language domain adaptation, preserving the pretrained tokenizer is normally the safest starting point. A tokenizer should be replaced only when the target introduces a script, language, or symbol inventory that it cannot represent. Development and test transcripts must never participate in tokenizer training.

Similarly, feature normalization is not a cosmetic configuration field. In one diagnostic experiment, changing a completed Nemotron checkpoint from global to per-feature normalization without retraining raised both ATC and general WER dramatically. Per-feature normalization is not inherently wrong—Parakeet TDT uses it natively—but changing preprocessing after training invalidates what the checkpoint learned.

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

The ATCO2 delivery contained 3,088,603 recordings and approximately 4,281.9 hours. Most of that material was not equally suitable for supervised training. After filtering for language, duration, confidence, text, and audio quality, we created a 314.721-hour English Silver release with 396,461 segments.

“Silver” is important. These transcripts were high-confidence CNET top hypotheses rather than human-verified references. ATCO2 also contained a much smaller human-Gold pool. We assigned that Gold data to non-overlapping roles:

| Split | Audio | Segments | Role |
| --- | ---: | ---: | --- |
| Gold training | 0.418 h | 393 | Final supervised refinement |
| Gold development | 0.100 h | 100 | Checkpoint and recipe selection |
| Gold community test | 2.000 h | 1,908 | Locked final evaluation |

The splits had zero overlap by airport-date, audio path, record ID, and source-recording ID. Reserving two hours for final evaluation left only about 25 minutes of Gold training audio, but protected the credibility of the result.

The public [Jacktol ATC-ASR dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) played a different role. It contains 7.405 hours in total, including about 5.9 training hours plus official validation and test splits. It served as a compact supervised source, an external comparison, and a source of training-only domain text.

The datasets are complementary, not interchangeable. ATCO2 offers a broad multi-airport Silver pool, auditable human acceptance, and airport and channel metadata. Jacktol offers a compact public benchmark with established splits. It also contains material derived from public ATCO2 recordings. We therefore audited transcript candidates acoustically and reported cross-corpus results with an overlap caveat or on a Jacktol-disjoint subset.

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

An n-gram language model was more effective. We built three- and four-gram KenLM candidates only from allowed training text, selected order and fusion weight on development, checked LibriSpeech, and then opened the locked test once. A balanced four-gram using ATCO2 Gold training text and Jacktol training text reduced the development-selected acoustic model from 19.21% to **17.61% ATCO2 WER**. Jacktol test reached 18.70%, and LibriSpeech remained close to baseline at 2.43%.

A larger fusion weight looked better on ATCO2 development but failed the general-English guardrail. The selected weight was therefore not the development-only minimum.

## Experiment 5: Reproducing Jacktol required a curriculum

The public Jacktol checkpoint reported 5.99% WER and scored 6.06% with our local normalizer. Our first flat 20,000-step emulation failed at 55.86%, despite using the expected model family and matched Jacktol data.

The successful approach was staged. Two broad stages mixed Jacktol, UWB ATC, and LibriSpeech; three shorter polishing stages progressively emphasized Jacktol while retaining English replay. The selected P2 checkpoint reached **5.93% Jacktol WER**, closely reproducing the published value. P3 reached 5.91%, only two fewer errors out of 8,510 reference words, but was slightly worse on UWB and LibriSpeech. P2 was the more defensible balanced choice.

The experiment explained why copying a visible final configuration had failed. The large accuracy gain came from the sequence of broad domain learning and targeted polishing. A model name, a dataset list, and one learning rate were not a complete recipe.

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

For a domain with abundant weak labels and scarce trusted references, our evidence supports this sequence:

1. Freeze a leakage-safe Gold development and test contract.
2. Evaluate the untouched checkpoint on domain and general speech.
3. Use Silver data for broad domain and channel adaptation.
4. Include general-domain replay when the model must preserve general speech.
5. Refine the balanced checkpoint on scarce Gold data at a lower learning rate.
6. Repeat the winning recipe across seeds.
7. Select and export checkpoints before opening the locked test.
8. Evaluate beam search, averaging, and n-gram fusion as separate stages.
9. Report domain WER, transfer results, and the forgetting guardrail together.

This is why fine-tuning remains partly an art. The art is not intuition without measurement. It is choosing the next experiment that distinguishes among competing explanations: insufficient domain exposure, noisy labels, weak optimization, architecture mismatch, forgetting, or decoding bias.

The skills make that reasoning durable. The orchestration skill keeps the objective, cheapest sufficient path, and guardrails visible. The NeMo fine-tuning skill preserves the model and evaluation contracts during execution. Together they turn ASR adaptation from a collection of commands into an evidence-driven workflow that another engineer can inspect, reproduce, and extend.
