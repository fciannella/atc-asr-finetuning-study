# Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills

Suppose a speech recognition model transcribes everyday conversation accurately, yet struggles with a pilot reading back a clearance. The model already knows English. What it needs to learn is how English sounds over a radio and how people use it in air traffic control. Fine-tuning lets us teach those differences by updating a pretrained model with examples from the new domain.

The challenge is deciding what to teach, how much to change, and how to check that existing capabilities survive. We explored those questions using air traffic control (ATC) speech and two NVIDIA agent skills. Here we follow the experiments from automatically transcribed audio to human annotations and, finally, an n-gram language model. We measure recognition quality with word error rate (WER), which counts substitutions, deletions, and insertions relative to the reference transcript. Lower is better.

## How the skills guide the work

Think of a skill as a reusable set of instructions for a coding agent. It helps the agent ask the right questions, carry out the work in a sensible order, and collect the evidence needed to judge the result.

The [`nemotron-asr-finetune`](https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune) skill organizes the project. It establishes the target domain, available data, quality objective, hardware, latency, and deployment requirements. After measuring the starting model, it helps choose the least expensive customization likely to address the observed errors.

For a handful of unfamiliar names, word boosting or custom vocabulary may be sufficient. When domain phrases cause trouble, an n-gram language model is worth testing. Differences in channel, accent, or noise may call for acoustic fine-tuning. This progression keeps the work proportional to the problem.

The [`nemo-speech-asr-finetune`](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune) skill handles the NeMo training workflow. It inspects the checkpoint, prepares data with Lhotse, checks transcript conventions, configures training, saves candidate checkpoints, and evaluates the exported model. It also checks that the tokenizer, loss function, and preprocessing remain compatible with the model.

Together, the skills connect experiment planning with execution. Each training run has a reason, a defined dataset, and a test that can tell us whether the change helped.

## Choose the architecture as part of the product decision

Before choosing a recipe, inspect the checkpoint. Its encoder, decoder, tokenizer, feature normalization, and loss function were designed and trained to work together.

| Architecture | Basic idea | Practical implication |
| --- | --- | --- |
| CTC | Predict frame-level tokens, then collapse blanks and repetitions | Parallelizable decoding and convenient external-LM integration |
| RNN-T | Combine an acoustic encoder with a prediction and joint network | Streaming-friendly and conditioned on previously emitted tokens |
| TDT | Extend the transducer family with token-and-duration outputs | Efficient sequence modeling with architecture-specific duration settings |

Our experiments used Nemotron 3.5 ASR Streaming 0.6B, Parakeet CTC 1.1B, and Parakeet TDT 0.6B v3. Each required a compatible training recipe. CTC can be attractive for offline decoding and language-model integration, while a streaming transducer may suit an application that must respond as someone speaks. Choose with the intended application in mind, then compare accuracy.

When adapting within the same language, keeping the pretrained tokenizer is usually the safest starting point. If a replacement is needed, train it only on training text. Input normalization deserves similar care: changing our completed Nemotron checkpoint from global to per-feature normalization without retraining sharply increased WER on both ATC and general speech. The lesson was to preserve the preprocessing the model had learned to expect.

## Why ATC is a demanding adaptation problem

ATC speech concentrates several ASR challenges in one domain:

- narrow-band radio, interference, clipping, and variable gain;
- short, context-dependent transmissions;
- accents and non-native English;
- callsigns, runways, headings, altitudes, and frequencies;
- specialized, compressed phraseology;
- number or callsign errors that are more consequential than ordinary conversational substitutions.

Both the sound and the language differ from ordinary conversation. An adapted model may nevertheless need to recognize general English too, so we measured those two capabilities throughout the study.

## ATCO2 and Jacktol provide different evidence

The ATCO2 delivery contained 3,088,603 recordings and approximately 4,281.9 hours. Filtering for language, duration, confidence, text, and audio quality produced a 314.721-hour English Silver release with 396,461 segments. “Silver” means high-confidence CNET hypotheses rather than human-verified references. The smaller human-Gold pool was assigned to non-overlapping roles:

| Split | Audio | Segments | Role |
| --- | ---: | ---: | --- |
| Gold training | 0.418 h | 393 | Final supervised refinement |
| Gold development | 0.100 h | 100 | Checkpoint and recipe selection |
| Gold community test | 2.000 h | 1,908 | Locked final evaluation |

The splits had zero overlap by airport-date, audio path, record ID, and source recording. Reserving two hours for evaluation left about 25 minutes for Gold training. That was a deliberate trade-off: we wanted enough held-out speech to assess what the model had learned.

The public [Jacktol ATC-ASR dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) contains 7.405 hours, including about 5.9 training hours and official validation and test splits. It supplied an external comparison and training-only domain text. Because it includes material derived from public ATCO2 recordings, cross-corpus claims required an acoustic overlap audit.

## A reproducible fine-tuning workflow

### 1. State the objective

Decide what an acceptable result looks like before training. We wanted lower WER on ATCO2 Gold, with a limited increase on LibriSpeech test-clean. LibriSpeech served as our check for catastrophic forgetting, the loss of previously learned recognition ability during adaptation.

### 2. Inventory and version the data

For every sample, we recorded its source, label quality, split, and original recording. We checked for missing audio and empty transcripts, inspected duration and token distributions, and made transcript conventions consistent. Related segments stayed together when splitting. We saved hashes of the final manifests so another engineer could verify the exact inputs.

General-English replay means continuing to show the model examples of ordinary English during ATC training. Lhotse sampling weights controlled how often each source appeared. We recorded both the available hours and the sampling ratio, since they describe different aspects of the training data.

### 3. Freeze evaluation

Give each split a clear job. Training examples update the weights; development examples help select checkpoints and decoder settings; the held-out test measures the selected configuration. LibriSpeech provides a separate check on general English.

All reported WERs used the same normalizer: lowercase, Unicode diacritic folding, punctuation and symbol removal, and whitespace normalization. In-training validation selected candidates, but final quality came from reloading the exported artifact and running standalone evaluation.

### 4. Train conservatively, then diagnose

We began conservatively and measured the effect of larger learning rates in later experiments. We saved the best validation checkpoints as well as the final checkpoint, because the best model can appear before training ends.

Where possible, each experiment changed one factor: data volume, learning rate, architecture, replay ratio, Gold refinement, checkpoint averaging, decoding, or language-model weight. This made it easier to understand both improvements and setbacks.

### 5. Save enough to reproduce the result

Alongside the `.nemo` file, save the starting checkpoint revision, tokenizer, preprocessing, manifest hashes, training configuration, selection rule, and decoder settings. Include the text normalizer and both domain and general-English scores. These details explain what the model is and how its quality was measured.

## Experiment 1: What did additional Silver data teach the model?

We began with `nemotron-3.5-asr-streaming-0.6b.nemo` and a peak learning rate of `3e-5`. The Silver releases were nested, allowing us to change scale without redefining earlier samples.

| Training data | Steps | ATCO2 Gold WER | LibriSpeech WER |
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

### Why an n-gram model after acoustic fine-tuning?

The ASR model scores candidate transcriptions using the audio and its learned context. An external n-gram language model adds a preference based on short token sequences observed in text. During shallow fusion, the decoder combines the ASR and LM scores. A weight called alpha controls how strongly the external LM influences the choice.

Consider the words following a runway clearance. ATC uses a relatively small set of recurring phrases, so some continuations are much more likely than others. An LM can use those patterns to help choose between acoustically similar candidates. It provides additional evidence, although a strong preference for familiar phrases can also steer the decoder away from what was actually said.

This is an economical experiment because the ASR checkpoint stays fixed and LM training requires only text. It lets us test an additional source of improvement without running another acoustic fine-tuning job.

### Constructing the LM corpora without test leakage

The first corpus contained only the 393 ATCO2 Gold training transcripts: 4,761 normalized tokens. We also constructed an 80% Gold / 20% general-English corpus by token count to test whether an LM replay mixture would preserve ordinary English.

The normalization matched our WER contract: case folding, Unicode diacritic folding, punctuation and symbol removal, and whitespace collapse. Gold development and community-test transcripts were used only for identifier-level isolation checks; their text never entered LM training. Silver ATCO2 was also excluded.

For the second round, we added the official Jacktol training text. After removing unusable rows, it contributed 6,495 lines and 65,807 tokens. Because the ATCO2 Gold corpus was tiny, we repeated its training lines to create a token-balanced corpus rather than allowing Jacktol to dominate. The result contained 132,461 tokens: 50.32% ATCO2 Gold and 49.68% Jacktol. Jacktol validation and test transcripts remained excluded.

Repeating the Gold lines increased their contribution to the n-gram counts; it did not increase the amount of unique text. Recording that distinction matters when describing the corpus and reproducing its balance.

### Selecting order and fusion weight

We trained three-gram and four-gram KenLM models and converted them for offline NeMo TDT decoding. The acoustic model was the frozen, top-three-averaged G3 finalist. Ordinary beam-4 decoding scored 18.77% on Gold development. Switching to the MALSD decoder without an LM scored 19.40%, so every LM candidate first had to recover the cost of changing decoder.

The first sweep evaluated 24 combinations of corpus, n-gram order, and alpha. A Gold-only four-gram with alpha `0.1` scored best at 17.96% development WER, an improvement of 0.81 points over ordinary beam decoding. Before testing that configuration on held-out ATCO2, we checked English: LibriSpeech WER changed from 2.32% to 2.48%, below our predefined 2.52% maximum. The selected Gold-only LM then reached **18.12%** on ATCO2, compared with 19.21% for the frozen acoustic finalist.

With Jacktol text added, we evaluated 20 further configurations against the Gold-only LM. A balanced four-gram at alpha `0.2` produced the lowest ATCO2 development WER, 17.15%, but raised LibriSpeech to 2.80%. That exceeded our accepted regression limit, so we rejected it before final testing.

We stepped down to alpha `0.1`. On Jacktol validation, it improved the Gold-only LM from 21.15% to 19.49%, while LibriSpeech reached 2.43% and passed the guardrail. Only then did we freeze the configuration and run the final tests.

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

Several plausible changes fell short in this study:

- post-hoc feature-normalization changes violated the checkpoint contract;
- increasing Silver scale without changing the recipe plateaued;
- high-learning-rate and ATC-only runs hid catastrophic forgetting;
- continuing Silver replay diluted final Gold correction;
- generic noise did not reproduce the remaining radio errors;
- hard-example oversampling reduced overall development accuracy;
- phrase boosting did not help the selected TDT checkpoint;
- the LM weight giving the lowest development WER exceeded the permitted English regression.

These outcomes helped us decide where to spend the next training or evaluation budget. They also kept the conclusions specific: a technique that failed with this data and checkpoint might still help elsewhere.

## The practical recipe that emerged

For a domain with abundant weak labels and scarce trusted references, our evidence supports six steps:

1. Freeze leakage-safe Gold development and test sets, then measure the untouched checkpoint on domain and general speech.
2. Use Silver data for broad channel adaptation, with general-domain replay when existing capability must be preserved.
3. Refine the balanced checkpoint on scarce Gold data at a lower learning rate and repeat the recipe across seeds.
4. Select and export checkpoints before opening the locked test.
5. Evaluate beam search, averaging, and n-gram fusion as separate controlled stages.
6. Report domain WER, transfer results, and the forgetting guardrail together.

The judgment in fine-tuning lies in choosing what to investigate next. A plateau may reflect noisy labels, an unsuitable learning rate, or a decoder that needs more domain context. The measurements help distinguish these explanations, and the next experiment should make that distinction clearer.

The skills help carry those decisions through the project. The orchestration skill keeps the objective and evaluation criteria in view; the NeMo skill guides the training and scoring needed to test each idea. That leaves the next engineer with a useful starting point: a model, the evidence behind it, and a clear account of what remains to be learned.
