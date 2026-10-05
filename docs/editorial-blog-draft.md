# 

| Article overview | Working title: Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills Format: Hands-on developer tutorial Draft deadline: October 2, 2026 (requested by Michelle Horton) Length: Under 1,400 words Target publication: October 13, 2026, alongside the livestream, pending editorial confirmation |
| :---- | :---- |
| **Submitter’s notes** | **Author / submitter:** Francesco Ciannella **Jira:** [TECHBLOG-5795](https://nvidia.atlassian.net/browse/TECHBLOG-5795) **Scope:** Use the Nemotron ASR fine-tuning skills with a coding agent to adapt a pretrained ASR model to domain-specific audio, using air traffic control (ATC) as the worked example. Cover data preparation, NeMo training, and baseline versus fine-tuned evaluation. **Community contribution:** NVIDIA plans to release a two-hour, human-annotated ATC dataset alongside the blog and livestream, giving developers and researchers data and guidance to build on. **Release links:** Dataset, skills, example code, and staging/publication links pending. **Results:** Exact model, dataset splits, training setup, and WER results to be verified against experiment artifacts. |
| **Editorial and technical review** | **Pitch reviewer:** Michelle Horton (confirmed by Elizabeth Goodman) **Editorial contact:** Elizabeth Goodman (cw) **Proposed reviewers / contacts:** Maryam Motamedi, Adi Margolin, Jayda Ritchie — specific responsibilities to be confirmed. **Additional input requested by Michelle:** Rosie Brown and Chris Alexiuk **Review status:** Michelle requested a draft under 1,400 words by October 2\. Publication timing remains subject to review. |
| **Campaign / livestream coordination** | **Livestream:** October 13, 2026 **Livestream details:** Maryam Motamedi and Rebecca Kao **PMM lead and campaign reviewers:** To be confirmed |
| **Awareness / outstanding details** | Confirm dataset documentation, license, download location, and how the released two-hour dataset relates to the experimental training, validation, and test splits. Verify measured results and reproducible example links before publication. |
| **Social copy — draft for review** | **LinkedIn:** Adapt speech recognition to your domain with NVIDIA Nemotron Speech skills. This hands-on ATC example walks through preparing data, fine-tuning with NeMo, and evaluating recognition quality. Alongside the blog, NVIDIA plans to release two hours of human-annotated ATC data to help the community experiment and build on the work. Join the October 13 livestream for more. \[Blog, dataset, and livestream links pending\] **X:** Fine-tune ASR for your domain with NVIDIA Nemotron Speech skills. Explore the ATC workflow and a planned community release of two hours of human-annotated ATC data. Join the October 13 livestream. \[Links pending\] |

# **Fine-Tune ASR for Your Domain with NVIDIA Nemotron Speech Skills**

## **A skill-guided case study in air-traffic-control speech**

Fine-tuning an automatic speech recognition model is easy to describe: take a pretrained model, show it transcribed examples from a new domain, and update its weights. Doing it well is more complicated. The model architecture, audio conditions, transcript conventions, data split, tokenizer, optimizer, decoding strategy, and success criteria all interact. A model can improve dramatically on a domain test set while quietly becoming much worse at ordinary speech.

This article explains that complete process. We use air-traffic-control (ATC) radio as the running example and two reusable NVIDIA fine-tuning skills as the guide rails. The goal is not to present one magic configuration. It is to show how to turn an adaptation request into a controlled, reproducible experiment, and how to reason about the trade-offs that appear along the way.

The experimental results will follow in the next section of the article. First, we need to establish the models, data, workflow, and evaluation contract that make those results meaningful.

# **What the fine-tuning skills are**

The skills are not model checkpoints, training frameworks, or hidden services. They are reusable operating procedures for an AI coding agent. They capture the questions to ask, the order in which decisions should be made, the checks that should happen before expensive training, and the evidence required before a model can be called better.

We used two complementary skills:

* [nemotron-asr-finetune](https://github.com/nvidia-riva/Nemotron-speech-skills/tree/main/skills/nemotron-asr-finetune) is the orchestration layer. It scopes the adaptation objective, selects the least expensive technique likely to work, routes the task to the appropriate specialist, and keeps domain accuracy, general accuracy, latency, and deployment constraints visible throughout the project.  
* [nemo-speech-asr-finetune](https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune) is the training specialist. It handles the concrete NeMo workflow: identifying the checkpoint family, preparing Lhotse-compatible data, preserving architecture and tokenizer contracts, constructing the training recipe, retaining useful checkpoints, and running standalone evaluation.

Locally, these skills are installed under \~/.agents/skills/nemotron-asr-finetune/ and \~/.agents/skills/nemo-speech-asr-finetune/. Separating orchestration from execution is intentional. One skill asks, “What should we change, and how will we know it worked?” The other answers, “How do we train and evaluate this particular NeMo ASR model correctly?”

### **Fine-tuning is not always the first rung**

The orchestration skill treats ASR customization as a ladder:

1. Word boosting for a small list of names or terms.  
2. Custom vocabulary and pronunciation support.  
3. An external n-gram language model for broader phrase and spelling bias.  
4. Acoustic-model fine-tuning for a true domain or acoustic mismatch.  
5. Training from scratch only when the earlier rungs are insufficient.

These methods can also be combined. A fine-tuned acoustic model may still benefit from an n-gram language model. The important lesson is that a request to “fine-tune the model” should first be translated into a measurable failure mode. If the problem is only ten unusual proper nouns, retraining hundreds of millions of parameters may be unnecessary.

# **The model families we considered**

An ASR checkpoint is more than a collection of weights. Its encoder, decoder, tokenizer, feature normalization, and loss function form a contract. A recipe written for one family should not be applied blindly to another.

| Architecture | How it predicts text | Practical character | Important fine-tuning constraint |
| :---- | :---- | :---- | :---- |
| CTC | Produces frame-level token probabilities and collapses blanks and repetitions | Simple, parallelizable decoding; can work well with external language models | Preserve the CTC head and tokenizer; decoding settings can materially affect final WER |
| RNN-T | Combines an acoustic encoder, a prediction network, and a joint network | Streaming-friendly and conditioned on previous emitted tokens | Preserve the transducer loss and decoder/joint configuration |
| TDT | Extends the transducer family by predicting tokens and their durations | Efficient sequence modeling with an explicit duration vocabulary | Keep the TDT loss, duration bins, and model type consistent |
| Hybrid transducer/CTC | Shares an encoder between a transducer head and a CTC head | Supports multiple decoding paths and useful auxiliary training | Evaluate and report which head produced each score |
| AED / Canary | Uses an encoder-decoder attention model, often with task or language prompts | Flexible multilingual and multitask behavior | Preserve prompt fields, language/task metadata, and decoder conventions |

Our study used three representative pretrained models:

* **Nemotron 3.5 ASR Streaming 0.6B**, a streaming transducer-family model.  
* **Parakeet CTC 1.1B**, a larger CTC model well suited to testing architecture and capacity effects.  
* **Parakeet TDT 0.6B v3**, a token-and-duration transducer model with its own native feature-processing contract.

There is no universally best architecture. A CTC model may offer convenient offline decoding and language-model integration; a transducer may be the better product choice for low-latency streaming. Model selection must therefore include the intended serving mode, latency budget, memory budget, and deployment target, not only the lowest offline WER.

### **The tokenizer is part of the model**

For adaptation within the same language and writing system, preserving the pretrained tokenizer is usually the safest starting point. Replacing it changes the output space and can discard useful knowledge encoded in the decoder.

A new tokenizer becomes justified when the target introduces a new script, language, or symbol inventory that the original tokenizer cannot represent adequately. Even then, it must be trained only on training text. Development and test transcripts must never influence the vocabulary.

The same principle applies to audio normalization. A checkpoint trained with global feature normalization cannot safely be converted to per-feature normalization by changing one configuration field after training. Preprocessing is learned context, not a cosmetic switch.

# **Why air-traffic-control speech is a useful case study**

ATC radio concentrates several hard ASR problems into one domain:

* narrow-band radio, background noise, interference, clipping, and variable gain;  
* short, context-dependent transmissions from pilots and controllers;  
* accents and non-native English;  
* callsigns, runway identifiers, headings, altitudes, frequencies, and spelled letters;  
* a constrained but highly specialized phraseology;  
* errors in numbers or callsigns that are more consequential than ordinary conversational substitutions.

This makes ATC a good teaching example. The acoustic channel differs from the speech on which a general model is typically trained, and the language distribution differs as well. At the same time, we may want the adapted model to retain ordinary English recognition. The experiment therefore needs two objectives: improve ATC accuracy and limit general-domain regression.

# **The ATCO2 and Jacktol data play different roles**

Our primary source was the ATCO2 delivery. It is a broad, multi-airport corpus rather than a single clean benchmark. The complete delivery contains 3,088,603 recordings and approximately 4,281.9 hours of audio. Most of that material is not equally suitable for supervised training.

After language, duration, confidence, text, and audio-quality filtering, we created a high-quality English **Silver** training release containing 396,461 segments and 314.721 hours. “Silver” is important: these are selected CNET top-hypothesis transcripts, not human-verified references. The corpus also contains a much smaller **Gold** pool whose transcripts were accepted by human annotators. Gold data is scarce and is therefore reserved carefully among training, development, and the locked community test.

The publicly available [jacktol/ATC-ASR-Dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) serves a different purpose. Its official training split is compact (about 5.9 hours) and it provides published train, validation, and test partitions. It is useful as an external ATC benchmark, a small supervised source, and a source of in-domain training text. We pin a specific dataset revision whenever it is used so that the experiment remains reproducible.

| Property | ATCO2 | Jacktol ATC-ASR |
| :---- | :---- | :---- |
| Availability | Project/vendor delivery | Public Hugging Face dataset |
| Scale | Thousands of raw hours; 314.721-hour filtered Silver release plus scarce Gold data | Compact; about 5.9 hours in the official training split |
| Coverage | Multiple airports with an uneven geographic distribution | A separate, narrower released corpus |
| Primary value | Large-scale domain adaptation and controlled Gold evaluation | External transfer test, compact supervised source, and domain text |
| Main caution | Confidence-filtered pseudo-labels are not equivalent to human Gold | Official validation and test data must remain outside training and language-model construction |

The two datasets are complementary, not interchangeable. Mixing them without provenance labels would hide what the experiment is actually measuring. We retain the source, label class, airport, split, and dataset revision for every sample. We also audit acoustic overlap before calling a result “external.”

# **The end-to-end fine-tuning workflow**

The skills turn the project into a sequence of explicit decisions.

### **1\. State the objective**

Start with the product outcome, not the training command. A useful objective might be:

*Reduce normalized WER on the locked ATCO2 Gold test set while keeping LibriSpeech test-clean WER within an agreed regression budget, using a model that can meet the target serving constraints.*

This statement identifies the domain metric, general-domain guardrail, evaluation split, and operational constraint. Without it, “better” can change meaning after the results arrive.

### **2\. Inventory the evidence**

Before training, determine:

* how many hours have reliable transcripts;  
* which labels are Gold, Silver, or synthetic;  
* whether source recordings cross split boundaries;  
* what domain-only text is available;  
* which development and test sets are genuinely independent;  
* whether the intended model is streaming or offline;  
* the acceptable latency, memory, and general-domain regression.

This stage often changes the experiment more than any optimizer setting. One hour of trusted Gold data can be more informative than many hours of noisy pseudo-labels, but using that Gold data for training means it can no longer provide an unbiased test.

### **3\. Freeze the evaluation contract**

We use three separate evaluation roles:

| Split | Purpose | May influence training? |
| :---- | :---- | :---- |
| Training | Updates model parameters | Yes |
| Development | Selects checkpoints and tunes decoding or LM weights | Indirectly, yes |
| Locked test | Produces the final ATC claim | No |
| General-domain guardrail | Measures catastrophic forgetting | No |

WER normalization must also be frozen. Our default comparison lowercases text, removes punctuation and non-lexical symbols, folds Unicode diacritics, and normalizes whitespace before scoring. The same normalizer must be applied to references and hypotheses.

Training-time validation WER is useful for selecting candidate checkpoints, but it is not the final number. The selected saved model artifact is reloaded and scored in a standalone evaluation. That verifies that the checkpoint can actually be exported and reproduced.

### **4\. Prepare the data as a versioned artifact**

Audio paths, durations, transcripts, sources, and split assignments are expressed as validated manifests. Lhotse provides a convenient representation for NeMo training, filtering, and weighted multi-source sampling.

Data preparation includes more than converting file formats:

* reject missing, unreadable, or zero-duration audio;  
* normalize transcript conventions consistently;  
* inspect duration and token distributions;  
* preserve callsigns and numbers according to the scoring policy;  
* group related segments by source recording before splitting;  
* record every exclusion and its reason;  
* fingerprint the final manifests so the exact population can be reconstructed.

When general-English replay is used, source weights determine how often ATC and general speech are sampled. This is preferable to physically duplicating audio, and it makes a ratio such as 80% ATC / 20% general English an explicit part of the recipe.

### **5\. Adapt the pretrained checkpoint**

The specialist skill first confirms the model family and its configuration. It then makes only deliberate changes: training manifests, batch strategy, learning rate, schedule, maximum steps, validation cadence, checkpoint retention, and logging.

For same-language domain adaptation, a conservative recipe normally starts with the original tokenizer, mixed-precision training, and a lower learning rate than pretraining. We use BF16 where supported and avoid automatic mixed-precision policies that silently change precision behavior.

Checkpointing is part of the experiment design. Keeping only the final step assumes that the best model occurs at the end; that is often false. We retain the strongest validation checkpoints, the final checkpoint, and enough metadata to reproduce the selection rule. Checkpoint averaging is evaluated as a candidate technique rather than assumed to help.

### **6\. Measure both specialization and forgetting**

Every serious run answers at least two questions:

1. Did ATC recognition improve?  
2. What capability did the model lose to obtain that improvement?

A domain-only model can achieve a striking ATC score while suffering catastrophic forgetting on ordinary English. Conversely, a heavily replay-balanced model may preserve general speech but leave useful ATC accuracy on the table. Neither is inherently wrong; they are different operating points. The product objective decides which one is acceptable.

### **7\. Diagnose before launching the next run**

The next experiment should change one interpretable factor whenever possible:

* data scale or label quality;  
* ATC/general replay ratio;  
* architecture;  
* peak learning rate or schedule;  
* number of updates;  
* checkpoint selection or averaging;  
* decoding beam or n-gram LM weight;  
* acoustic augmentation.

Changing all of them at once may produce a better number, but it teaches us almost nothing about why.

### **8\. Export the complete model contract**

A useful output is not just a .nemo file. It includes the starting checkpoint and revision, tokenizer, feature processing, manifests and their hashes, training configuration, logs, checkpoint-selection rule, decoder settings, text normalizer, domain and guardrail scores, and the intended deployment profile.

That package is what allows another engineer to reproduce the result or safely continue the work.

# **Why fine-tuning is partly an art**

Fine-tuning is scientific when the data, hypotheses, and evaluation are controlled. It feels like an art because several valid objectives compete and the best settings are not independent.

A larger learning rate may unlock rapid domain learning but accelerate forgetting. More Silver data may add acoustic variety while also reinforcing transcript errors. Gold data may be the most valuable training material, yet spending it on training reduces the evidence available for unbiased evaluation. A language model may fix domain phraseology without improving acoustic confusions. A checkpoint with the best development WER may fail on the locked test or violate the general-English guardrail.

The craft lies in choosing the next experiment that separates these explanations. The skills help by making that reasoning explicit: start with the least expensive credible intervention, preserve the model’s architectural contract, freeze the evaluation before optimization, keep every dataset role visible, and promote a model only when both its gain and its cost have been measured.

# **Where the results begin**

With this foundation in place, the results section can be read as a sequence of answered questions rather than a leaderboard: How much did the first ten hours teach us? Did more Silver data continue to help? What changed when we increased the learning rate or switched architecture? Could general-English replay prevent forgetting? How much value came from scarce Gold data, decoding, and an external n-gram language model?

# **What We Learned Fine-Tuning ASR for Air-Traffic Control**

## **Experiments and results from Silver adaptation to Gold refinement**

This is the results companion to \[How to Fine-Tune ASR for a New Domain\](skill-guided-asr-finetuning.md). The first article explains the model families, fine-tuning skills, data roles, and evaluation workflow. This article follows the experiments in the order in which they answered our questions.

The study was not a single training run. It progressed through five phases:

1. Measure how Nemotron ASR responds to increasing amounts of Silver ATCO2 data.  
2. Test optimization, model architecture, and general-English replay.  
3. Introduce the small, isolated human-Gold ATCO2 training split.  
4. Reproduce the public Jacktol result and measure cross-corpus transfer.  
5. Improve the frozen acoustic model through decoding and n-gram language models.

The main lesson is that the data role and training sequence mattered as much as the number of hours.

## **How to read the numbers**

All WER values in this article are standalone, normalized word error rates. Lower is better. References and hypotheses are lowercased, Unicode diacritics are folded, punctuation and symbols are removed, and whitespace is normalized before scoring.

Four evaluation sets appear repeatedly:

| Evaluation set | Role |
| :---- | :---- |
| ATCO2 locked community test | Primary target-domain test: 2.000 hours, 1,908 human-Gold segments |
| Jacktol test | Public ATC comparison: 0.762 hours, 813 segments |
| UWB ATC test | Independent University of West Bohemia ATC transfer guardrail: 2.434 hours, 2,822 segments |
| LibriSpeech test-clean | General-English forgetting guardrail: 5.403 hours, 2,620 segments |

Development data selected checkpoints, decoder settings, and language-model weights. The locked ATCO2 test was not used for those decisions. Some early experiments originally used a historical ATC development set; we later re-evaluated all surviving Silver checkpoints on the same locked ATCO2 Gold test so that the retrospective comparison below has one consistent target metric.

## **Phase 1: How much Silver ATCO2 data was useful?**

We started from nemotron-3.5-asr-streaming-0.6b.nemo. The Silver releases were nested: the 50-hour release contained the 10-hour release, the 100-hour release contained the 50-hour release, and the Full HQ release contained 314.721 hours and 396,461 confidence-filtered CNET transcripts.

The initial recipe used a peak learning rate of 3e-5.

| Training release | Steps | ATCO2 Gold test WER | LibriSpeech WER |
| :---- | :---- | :---- | :---- |
| Untouched Nemotron | 0 | 75.57% | 3.52% |
| 10 h Silver | 9,791 | 44.02% | 5.46% |
| 50 h Silver | 24,001 | 42.71% | 5.77% |
| 100 h Silver | 10,000 | 44.77% | 5.41% |
| 100 h Silver | 20,000 | 43.63% | 5.74% |
| 314.7 h Full HQ Silver | 62,944 | 41.79% | 6.09% |

The first ten hours produced the decisive jump: 31.55 absolute WER points on the current locked test. Adding another 304.7 hours improved the result by only 2.23 more points under the same conservative recipe.

The 100-hour comparison also showed why “hours” cannot be interpreted without the optimization budget. At 10,000 steps, 100 hours underperformed the 50-hour run. Doubling the horizon recovered most of the regression, but still did not beat the Full HQ result.

**Finding:** the model learned the vocabulary, phrase structure, and radio domain quickly. After that first adaptation, label quality, optimization, source balance, and model architecture were stronger levers than raw Silver volume.

## **Phase 2: Optimization, architecture, and forgetting**

### **Learning rate unlocked specialization—with a cost**

The Nemotron Full HQ learning-rate screen changed the result more than the scale study. Raising the peak learning rate from 3e-5 to 1e-4 and selecting the 14,000-step checkpoint reduced locked ATCO2 WER from 41.79% to 32.12%.

However, LibriSpeech WER rose from the untouched model’s 3.52% to 16.88%. The same setting that made the optimizer move decisively into the ATC domain also caused severe general-domain forgetting.

The intermediate 6e-5 arm completed and reached 35.04% best in-training validation WER near 19,000 steps. It was not promoted to standalone scoring after losing the selection screen, so it should not be presented as a final ATCO2 or LibriSpeech result.

### **Feature normalization was not a post-hoc switch**

We cloned the completed Nemotron checkpoint and changed global feature normalization to per-feature normalization without retraining. ATC WER rose to 84.41% and LibriSpeech WER to 89.00%.

This does not show that per-feature normalization is inherently bad—Parakeet TDT uses it natively. It shows that preprocessing is part of the checkpoint’s learned contract. Changing it after training invalidates the model.

### **Architecture materially changed the frontier**

We next trained Parakeet CTC 1.1B on the same 314.7-hour Silver release.

| Model | Training | ATCO2 Gold test WER | LibriSpeech WER |
| :---- | :---- | :---- | :---- |
| Parakeet CTC 1.1B base | None | 63.51% | 2.04% |
| Parakeet CTC 1.1B | 314.7 h Silver | 26.78% | 34.63% |

The ATC-only Parakeet model was far stronger than the initial Nemotron scale runs, but it was an ATC specialist rather than a balanced ASR model. Its general-English regression was catastrophic.

### **General-English replay controlled forgetting**

We paired the 314.721-hour Silver ATCO2 pool with a deterministic 314.721-hour LibriSpeech pool. Sampling weights controlled how frequently each source appeared; they did not imply that all source audio was consumed equally in every epoch.

| Experiment | Source pools and sampling | ATCO2 Gold test WER | LibriSpeech WER |
| :---- | :---- | :---- | :---- |
| Nemotron replay 1:1 | 314.721 h ATCO2 \+ 314.721 h English; 50% / 50% | 36.55% | 3.39% |
| Nemotron replay 2:1 | Same pools; 66.7% / 33.3% | 35.96% | 3.39% |
| Parakeet CTC replay 1:1 | Same pools; 50% / 50% | 32.81% | 2.15% |
| Parakeet TDT Silver control | 314.716 h audited ATCO2 \+ 314.721 h English; 80% / 20% | 30.28% | 2.18% |

Replay recovered almost all general-English quality while retaining substantial ATC improvement. The Parakeet TDT run became the Silver control and the parent checkpoint for the Gold experiments.

**Finding:** there was no useful single-axis leaderboard. The ATC-only CTC model had the best Silver ATC number, while the TDT control was a much better starting point for continued work because it preserved English and supported clean Gold refinement.

## **Phase 3: What did 25 minutes of human-Gold training add?**

We isolated the human-Gold ATCO2 data into non-overlapping roles:

* Training: 0.418 hours, 393 segments, 19 airport-day groups.  
* Development: 0.100 hours, 100 segments.  
* Locked community test: 2.000 hours, 1,908 segments.

Train, development, and test had zero overlap by airport-date, audio path, record ID, and source-record ID. Only the 0.418-hour training split supplied Gold supervision.

We then ran a controlled G1–G4 ablation with Parakeet TDT 0.6B v3. Each Gold refinement used 2,000 steps, peak LR 1e-5, and the same scoring contract.

| Run | Starting checkpoint | Training composition | ATCO2 WER | Jacktol WER | UWB WER | LibriSpeech WER |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| G1, seed 1234 | Untouched Parakeet TDT | 0.418 h Gold only | 26.44% | 32.56% | 43.29% | 10.91% |
| G2, seed 1234 | Untouched Parakeet TDT | 0.418 h Gold \+ 314.721 h English pools; 85% / 15% | 24.30% | 30.40% | 41.38% | 2.67% |
| G3, seed 1234 | Silver-adapted TDT control | Same Gold \+ English pools; 85% / 15% | 20.51% | 21.56% | 32.51% | 2.41% |
| G3, seed 42 | Silver-adapted TDT control | Same Gold \+ English pools; 85% / 15% | **20.00%** | **21.36%** | **32.23%** | 2.31% |
| G4, seed 1234 | Silver-adapted TDT control | 314.716 h Silver \+ 0.418 h Gold \+ 314.721 h English; 70% / 15% / 15% | 27.45% | 23.17% | 34.00% | 2.38% |
| G4, seed 42 | Silver-adapted TDT control | Same three pools; 70% / 15% / 15% | 27.38% | 23.08% | 33.73% | **2.24%** |

G1 proved that even 25 minutes of matching Gold data could strongly adapt the base model. It also showed why a domain number cannot stand alone: LibriSpeech degraded to 10.91%.

G2 added English replay and recovered most general quality, but its cross-ATC results remained weak. G3 was the important result. Broad Silver adaptation first, followed by a small low-LR Gold correction with English replay, reduced the Silver control from 30.28% to 20.00% on the locked test while holding LibriSpeech near the 2.18% control.

G4 continued sampling Silver during the Gold correction and performed much worse. The abundant pseudo-labels diluted the scarce human signal. The model benefited from Silver as an earlier domain-learning stage, not as a dominant source during the final correction.

**Finding:** Gold was most valuable as a precise second-stage correction after broad domain adaptation.

## **Phase 4: Refining the G3 model**

### **Beam search provided a statistically supported gain**

The G3 seed-42 acoustic model scored 20.00% with greedy decoding. Beam width 4 reduced the locked ATCO2 result to **19.08%** and left LibriSpeech effectively unchanged at 2.28%.

A paired 5,000-sample bootstrap estimated the improvement at 0.92 WER points, with a 95% interval from 0.68 to 1.15 points. No resample favored the greedy control. This was a real decoder gain, not ordinary measurement noise.

### **Checkpoint averaging helped development, but not the locked test**

The top three G3 seed-42 validation checkpoints were averaged before the test was opened. The averaged artifact scored 18.77% on Gold development and became the development-selected finalist. Its final standalone results were:

| Evaluation | WER |
| :---- | :---- |
| ATCO2 locked community test | 19.21% |
| Jacktol test | 20.36% |
| UWB test | 31.23% |
| LibriSpeech test-clean | 2.32% |

The average generalized well, but it finished 0.13 points behind the single seed-42 beam-4 checkpoint on ATCO2. Averaging was useful, not automatically superior.

### **The recipe reproduced across seeds**

Additional G3 runs scored 19.58% for seed 17 and 19.04% for seed 73 on Gold development with beam-4 decoding. Together with seeds 42 and 1234, these runs supported a recipe effect rather than a lucky initialization. The extra seeds were development confirmations; they were not used to repeatedly probe the locked test.

### **Plausible refinements did not all help**

| Refinement | Gold development WER | Outcome |
| :---- | :---- | :---- |
| Top-three G3 average, beam 4 | **18.77%** | Development-selected control |
| Seed 17 reproduction | 19.58% | Reproduced the recipe, but did not win selection |
| Seed 73 reproduction | 19.04% | Close to the frontier, but did not beat averaging |
| Hard-slice reweighting | 20.40% | Over-emphasizing difficult Gold examples hurt overall accuracy |
| Mild gain \+ white-noise proxy | 19.68% | Safe, but generic noise did not reproduce the remaining radio errors |
| Phrase boosting | 19.40% or worse | MALSD and positive boost weights lost to ordinary beam 4 |

The negative results were informative. Difficulty sampling, generic noise, and a list of 200 training-derived phrases were not the missing final-mile lever.

## **Phase 5: Reproducing the public Jacktol result**

The public qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC checkpoint reports 5.99% Jacktol WER. Using our local normalized scorer, the unchanged checkpoint reached 6.06%, which closely reproduced the published claim.

Our first flat emulation—20,000 steps on Jacktol plus English—failed at 55.86% Jacktol WER. The failure revealed that matching a model name, dataset, and learning rate was not enough. The successful recipe was a staged curriculum using Jacktol, UWB, and English before target-specific polishing.

| Stage | Training mixture | Steps / LR | Jacktol test | UWB test | LibriSpeech |
| :---- | :---- | :---- | :---- | :---- | :---- |
| A1 | 5.896 h Jacktol \+ 10.534 h UWB \+ 100.344 h English; 30% / 30% / 40% | 14,465 / 3e-5 | 9.05% | 16.57% | 3.98% |
| A2 | Same pools; 40% / 35% / 25% | 23,144 / 2e-5 | 6.56% | **12.76%** | **4.11%** |
| P1 | Jacktol \+ English; 80% / 20% | 5,000 / 1e-5 | 6.43% | 13.24% | 4.20% |
| P2 | Jacktol \+ English; 85% / 15% | 4,000 / 2e-5 | **5.93%** | 13.54% | 4.07% |
| P3 | Jacktol \+ English; 90% / 10% | 3,000 / 1.5e-5 | **5.91%** | 13.66% | 4.10% |

P3 had the lowest Jacktol score, but it corrected only two more words than P2 out of 8,510 and was worse on both transfer guardrails. P2 was therefore the selected balanced endpoint.

This reproduced the public result rather than merely approaching it: 5.93% for selected P2 and 5.91% for P3, compared with 5.99% reported publicly and 6.06% measured locally for the public checkpoint.

### **Cross-corpus evaluation explained an apparent mismatch**

The public checkpoint scored 15.91% on the full 2.000-hour ATCO2 test. We found confirmed source-recording overlap between Jacktol and part of ATCO2. After removing 213 affected segments, it scored 16.65% on the remaining 1.749 hours. The disjoint score is the cleaner transfer estimate, but it is evaluated on a different subset and must not be placed directly beside full-test numbers without qualification.

Our P2 checkpoint scored 17.79% on the full ATCO2 test. Adding 0.678 hours of leakage-safe ATCO2 Gold to P2’s training mixture improved ATCO2 to **15.44%**, left Jacktol exactly at 5.93%, and changed UWB and LibriSpeech only slightly. A later, more aggressive repair stage regressed Jacktol and UWB and was rejected.

**Finding:** the very low Jacktol result came from matched human labels and a carefully staged curriculum, not from a mysterious model advantage. Cross-dataset evaluation and overlap auditing were necessary to interpret it correctly.

## **Phase 6: N-gram language-model fusion**

The last phase kept the development-selected G3 acoustic checkpoint frozen and changed only decoding. We trained compact KenLM n-gram models from training text; no development or test transcript entered an LM corpus.

The first LM family used the isolated ATCO2 Gold training transcripts. The second added the official Jacktol training text and balanced the two domain-text sources. Three- and four-gram variants and fusion weights were selected on development data before one frozen locked-test pass.

| Decoder | LM training text | ATCO2 locked WER | Jacktol test WER | LibriSpeech WER |
| :---- | :---- | :---- | :---- | :---- |
| G3 averaged acoustic model, beam 4 | None | 19.21% | 20.36% | 2.32% |
| Gold-domain 4-gram | ATCO2 Gold train only | 18.12% | — | Guardrail passed |
| Balanced-domain 4-gram, alpha 0.1 | ATCO2 Gold train \+ Jacktol train, 50.32% / 49.68% text balance | **17.61%** | **18.70%** | **2.43%** |

The Gold-only LM removed another 1.09 absolute ATCO2 points without retraining the acoustic model. Adding Jacktol training text improved ATCO2 further and also improved Jacktol transfer. LibriSpeech moved from 2.32% to 2.43%, remaining close to the acoustic finalist.

A larger fusion weight looked better on ATCO2 development but failed the general-English guardrail. We therefore selected alpha 0.1, not the development-only minimum. This is precisely why decoding parameters need the same domain-plus-general evaluation contract as acoustic training.

**Finding:** once the acoustic model was strong, a small training-text-only language model produced a larger final-mile gain than the tested generic augmentation and phrase-boosting strategies.

## **The headline results, with their contracts**

There is no honest single “best model” without naming the evaluation and product objective.

| Claim | Result | Contract |
| :---- | :---- | :---- |
| Best Silver-only ATCO2 specialist | 26.78% | Parakeet CTC; severe English forgetting |
| Best balanced Silver control | 30.28% ATCO2 / 2.18% LibriSpeech | Parakeet TDT with English replay |
| Best measured acoustic-only ATCO2 result | 19.08% | G3 seed 42, beam 4, locked 2 h Gold test |
| Development-selected averaged acoustic finalist | 19.21% ATCO2 / 2.32% LibriSpeech | Top-three G3 checkpoint average |
| Best ATCO2 result after n-gram fusion | **17.61%** | Frozen averaged G3 \+ balanced 4-gram, alpha 0.1 |
| Best absolute Jacktol reproduction | 5.91% | P3; slightly weaker transfer guardrails |
| Selected balanced Jacktol reproduction | **5.93%** | P2; only two more errors than P3 |

## **What changed our understanding**

### **1\. More hours were not automatically better**

Ten Silver hours captured most of the initial Nemotron gain. Hundreds of additional hours produced diminishing returns until we changed optimization and architecture.

### **2\. Learning rate and forgetting had to be optimized together**

The higher Nemotron learning rate dramatically improved ATC and dramatically harmed English. The apparent breakthrough was incomplete until the guardrail was measured.

### **3\. Replay was a reliable anti-forgetting control**

Both Nemotron and Parakeet retained much more general capability when English examples remained in the sampling mixture. Replay ratios were part of the model objective, not just a data-loader detail.

### **4\. Silver and Gold served different purposes**

Silver data taught broad channel and domain structure. Scarce Gold data was most effective as a later correction. Continuing to replay Silver during that correction weakened the Gold signal.

### **5\. Training order mattered**

The Jacktol reproduction succeeded only after broad mixed-domain stages followed by narrower polishing. A flat run with similar ingredients failed. A dataset list is not a complete recipe.

### **6\. Decoder work was worth doing**

Beam search provided a statistically supported gain. Phrase boosting did not help, but n-gram fusion did. “No more acoustic training” did not mean “no more accuracy was available.”

### **7\. Negative experiments protected us from false conclusions**

Post-hoc normalization, generic radio-proxy noise, hard-slice oversampling, excessive Gold repair, and positive phrase boosting all sounded plausible. Controlled evaluation showed why they should not be promoted.

## **The practical recipe that emerged**

For a new specialized ASR domain with abundant weak labels and scarce human references, our evidence supports this sequence:

1. Freeze a leakage-safe Gold development and test contract.  
2. Measure the untouched pretrained model on both domain and general speech.  
3. Use broad Silver data for initial domain and channel adaptation.  
4. Include general-domain replay whenever the deployed model must preserve general speech.  
5. Refine the best balanced checkpoint on scarce Gold data at a lower learning rate.  
6. Repeat the winning recipe across seeds rather than trusting one run.  
7. Select and export checkpoints before opening the locked test.  
8. Evaluate beam search, averaging, and n-gram fusion as separate controlled stages.  
9. Report the domain score, transfer scores, and forgetting guardrail together.

That process is the real result of the study. The final WER matters, but the reusable contribution is knowing which evidence must accompany it—and which experiment should come next.

# 