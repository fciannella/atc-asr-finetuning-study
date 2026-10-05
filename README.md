# ATC ASR Fine-Tuning Study

Private working repository for the ATCO2 and Jacktol ASR adaptation study. It collects the educational article, experiment narrative, frozen result summaries, and a small set of reproducibility scripts without copying the underlying audio, model checkpoints, or internal training manifests.

## Start here

- [Fine-tuning guide](docs/fine-tuning-guide.md) explains the skill-guided ASR adaptation workflow.
- [Experiments and results](docs/experiments-and-results.md) follows the study from Silver adaptation through Gold refinement and n-gram fusion.
- [Fine-tuning blog](docs/fine_tuning_blog.md) is the publication-oriented working document.
- [ATCO2 Gold and Jacktol comparison](docs/atco2-jacktol-comparison.md) summarizes the two data contracts and the overlap audit.
- [Experiment narrative](docs/experiment-narrative.md) preserves the longer chronological account.

## Repository layout

```text
docs/       Human-readable articles, policies, and study notes
reports/    Frozen, machine-readable result snapshots
scripts/    Selected data-audit, evaluation, and experiment-selection utilities
```

The report snapshots contain aggregate measurements and experiment metadata. They do not contain audio, transcripts, model weights, credentials, or internal filesystem paths.

## Skills used

The work used two complementary agent skills:

- ASR customization orchestration: <https://github.com/NVIDIA/skills/tree/main/skills/nemotron-asr-finetune>
- NeMo ASR fine-tuning execution: <https://github.com/NVIDIA-NeMo/Speech/tree/main/.claude/skills/nemo-speech-asr-finetune>

The orchestration skill defines the decision process—baseline, cheapest sufficient customization path, data contract, domain evaluation, and forgetting guardrail. The NeMo skill owns checkpoint-aware data preparation, training, checkpoint selection, and standalone evaluation.

## Data and model artifacts

This repository intentionally does **not** redistribute:

- ATCO2 audio or transcripts;
- Jacktol audio or local materializations;
- LibriSpeech or UWB audio;
- NeMo checkpoints, optimizer state, or exported models;
- W&B credentials or raw run storage;
- manifests containing private storage paths.

The planned ATCO2 community release must be obtained from its approved distribution location and license once those details are finalized. Jacktol is available from <https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset> under its upstream terms.

## Reproducibility boundary

The scripts are included as reference implementations. Most accept explicit paths and expect manifests or prediction files that are not committed here. Before running them, recreate the required data under an authorized storage location and inspect each command's `--help` output.

The authoritative metric is standalone normalized WER: lowercase, Unicode diacritic folding, punctuation and symbol removal, and whitespace normalization. Development sets select checkpoints and decoder settings; locked tests are opened only after selection.

## Status

This is a private review repository. Results and prose remain subject to technical, data-release, and editorial review. No dataset license or model license is granted by the presence of documentation in this repository.
