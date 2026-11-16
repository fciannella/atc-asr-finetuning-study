# November 16 Nemotron Transcribe recovery

The active scope is to finish the Silver stage at 20,000 updates, select among steps 9,000, 15,000 and 20,000 using the existing development WER and English guard, and evaluate the locked checkpoint against the original base on the 1,269-row common test. Gold is held by explicit user decision pending more separate human-labeled data. `chain.json` contains no Gold jobs.

`sampler_restore.py` patches the pinned TorchData zero-worker loader when a Lhotse sampler and iterator are the same object. Identical redundant states are restored once; unequal states fail. This prevents an extra pending sampler replay at iterator reset. `audit_restore.py` reproduces the original bug and checks exact continuation (audio identities, crops and token arrays) for packed and unpacked samplers across eight ranks, epoch transitions, unequal-state rejection and ordinary nonaliased loaders. It requires the original private Silver configuration, pinned model/tokenizer and manifests and is not a standalone public dataset test.

The audit runs with the pinned NeMo Speech revision `e5d79d5a49cbec49d79440b0070aac0c32a8dbb8`, Torch 2.11.0+cu129, Lhotse 2.0.0a6, and Lightning 2.4.0. The two packing files retain the already-tested November 9 optimizations. No package upgrade is part of recovery.

Run in that environment, with this directory and the original run on `PYTHONPATH`:

```bash
python audit_restore.py /path/to/private/original-run 0
```

Repeat for ranks 0 through 7. The remote launcher requires all eight passing reports and frozen code/data hashes before starting eight-GPU training. The first resume replays the legacy step-9,000 sampler once; the second short resume must finish within 60 minutes before longer training chunks are allowed.

See the [experiment record](../../docs/nemotron-transcribe-p4-curriculum-2026-11-07.md) and [aggregate report](../../reports/nemotron-transcribe-p4-curriculum-2026-11-07.json). Audio, transcripts, checkpoint weights and row-level state remain in restricted storage.
