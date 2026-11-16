# Parakeet Gold / synthetic campaign

See [execution record](../../docs/parakeet-gold-synthetic-2026-11-16.md). Training scripts run in the pinned OCI container and expect the private, hash-frozen manifests and checkpoints at the paths in `contract.json`. These data are deliberately absent here. `code-freeze.json` is the remote ASR execution freeze; local synthetic scripts have separate provenance and are not silently added to that freeze.

`dd-run.py` uses the project `designer-config` and reads the existing local NVIDIA credential into process memory. Adjust its credential-file path for another authorized environment; never put credentials in this repository. The retained preview builder uses constrained templates; `atc_freeform_rejected.py` records the rejected free-form approach for reproducibility.

Run `data-designer validate` and `preview --save-results` through `dd-run.py`. Synthetic artifacts remain previews pending listening/phraseology review. No synthetic training arm is launched by these scripts. Raw licensed audio, manifests, predictions and private error reviews stay outside GitHub.
