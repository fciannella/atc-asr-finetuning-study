# Frozen ATCO2 package builder and verifiers

These scripts support the [November 13 v2 release](../../docs/atco2-overlap-removed-release-2026-11-13.md). Audio, transcripts, manifests, exclusion masks and old-package archives remain outside this public repository.

`build_release.py` is the exact executed builder. Place it and `README.dataset.md` under `atc-ablation-v1/experiments/atco2-overlap-removed-release-2026-11-13/` in the restricted OCI workspace. It requires the November 6/13 audit masks and frozen Gold/test manifests at the paths and hashes enforced in the script. Run with NumPy and SoundFile. It refuses to overwrite an existing `package` directory.

The builder writes the AudioFolder package, a deterministic `package.tar.gz`, `build-receipt.json`, and a restricted `private-release-source-manifest.jsonl`. Publish only the package contents; the source manifest contains internal paths and must stay restricted.

`validate_download.py` checks the transported archive and every package/WAV checksum, then loads both HF configurations locally. It requires `build-receipt.json` beside the archive. Use an environment with `datasets`, its audio dependencies (PyTorch/TorchCodec), SoundFile and `huggingface_hub`.

`verify_published.py` checks the tagged private Hub revision, both remote metadata CSVs, sampled remote audio, and all three consumer configuration/split paths. It expects `publication-receipt.json` and the verified local package. Authentication must already be available through the normal HF credential store. It never prints credentials or transcripts.

The [aggregate receipt](../../reports/atco2-overlap-removed-release-2026-11-13.json) preserves the build, local-validation, publication and consumer-verification objects. Deployment and destructive retirement are deliberately not automated by these public scripts. Historical-URL cleanup remains subject to the documented HF permission limitation.
