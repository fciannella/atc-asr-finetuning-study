# Frozen ATCO2 release with known overlap removed

Created **November 13, 2026**. The current private [NVIDIA dataset](https://huggingface.co/datasets/nvidia/atco2-eval) now contains **1,503 clips / 90.6411 minutes**, replacing the original 1,908-clip two-hour package.

Version: **`atco2-known-overlap-removed-v2-2026-11-13`**. HF tag: [`v2-2026-11-13`](https://huggingface.co/datasets/nvidia/atco2-eval/tree/v2-2026-11-13). Exact release revision: [`747770ceb0adaf061e28e7948beddb3bd332f631`](https://huggingface.co/datasets/nvidia/atco2-eval/tree/747770ceb0adaf061e28e7948beddb3bd332f631).

## Contents and model eligibility

| HF configuration / split | Clips | Minutes | Use |
|---|---:|---:|---|
| `default` / `test` | 1,269 | 75.7615 | Original test after the completed public ATCO2 and Jacktol exclusions; the current comparison for existing P4/N4. |
| `default` / `extension` | 234 | 14.8796 | Remaining former Gold-training clips; not held out for P4/N4, which already trained on them. |
| `combined` / `test` | 1,503 | 90.6411 | Both groups, for future eligible checkpoint lineages that exclude these recordings throughout training and selection, including Silver exposure. |

The combined view references the same files. The default splits are disjoint, with no shared source-record IDs or airport/date groups. All 1,503 clips have unique IDs and unique complete PCM hashes.

```python
from datasets import load_dataset

repo = "nvidia/atco2-eval"
revision = "747770ceb0adaf061e28e7948beddb3bd332f631"

# Existing P4/N4 comparison.
test = load_dataset(repo, "default", split="test", revision=revision)

# Future eligible lineages only.
combined = load_dataset(repo, "combined", split="test", revision=revision)
```

Access requires permission to the private repository. Install the `datasets` audio dependencies for decoding (the verification environment used datasets 5.2.0, PyTorch, TorchCodec and SoundFile).

## Freeze and validation

The builder consumes the frozen test and Gold manifests and the exact audited exclusion masks. It removes **639 original-test clips** and **159 Gold-training clips**, retaining the 1,503-clip union. [Original test audit](atco2-training-overlap-audit-2026-11-06.md) and [Gold audit](gold-public-overlap-audit-2026-11-13.md) explain the matching methods and their limits.

The package contains 16-kHz mono PCM WAVs, unchanged reference text, provenance fields, `release.json`, and `SHA256SUMS`. No internal source paths are included in the HF rows. The package records actual sampled durations; the tiny sub-millisecond difference from summed manifest durations follows the established integer-sample cropping used by the audits.

- All 1,503 output WAVs passed independent source-PCM verification; all 1,269 original-test crops match the audio used for the existing scores.
- All 1,509 local package files and the transported archive passed checksum verification. Both default splits and the combined configuration loaded successfully with the Hugging Face datasets library.
- All 1,509 HF file hashes matched the package after upload and again after history squashing.
- Remote metadata, streamed configurations and sampled audio passed checks at the release tag. The separate web Dataset Viewer endpoint still returns HTTP 500; browser-viewer playback is not claimed as verified.
- Build job **38488021** completed on OCI with exit code `0:0` in **26 seconds**.
- Frozen `release.json` SHA-256: `432ec4640ae77277bb6324d34a17e70d213d3bc76a9504f60c2f0eb13002eb26`.
- `SHA256SUMS` SHA-256: `c3a5f334876d67a6768f6a0b6885ffbc21fb85e5b28075e6c77f98d8027f6f1e`.
- Transfer archive SHA-256: `0bacca4f94c255c2590820679329a3a75e7155167756456f603c4d72e00bafb2`.

The restricted OCI canonical directory is `atc-ablation-v1/releases/atco2-known-overlap-removed-v2-2026-11-13/`. Its `package` link points to the sealed package, and `protected-evaluation.jsonl` identifies all 1,503 clips and their source recordings for future training exclusions. This version must not be silently regenerated or reassigned; a changed membership requires a new version. Historical training exposure remains historical exposure.

## HF retirement status

The current NVIDIA repository contains only the new package. The obsolete `fciannella/atc02-preview` repository was **deleted**, the old converted-Parquet branch was **removed**, and NVIDIA `main` was reduced to one release commit. Both old 1,908-clip packages were fully archived and hash-verified in restricted local storage before these changes.

**Historical URL purge remains incomplete.** HF still resolves the old NVIDIA dataset commit `0b408685369df18230bfbfc73e03e4070e128b23` and old conversion commit `19121d47d82a905ffdae09a15fb01d0d9460e983` even after branch deletion and history squashing. The old main revision still lists 1,908 WAVs. Creating a fresh private `nvidia/atco2-eval-v2-2026-11-13` repository was rejected with **HTTP 403: no repository-creation rights in `nvidia`**. An NVIDIA HF admin must provision a clean repository or grant the necessary rights, or arrange permanent removal of unreachable historical data. Do not describe the historical data as fully purged.

The current release retains private visibility. Public-release permissions and licensing are not changed by this replacement.

## Scope

“Known overlap removed” means the identified matches in the completed pinned ATCO2/Jacktol audits are excluded. It does not certify every public corpus, all transformations, or inaccessible model-training ancestry. The former Gold extension is specifically labeled because it overlaps P4/N4 training. This package is not a universal zero-contamination benchmark and is not a new blind test.

The existing benchmark scores remain scores on the 1,269-clip test view. No combined 1,503-clip WER is claimed. The combined duration remains **29.3589 minutes short of two hours**.

See the [aggregate release receipt](../reports/atco2-overlap-removed-release-2026-11-13.json) and [builder/verification scripts](../scripts/atco2-overlap-removed-release-2026-11-13/README.md).
