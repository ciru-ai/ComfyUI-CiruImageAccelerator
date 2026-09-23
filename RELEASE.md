# Publishing this custom node

The intended distribution is a standalone GitHub repository plus the Comfy Registry, which makes the node installable through ComfyUI-Manager. A Hugging Face repository is optional for examples and benchmark images; it is not needed to install the code. The Qwen model weights remain in the official model repository.

## Current candidate

- Version: `0.1.0`.
- GitHub target: `ciru-ai/ComfyUI-CiruImageAccelerator` (not published by this source candidate).
- Validated source: see `VALIDATION.md` for the exact Strix Halo stack, two integrated image hashes, and the live ComfyUI GUI test.
- The package is source-only. It contains two GUI workflow templates, four API examples, example PNGs, notices, an installation checker, and tests.

## Before publication

1. Confirm the repository name and create the public GitHub repository under `ciru-ai`. Push the committed standalone package, then tag `v0.1.0` and attach the source ZIP.
2. Sign in at [Comfy Registry](https://registry.comfy.org/), create or select the Ciru publisher, and put its **exact** immutable ID in `pyproject.toml` as `PublisherId = "..."`. Do not guess the ID from the GitHub owner name.
3. Create a Registry publishing API key for that publisher. Keep it out of source control and logs. With the repository checked out and the official Comfy CLI installed, run `comfy node publish`, or configure the [official publish action](https://docs.comfy.org/registry/publishing) with the `REGISTRY_ACCESS_TOKEN` repository secret.
4. Verify the public Registry entry, ComfyUI-Manager installation, and one of the included GUI workflows from a clean checkout. Update `VALIDATION.md` with that clean public-install result.

The `PublisherId` and a Registry key are the only account-specific inputs missing from Registry publication. Git clone and source ZIP installation work without them. The Registry packages Git-tracked files and applies `.comfyignore`, so commit the release files and keep development artifacts out of the archive.

For subsequent releases, bump `[project].version` using semantic versioning and test the exact source being tagged. The default 12-full schedule, supported attention shapes, model filenames, and model license should be reviewed when upstream ComfyUI or Qwen packaging changes.
