# Releasing this custom node

The package is published at [GitHub](https://github.com/ciru-ai/ComfyUI-CiruImageAccelerator) and [Comfy Registry](https://registry.comfy.org/publishers/ciru/nodes/ciru-image-accelerator). ComfyUI-Manager can install the Registry package. Model weights remain in the official Qwen repositories and are not bundled here.

## Current release

- Version: `0.2.1` ([GitHub release](https://github.com/ciru-ai/ComfyUI-CiruImageAccelerator/releases/tag/v0.2.1)).
- Qwen Image 2.1 text-to-image and reference-image editing are supported. For edits, use `full_evaluations` equal to sampler steps. Prediction on edits requires the experimental `allow_edit_prediction` opt-in.
- The source package includes a 1024 image-edit GUI template, text-to-image templates, API examples, an installation checker, tests, and the validation record in `VALIDATION.md`.

## Next release

1. Update `[project].version`, `CHANGELOG.md`, and `VALIDATION.md`. Test the exact source to be tagged.
2. Commit and push the source, tag the version, and attach a source ZIP to the GitHub release.
3. From the repository root, publish with the official Comfy CLI: `comfy node publish --changelog-file release-notes.md`. Supply the Ciru publisher PAT through the CLI prompt or `--token`; never commit or log it.
4. Check that the Registry version is active and visible publicly, then verify installation and an example workflow from the published package.

The Registry packages Git-tracked files and applies `.comfyignore`. Review the default 12-full schedule, supported attention shapes, model filenames, and model license whenever upstream ComfyUI or Qwen packaging changes.
