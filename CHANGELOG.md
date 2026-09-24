# Changelog

## 0.2.0

- Support reference-image editing when `full_evaluations` equals sampler steps, retaining the 2048 Strix packed-attention path.
- Add an explicit `allow_edit_prediction` opt-in for experimental faster edits. The default blocks predicted edit steps.
- Document the 1024 and 2048 edit runs and the observed source-fidelity limits.

## 0.1.3

- Rename the Registry listing and ComfyUI node label to Ciru AMD Halo Qwen 2.1 Turbo. The install ID and workflow node type stay the same.

## 0.1.2

- Explain the node connection and recommended starting settings directly in the Registry description and README introduction.

## 0.1.1

- Keep the packed Strix attention path active for long Qwen Image 2.1 prompts whose image queries attend to additional text keys and values.

## 0.1.0

- Add a Qwen Image 2.1 ComfyUI node with adjustable full-denoiser evaluations. The default uses 12 full evaluations across 30 sampler steps.
- Add the shape-guarded Strix Halo attention path for 2048-class images. Native attention remains the default at 1024 and smaller attention shapes.
- Include 1024 and 2048 GUI workflows, four API examples, an environment checker, source notices, and validation results.
