# Release validation

## Image-edit GUI template (0.2.1)

The new 1024 edit template has a structurally checked ComfyUI graph: `LoadImage` feeds `TextEncodeQwenImage21` as `images.image_1`, `VAELoader` feeds its `vae` input, its latent output feeds `KSampler`, and `QwenImage21Cache → CiruTurboPrediction → KSampler` remains the model path. The template sets 30 sampler steps, 30 full evaluations, and edit prediction off. This is a packaging and discoverability update; the runtime edit results below were produced with the same node code in 0.2.0. A fresh GUI run of this exact template has not been claimed.

## Full-evaluation image editing

The packaged node completed a one-reference Qwen Image 2.1 edit on the Radeon 8060S with the official INT8 ConvRot denoiser and encoder, BF16 VAE, Euler/simple, CFG 1, and no predicted steps. The 1024-square edit ran 30/30 full evaluations in 317.51 s (102.22 s reference encoding, 209.00 s denoising, 6.30 s decoding). The node reported 30 full and zero predicted calls. Its output changed the requested sky and retained the scene layout, but also increased contrast and texture beyond the requested region. This is a source-fidelity limitation of the observed edit result, not a claim of quality improvement.

A second edit used a 1024-square reference and a 2048-square output canvas with 20/20 full evaluations. It completed in 582.83 s (84.14 s encoding, 475.62 s denoising, 23.07 s decoding). The node reported 640 packed attention calls, 20 full evaluations and zero predicted calls. The image preserved the woman and mech composition while changing the sky. Both images, sidecars and logs are in the research workspace at `phase45/edit-node/`. The previously inactive `qwen-main.service` was left inactive after the tests.

An isolated experimental copy with predicted edits enabled completed the same 1024-square, 30-step edit using 12 full evaluations and 18 predicted calls in 184.99 s (93.98 s encoding, 84.57 s denoising, 6.44 s decoding). Its output had mean RGB pixel difference 2.53/255 versus the 30/30 edit and looked close in the subject and scene layout. Both edits changed contrast and texture outside the requested sky, and the single comparison does not establish general edit fidelity. The release therefore keeps edit prediction behind the explicit `allow_edit_prediction` opt-in. The completed runs cover one reference image; multiple references and other sampler/CFG combinations were not exercised here.

## v0.1.1–v0.1.2: long-prompt Strix attention

Qwen Image 2.1's image queries can attend to additional text keys and values. Three prompt-rewritten 2048-square scenes exercised this case on Radeon 8060S / gfx1151 with the official INT8 ConvRot denoiser and encoder, BF16 VAE, Euler/simple, CFG 1, and 30 sampler steps. The fixed node recorded 384 packed attention calls for 12 full evaluations and 960 for 30 full evaluations; each run also recorded 32 native calls for the separate masked text attention. Timings include encoding, denoising, and VAE decoding, but exclude model loading and prompt rewriting.

| Scene | 12 full / 30 steps | 30 full / 30 steps |
|---|---:|---:|
| Moon sorceress | 186.1 s | 455.9 s |
| Mech pilot | 183.1 s | 452.7 s |
| Ice astronaut | 181.0 s | 440.2 s |

The images, sidecars, and run log are saved in the research workspace at `phase42/rewritten-multires-comparison/`. The nine package unit tests passed in Sozo's ComfyUI environment. The run script restored the previously active `qwen-main.service` after generation.

## v0.1.0

Validated on Radeon 8060S / gfx1151, ComfyUI `c194dd00cd42aa18d9dbf27d977bf6b85d9ea565`, Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, comfy-kitchen `0.2.35`, official Qwen Image 2.1 INT8 ConvRot denoiser and encoder, BF16 VAE. Euler/simple, CFG 1, batch one, 30 sampler steps, 12 full denoiser evaluations and 18 predicted evaluations. Timings below include prompt encoding, denoising and VAE decoding; they exclude loading and prompt rewriting.

| Path | Prompt | Generation | Attention calls | PNG SHA-256 | Result |
|---|---|---:|---:|---|---|
| Packaged 1024, forced Strix attention | Gothic armor | 33.43 s | 384 interleaved | `a68cfb9409b570d0a615807efbbec87c38d1d7b93f19c9dd6caf2d98eed10e9f` | Byte-identical to earlier 12/30 runner PNG |
| Current 1024 default `auto` | Gothic armor | 34.74 s | 416 native | `0980fa9f6e66f1e6db4c2a56c7a7b1f3f3bcd00020c747dfc974336b42cff550` | Same prompt/seed, visually close to forced Strix; RGB pixel MAE 0.80 |
| Packaged 2048, forced Strix attention | Rewritten ice elephant | 173.19 s | 384 packed | `58041b15fb85371ef2f33f0c5e93e2f24f1157464574ac80a46c5c900e706bdc` | Byte-identical to earlier 12/30 runner PNG |
| Installed GUI template, 1024 auto before the default-attention change | Example gothic armor | 34.73 s reported by ComfyUI | 384 interleaved | See ComfyUI output | Completed and displayed in SaveImage node |

The forced-Strix artifacts are in the research workspace at `phase33/full-package/`; their JSON sidecars record the exact prompts and settings. The current 1024 auto artifact and log are at `phase34/default-attention/`. The same 2048 prompt's full 30/30 packed-attention result took 420.05 s in the earlier matched run. The comparison is against our own dense-attention path, not a claim of superiority over every stock or third-party attention library.

A copy of the package loaded successfully from an isolated `custom_nodes` folder. ComfyUI registered `CiruTurboPrediction` with 12/auto defaults and validated all four API examples. With the package installed under its final directory name, both GUI workflows appeared in ComfyUI Templates and opened. The 1024 GUI workflow completed one image and logged the expected 12 full / 18 predicted calls. The previously active `qwen-main.service` was restored after the GPU tests.

The included `tools/doctor.py` was run from the installed `custom_nodes` symlink with the same Python/ROCm environment as ComfyUI. It detected gfx1151, HIP 7.15, Triton 3.8 and all three expected model files. This is a prerequisite check, not a substitute for the GUI generation above.

The committed standalone source was exported as a ZIP, extracted into a fresh temporary directory on Sozo, and loaded through ComfyUI's custom-node loader. Its 12/auto defaults and all four API workflows validated from the extracted archive. The ZIP passed an integrity check; this archive smoke test did not generate another image. `qwen-main.service` remained active.

The `workflows/` directory now contains only the two GUI templates; `api_examples/` contains four API prompts. The GUI templates were adapted from the official Comfy Org Qwen Image 2.1 template under its MIT license (see `NOTICE.md`).

At v0.1.0, the known scope was text-to-image only; reference-image editing was rejected. The full-evaluation edit result above extends that scope. Batch sizes above one and unexpected sampler call patterns remain rejected. In `auto`, the custom attention override is selected only for the BF16, batch-one, 32-head, 16,384-token Qwen shape on gfx1151 used by 2048-square images. The 1024 custom path remains available with explicit `strix` but is not the default; it can be slower on smaller workloads. Other shapes and devices keep their prior attention implementation. LoRAs, CFG >1, other samplers, CUDA, and other AMD GPUs have not been release-qualified for quality or speed.
