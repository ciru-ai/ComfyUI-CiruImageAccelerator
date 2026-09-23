# Release validation

## v0.1.1: long-prompt Strix attention

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

Known scope: text-to-image only. Reference-image editing is rejected. Batch sizes above one and unexpected sampler call patterns are rejected. In `auto`, the custom attention override is selected only for the BF16, batch-one, 32-head, 16,384-token Qwen shape on gfx1151 used by 2048-square images. The 1024 custom path remains available with explicit `strix` but is not the default; it can be slower on smaller workloads. Other shapes and devices keep their prior attention implementation. LoRAs, CFG >1, other samplers, CUDA, and other AMD GPUs have not been release-qualified for quality or speed.
