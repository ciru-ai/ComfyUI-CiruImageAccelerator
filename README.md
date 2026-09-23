# Ciru Image Accelerator for ComfyUI

A single `MODEL` → `MODEL` node for Qwen Image 2.1 text-to-image. It combines adjustable denoiser prediction with a shape-guarded attention path for AMD Strix Halo (gfx1151). The default is **12 full evaluations within 30 sampler steps**. The other 18 steps still run, using predicted denoiser outputs. Increasing the full-evaluation count moves the result toward the full model at a speed cost.

The node contains no model weights. It uses the Qwen Image 2.1 model, text encoder and VAE already installed in ComfyUI.

## Install

This repository is a normal ComfyUI custom node. Once the source repository is published:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/ciru-ai/ComfyUI-CiruImageAccelerator.git
# Restart ComfyUI, then reload the browser.
```

After the Comfy Registry entry is published, search for **Ciru Image Accelerator** in ComfyUI-Manager or run `comfy node install ciru-image-accelerator` with the Comfy CLI. The Registry listing is still pending publisher setup; Git clone and the source ZIP are the available installation paths for this candidate.

For a ZIP installation, extract the release so that `ComfyUI/custom_nodes/ComfyUI-CiruImageAccelerator/__init__.py` exists. No `pip install` step or replacement Torch build is required. On gfx1151, the attention optimization requires a working Triton installation in the same Python environment as ComfyUI; `auto` falls back to ComfyUI's native attention if Triton is missing and logs that choice. Install a Triton build compatible with your existing ROCm/PyTorch stack if you want that optimization. The package never installs dependencies at runtime.

Check the installation with the Python interpreter **and environment used to launch ComfyUI**:

```bash
/path/to/comfy-python /path/to/ComfyUI/custom_nodes/ComfyUI-CiruImageAccelerator/tools/doctor.py
```

The check reports the detected GPU, PyTorch/HIP/Triton versions and expected model paths without loading the weights. If PyTorch cannot import, run it through the same ROCm library-path setup as your ComfyUI launcher. A successful check confirms prerequisites; run one included workflow to verify the attention kernel compiles on your stack.

Use a recent ComfyUI that includes the `TextEncodeQwenImage21` and `QwenImage21Cache` nodes. Download the INT8 ConvRot denoiser and encoder and BF16 VAE from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). Place them in the standard `models/diffusion_models`, `models/text_encoders` and `models/vae` folders shown on that page. These model files are not redistributed here.

## Use

Add **Ciru Image Accelerator** after the Qwen Image 2.1 model loader and cache node, and before `KSampler`:

```text
UNETLoader → QwenImage21Cache → Ciru Image Accelerator → KSampler
```

Use `TextEncodeQwenImage21` for conditioning. Start with Euler, simple scheduler, CFG 1, 30 steps, batch one. Prediction requires at least 20 sampler steps. The included workflows demonstrate 1024 and 2048 square output. Prompt enhancement is optional and stays separate from this node.

In ComfyUI, open **Templates → Extensions → ComfyUI-CiruImageAccelerator** and choose the 1024 or 2048 GUI workflow. The editable node controls are visible between the cache and sampler. The same setups are in `api_examples/` for API users; use `qwen21-1024-default.api.json` or `qwen21-2048-default.api.json`, and the `-full` files for 30/30 references. The GUI workflows in `workflows/` are what ComfyUI shows in its Templates menu.

The node has three controls:

- `enabled`: turn both optimizations on or off.
- `full_evaluations`: 12 by default. Seven is faster with a larger quality tradeoff; 8 is an intermediate option. Set it equal to sampler steps for no prediction. The minimum is 6. All sampler steps still execute.
- `attention`: `auto` uses the measured dense BF16 kernel only for qualified shapes on ROCm gfx1151. `native` leaves ComfyUI's attention unchanged. `strix` requires the gfx1151/Triton path and reports an error if it is unavailable.

At 1024 square the Strix kernel keeps the original interleaved head layout. At 2048 square it packs each head contiguously before the same kernel. Other shapes, masks, dtypes, batches and hardware use the existing attention path. An existing attention override is preserved by `auto`; select `native` when combining this node with another attention extension.

## Example output and speed

The same rewritten elephant prompt, seed 99, CFG 1 and 30 Euler/simple steps on a Radeon 8060S. The comparison changes the number of full denoiser evaluations; it does not change the model weights.

| Full 30/30 | Default 12/30 |
|---|---|
| ![Full elephant at 1024](assets/elephant-1024-full.png) | ![Twelve-full elephant at 1024](assets/elephant-1024-default12.png) |
| 74.06 s | 34.62 s |

At 2048 square, the packed-attention full reference took 420.05 s and the packaged 12/30 path took 173.19 s. The 2048 packaged image was byte-identical to the saved 12/30 reference. These are single runs with model loading and prompt rewriting excluded. The full reference and 1024 images were produced in earlier matched runs; see `VALIDATION.md` for artifact hashes and exact scope.

## Current validation and limits

The package was exercised with Qwen Image 2.1 INT8 ConvRot denoiser and encoder, BF16 VAE, Radeon 8060S/gfx1151, ComfyUI `c194dd00cd42aa18d9dbf27d977bf6b85d9ea565`, Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, Euler/simple, CFG 1, batch one. The combined package produced byte-identical 1024 and 2048 PNGs to previous 12-evaluation research results. See `VALIDATION.md` for the clean-install and GUI workflow tests.

Image editing/reference conditioning is explicitly rejected. The predictor rejects batch sizes above one and unexpected extra model calls. Other samplers, CFG settings, LoRAs, CUDA and other AMD GPUs are not yet qualified for image quality or speed; the attention path stays native outside its gfx1151 shape guard. Prediction changes the generated image, especially at the most aggressive settings. Seven and twelve full evaluations matched the earlier research runner PNGs at a 25-step fixture.

The package does not claim to improve all ComfyUI attention implementations. The large 2048 layout improvement was measured against our earlier interleaved dense kernel, not against a fully optimized alternative. Existing model weights and Qwen's [Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) remain the user's responsibility; the model license restricts commercial use without a separate license.

## Credits

Qwen supplies Qwen Image 2.1 and its model terms. ComfyUI and comfy-kitchen supply the model integration and INT8 ConvRot path. The prediction method is inspired by [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer); this package adapts residual prediction and tested schedules to Qwen Image 2.1. Ciru's contribution includes the scoped ComfyUI node, schedule controls, Strix Halo attention layout diagnosis and integrated backend. This repository contains code and workflows, not Qwen weights.
