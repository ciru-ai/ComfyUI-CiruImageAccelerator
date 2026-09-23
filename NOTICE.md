# Notices and provenance

Ciru Image Accelerator code is distributed under GPL-3.0-only (see `LICENSE`). No model weights are included.

The example GUI workflows were adapted from Comfy Org's Qwen Image 2.1 text-to-image workflow template. Its MIT license is included in `WORKFLOW_TEMPLATE_LICENSE`. The workflows were flattened and modified to add QwenImage21Cache and Ciru Image Accelerator nodes, a 30-step default, and 1024/2048 presets. Source: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json

The residual prediction method is inspired by TaylorSeer (Shenyi-Z/TaylorSeer). The predictor here is an independent Qwen Image 2.1 integration, not a copy of TaylorSeer source. Source: https://github.com/Shenyi-Z/TaylorSeer

The attention backend is Ciru's shape-guarded packaging of the Strix Halo kernel and layout work from this repository's research code. It uses PyTorch and Triton at runtime and does not contain code or weights from Qwen Image 2.1.

The Qwen Image 2.1 model and Comfy-Org's model files are licensed separately under the Qwen Research License: https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE
