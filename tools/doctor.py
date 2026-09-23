"""Read-only installation check for Ciru Image Accelerator.

Run with the Python interpreter used to start ComfyUI:
    python custom_nodes/ComfyUI-CiruImageAccelerator/tools/doctor.py
"""

import argparse
from pathlib import Path
import sys


FILES = {
    "diffusion_models": "qwen_image_2.1_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_8b_int8_convrot.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-dir", type=Path, help="ComfyUI root if this package is outside custom_nodes")
    args = parser.parse_args()
    # Keep a custom_nodes symlink in the path so the default ComfyUI root is
    # still found when the source checkout lives elsewhere.
    package = Path(__file__).absolute().parents[1]
    comfy = args.comfy_dir or package.parents[1]
    print(f"Python: {sys.executable}")
    print(f"Package: {package}")
    print(f"ComfyUI: {comfy}")

    try:
        import torch
    except ImportError:
        print("PyTorch: missing in this interpreter; use ComfyUI's Python")
        return 1
    print(f"PyTorch: {torch.__version__}; HIP: {torch.version.hip or 'none'}")
    architecture = "none"
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
        print(f"GPU: {properties.name}; architecture: {architecture or 'unknown'}")
    else:
        print("GPU: unavailable to PyTorch")

    try:
        import triton
    except ImportError:
        print("Triton: missing; auto attention will use ComfyUI's native path")
        has_triton = False
    else:
        has_triton = True
        print(f"Triton: {triton.__version__}")

    missing = []
    for folder, filename in FILES.items():
        path = comfy / "models" / folder / filename
        present = path.is_file()
        print(f"Model: {'found' if present else 'not in default folder'} {path}")
        if not present:
            missing.append(filename)
    if missing:
        print("Model files can also be supplied through ComfyUI extra model paths.")

    if torch.version.hip and architecture == "gfx1151" and has_triton:
        print("Strix attention prerequisites: present; run an example workflow to verify kernel compilation")
    else:
        print("Strix attention prerequisites: incomplete; prediction can still use native attention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
