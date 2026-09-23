"""Validate this custom node and its API workflows in an existing ComfyUI venv.

Run from the ComfyUI root with its Python interpreter and runtime environment:
    PYTHONPATH=. python /path/to/ComfyUI-CiruImageAccelerator/tests/smoke_comfy.py
"""

import asyncio
import json
import os
from pathlib import Path
import sys


PACKAGE = Path(os.environ.get("CIRU_PACKAGE_DIR", Path(__file__).resolve().parents[1]))
WORKFLOWS = Path(os.environ.get("CIRU_WORKFLOW_DIR", PACKAGE / "api_examples"))

# ComfyUI parses process arguments on import. No model is loaded by this check.
sys.argv = ["main.py", "--gpu-only"]

import comfy.options

comfy.options.enable_args_parsing()
import nodes
import execution


async def main():
    # Load only the Qwen nodes needed by these examples. The full extras scan
    # expects a live PromptServer and can warn in this headless check.
    qwen_nodes = Path(nodes.__file__).parent / "comfy_extras" / "nodes_qwen.py"
    if not await nodes.load_custom_node(str(qwen_nodes), module_parent="comfy_extras"):
        raise RuntimeError(f"ComfyUI could not load {qwen_nodes}")
    if not await nodes.load_custom_node(str(PACKAGE)):
        raise RuntimeError(f"ComfyUI could not load {PACKAGE}")
    cls = nodes.NODE_CLASS_MAPPINGS["CiruTurboPrediction"]
    assert cls.INPUT_TYPES()["required"]["attention"][1]["default"] == "auto"
    assert cls.INPUT_TYPES()["required"]["full_evaluations"][1]["default"] == 12
    print(f"Loaded {PACKAGE.name} with 12-full/auto defaults")

    workflows = sorted(WORKFLOWS.glob("*.api.json"))
    if not workflows:
        raise RuntimeError(f"No API workflows found in {WORKFLOWS}")
    for path in workflows:
        prompt = json.loads(path.read_text())
        valid, error, _, node_errors = await execution.validate_prompt(path.stem, prompt, None)
        if not valid:
            raise AssertionError(f"{path.name}: {error}; nodes={node_errors}")
        print(f"Valid: {path.name}")


if __name__ == "__main__":
    asyncio.run(main())
