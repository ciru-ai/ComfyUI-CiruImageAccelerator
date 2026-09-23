"""Per-request checks against ComfyUI's wrapper API.

Run from the ComfyUI root with its Python and runtime environment:
    PYTHONPATH=. python /path/to/ComfyUI-CiruImageAccelerator/tests/test_lifecycle.py
"""

import importlib.util
import contextvars
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import comfy.patcher_extension
import torch


package_dir = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ciru_image_accelerator_test", package_dir / "__init__.py",
    submodule_search_locations=[str(package_dir)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)


QwenDiffusion = type("QwenDiffusion", (), {})
QwenDiffusion.__module__ = "comfy.ldm.qwen_image21"


class FakeModel:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.load_device = torch.device("cpu")
        self.wrappers = {}

    def clone(self):
        other = FakeModel()
        other.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        other.wrappers = {kind: {key: list(items) for key, items in groups.items()}
                          for kind, groups in self.wrappers.items()}
        return other

    def get_model_object(self, _):
        return QwenDiffusion()

    def get_wrappers(self, kind, key):
        return self.wrappers.get(kind, {}).get(key, [])

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)


class LifecycleTests(unittest.TestCase):
    def wrappers(self, full=12, attention="native"):
        model = package.CiruTurboPrediction().apply(FakeModel(), True, full, attention)[0]
        keys = comfy.patcher_extension.WrappersMP
        outer = model.get_wrappers(keys.OUTER_SAMPLE, "ciru_turbo_prediction")[0]
        diffusion = model.get_wrappers(keys.DIFFUSION_MODEL, "ciru_turbo_prediction")[0]
        return outer, diffusion

    @staticmethod
    def execute_sample(outer, diffusion, interrupt_at=None):
        calls = []

        def full_model(x, *_args):
            calls.append(1)
            return x + len(calls) / 100

        def sample(*_args):
            for index in range(30):
                if index == interrupt_at:
                    raise RuntimeError("cancelled")
                x = torch.zeros((1, 2))
                diffusion(full_model, x, None, None, None, None, {"cond_or_uncond": [0]})

        outer(sample, None, None, None, list(range(31)))
        return len(calls)

    def test_new_sampling_pass_after_exception(self):
        outer, diffusion = self.wrappers()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.execute_sample(outer, diffusion, interrupt_at=5)
        self.assertEqual(self.execute_sample(outer, diffusion), 12)
        self.assertEqual(self.execute_sample(outer, diffusion), 12)

    def test_full_count_uses_no_prediction(self):
        outer, diffusion = self.wrappers(full=30)
        self.assertEqual(self.execute_sample(outer, diffusion), 30)

    def test_disabled_and_existing_attention_are_preserved(self):
        model = FakeModel()
        self.assertIs(package.CiruTurboPrediction().apply(model, False)[0], model)
        previous = lambda *args, **kwargs: None
        model.model_options["transformer_options"]["optimized_attention_override"] = previous
        patched = package.CiruTurboPrediction().apply(model, True, 12, "auto")[0]
        self.assertIs(patched.model_options["transformer_options"]["optimized_attention_override"], previous)
        with self.assertRaisesRegex(ValueError, "Another custom attention"):
            package.CiruTurboPrediction().apply(model, True, 12, "strix")

    def test_unqualified_attention_uses_original(self):
        from ciru_image_accelerator_test.attention_backend import make_attention_override

        active = contextvars.ContextVar("test_pass", default=None)
        state = type("State", (), {"attention_counts": {}})()
        token = active.set(state)
        try:
            override = make_attention_override(active)
            q = torch.zeros((1, 10, 128), dtype=torch.bfloat16)
            marker = object()
            result = override(lambda *_args, **_kwargs: marker, q, q, q, 1)
            self.assertIs(result, marker)
            self.assertEqual(state.attention_counts, {"native": 1})
        finally:
            active.reset(token)

    def test_auto_keeps_1024_native_but_2048_uses_strix(self):
        from ciru_image_accelerator_test import attention_backend

        def inputs(tokens):
            q = torch.empty((1, tokens, 4096), dtype=torch.bfloat16, device="meta")
            return q, q, q, 32, None, {}

        with patch.object(attention_backend, "device_is_gfx1151", return_value=True):
            self.assertFalse(attention_backend.qualified_inputs(*inputs(4096)))
            self.assertTrue(attention_backend.qualified_inputs(*inputs(4096), allow_1024=True))
            self.assertTrue(attention_backend.qualified_inputs(*inputs(16384)))
            self.assertFalse(attention_backend.qualified_inputs(*inputs(8192), allow_1024=True))


if __name__ == "__main__":
    unittest.main()
