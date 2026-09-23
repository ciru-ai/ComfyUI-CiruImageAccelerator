"""ComfyUI node for Qwen Image 2.1 prediction and Strix attention."""

import contextvars
import logging

import comfy.patcher_extension
import torch

from .core import SamplingState
from .schedule import DEFAULT_FULL_EVALUATIONS, MIN_FULL_EVALUATIONS


class CiruTurboPrediction:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "full_evaluations": ("INT", {"default": DEFAULT_FULL_EVALUATIONS,
                                         "min": MIN_FULL_EVALUATIONS, "max": 10000, "step": 1}),
            "attention": (["auto", "native", "strix"], {"default": "auto"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Ciru/Image Acceleration"
    DESCRIPTION = ("Qwen Image 2.1 only. Twelve full denoiser evaluations by default. "
                   "Auto keeps native attention at 1024 and uses the gfx1151 path at 2048 square.")

    def apply(self, model, enabled=True, full_evaluations=DEFAULT_FULL_EVALUATIONS,
              attention="auto"):
        if not enabled:
            return (model,)
        if attention not in ("auto", "native", "strix"):
            raise ValueError(f"Unknown attention setting: {attention}")
        diffusion = model.get_model_object("diffusion_model")
        if "qwen_image21" not in type(diffusion).__module__:
            raise ValueError("Ciru Turbo Prediction requires a Qwen Image 2.1 model")
        if model.model_options.get("transformer_options", {}).get("easycache") is not None:
            raise ValueError("Ciru Turbo Prediction cannot be combined with EasyCache")
        if model.get_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                              "ciru_turbo_prediction"):
            raise ValueError("Only one Ciru Turbo Prediction node can be applied to a model")

        patched = model.clone()
        active_pass = contextvars.ContextVar("ciru_turbo_prediction_pass", default=None)

        def outer_sample_wrapper(executor, *args, **kwargs):
            sigmas = args[3] if len(args) > 3 else kwargs["sigmas"]
            steps = len(sigmas) - 1
            state = SamplingState(steps, full_evaluations)
            token = active_pass.set(state)
            try:
                return executor(*args, **kwargs)
            finally:
                logging.info("Ciru Turbo Prediction: %s", state.summary())
                active_pass.reset(token)

        def diffusion_wrapper(executor, *args, **kwargs):
            state = active_pass.get()
            if state is None:
                return executor(*args, **kwargs)
            options = args[5] if len(args) > 5 else kwargs.get("transformer_options", {})
            if not isinstance(options, dict):
                raise RuntimeError("Turbo Prediction could not identify model conditioning")
            branch_id = options.get("cond_or_uncond")
            branch = tuple(branch_id) if branch_id is not None else ("single",)
            refs = args[3] if len(args) > 3 else kwargs.get("ref_latents")
            if refs:
                raise RuntimeError("Turbo Prediction has not been qualified for image editing")
            return state.evaluate(lambda: executor(*args, **kwargs), args[0], branch)

        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                     "ciru_turbo_prediction", outer_sample_wrapper)
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                     "ciru_turbo_prediction", diffusion_wrapper)

        if attention != "native":
            options = patched.model_options.setdefault("transformer_options", {})
            if "optimized_attention_override" in options:
                if attention == "strix":
                    raise ValueError("Another custom attention override is already installed on this model")
                logging.info("Ciru AMD Halo Qwen 2.1 Turbo: preserving an existing attention override")
            else:
                from .attention_backend import device_is_gfx1151, make_attention_override

                device = patched.load_device
                if not isinstance(device, torch.device) or not device_is_gfx1151(device):
                    if attention == "strix":
                        raise RuntimeError("Strix attention requires a ROCm gfx1151 GPU")
                    logging.info("Ciru AMD Halo Qwen 2.1 Turbo: native attention on %s", device)
                else:
                    try:
                        options["optimized_attention_override"] = make_attention_override(
                            active_pass, allow_1024=attention == "strix")
                    except RuntimeError:
                        if attention == "strix":
                            raise
                        logging.warning("Ciru AMD Halo Qwen 2.1 Turbo: Triton unavailable; using native attention")
        return (patched,)


NODE_CLASS_MAPPINGS = {"CiruTurboPrediction": CiruTurboPrediction}
NODE_DISPLAY_NAME_MAPPINGS = {"CiruTurboPrediction": "Ciru AMD Halo Qwen 2.1 Turbo"}
