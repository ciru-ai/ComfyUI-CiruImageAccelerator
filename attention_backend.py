"""Shape-guarded attention selection for the measured Strix Halo path."""

import logging

import torch


AUTO_QUERY_TOKENS = (16384,)
FORCED_QUERY_TOKENS = (4096, 16384)
QUALIFIED_HEADS = 32
QUALIFIED_HEAD_DIM = 128


def device_is_gfx1151(device) -> bool:
    if not torch.version.hip or device.type != "cuda":
        return False
    try:
        properties = torch.cuda.get_device_properties(device)
    except (RuntimeError, AssertionError):
        return False
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0] == "gfx1151"


def qualified_inputs(q, k, v, heads, mask, kwargs, *, allow_1024=False) -> bool:
    if mask is not None or kwargs.get("skip_reshape", False):
        return False
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return False
    allowed_tokens = FORCED_QUERY_TOKENS if allow_1024 else AUTO_QUERY_TOKENS
    # Qwen Image 2.1 appends prompt tokens to the image keys/values. The
    # image queries remain 4096 or 16384 tokens, even for long prompts.
    if q.shape[1] not in allowed_tokens or k.shape[1] < q.shape[1] or v.shape[1] != k.shape[1]:
        return False
    if heads != QUALIFIED_HEADS or q.shape[2] != heads * QUALIFIED_HEAD_DIM:
        return False
    if k.shape[2] != q.shape[2] or v.shape[2] != q.shape[2]:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    return q.device == k.device == v.device and device_is_gfx1151(q.device)


def make_attention_override(active_pass, *, allow_1024=False):
    # Import Triton only when this backend is selected. Portable prediction can
    # load on systems without Triton or AMD hardware.
    try:
        from .attention_dense import attention as dense_attention
    except ImportError as exc:
        raise RuntimeError(
            "Strix attention requires a working Triton installation in ComfyUI's Python environment"
        ) from exc

    def override(original, q, k, v, heads, mask=None, **kwargs):
        if not qualified_inputs(q, k, v, heads, mask, kwargs, allow_1024=allow_1024):
            state = active_pass.get()
            if state is not None:
                state.attention_counts["native"] = state.attention_counts.get("native", 0) + 1
            return original(q, k, v, heads, mask=mask, **kwargs)

        length = q.shape[1]
        q, k, v = (tensor.reshape(1, -1, heads, QUALIFIED_HEAD_DIM).transpose(1, 2)
                   for tensor in (q, k, v))
        if length == 16384:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            layout = "packed"
        else:
            layout = "interleaved"
        output = dense_attention(q, k, v, 128, 32, 8)
        state = active_pass.get()
        if state is not None:
            state.attention_counts[layout] = state.attention_counts.get(layout, 0) + 1
        if kwargs.get("skip_output_reshape", False):
            return output
        return output.transpose(1, 2).reshape(1, length, heads * QUALIFIED_HEAD_DIM)

    shapes = "1024 interleaved and 2048 packed" if allow_1024 else "2048 packed; 1024 native"
    logging.info("Ciru Image Accelerator: gfx1151 dense attention ready (%s)", shapes)
    return override
