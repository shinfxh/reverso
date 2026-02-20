"""Autoregressive forecasting utilities for Reverso."""
import json
import math
from types import SimpleNamespace

import torch

from reverso.model import Model


def load_checkpoint(model: Model, checkpoint_path: str, device: str = "cuda"):
    """Load a checkpoint into an existing Reverso model.

    Handles common checkpoint formats (raw state_dict, or dicts keyed by
    "model_state_dict", "state_dict", "model", "ema", "ema_state_dict")
    and strips the "module." prefix left by DDP.
    """
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = raw
    if isinstance(raw, dict):
        for k in ("model_state_dict", "state_dict", "model", "ema", "ema_state_dict"):
            if k in raw and isinstance(raw[k], dict):
                state_dict = raw[k]
                break
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def load_model(checkpoint_path: str, args_json: str, device: str = "cuda"):
    """Load a Reverso model from a checkpoint and config JSON.

    Returns:
        (model, cfg) tuple.
    """
    with open(args_json) as f:
        cfg = SimpleNamespace(**json.load(f))

    model = Model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model, cfg


@torch.no_grad()
def forecast(
    model: Model,
    context: torch.Tensor,
    prediction_length: int,
    seq_len: int,
    output_token_len: int,
    use_amp: bool = True,
) -> torch.Tensor:
    """Autoregressive multi-step forecast.

    Follows the rollout pattern from eval_gift.py's _decode_autoregressive.

    Args:
        model: Reverso Model (already on the target device, in eval mode).
        context: Input context tensor of shape (B, L, 1).
        prediction_length: Number of future steps to predict.
        seq_len: Model's context window length (cfg.seq_len).
        output_token_len: Steps produced per model call (cfg.output_token_len).
        use_amp: Whether to use bfloat16 autocast (requires CUDA).

    Returns:
        Predictions tensor of shape (B, prediction_length, 1).
    """
    device = context.device
    B, _, C = context.shape
    roll_len = output_token_len
    steps = math.ceil(prediction_length / roll_len)

    batch_ctx = context
    preds = []

    y_mark = torch.zeros(B, output_token_len, C, device=device, dtype=context.dtype)

    for _ in range(steps):
        x_in = batch_ctx[:, -seq_len:, :]
        x_mark = torch.zeros_like(x_in)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(x_in, x_mark, y_mark)
        else:
            outputs = model(x_in, x_mark, y_mark)

        out_chunk = outputs[:, -output_token_len:, :]
        take_chunk = out_chunk[:, :roll_len, :]
        preds.append(take_chunk)
        batch_ctx = torch.cat([batch_ctx, take_chunk], dim=1)

    return torch.cat(preds, dim=1)[:, :prediction_length, :]
