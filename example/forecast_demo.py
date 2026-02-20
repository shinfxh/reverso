"""
Demo: autoregressive forecasting on simple synthetic signals.

Run all signals:     python example/forecast_demo.py --signal all
Run one signal:      python example/forecast_demo.py --signal sine
List available:      python example/forecast_demo.py --list
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from reverso.forecast import load_model, forecast


# ---------------------------------------------------------------------------
# Signal generators — each returns float32 array of length n
# ---------------------------------------------------------------------------

def signal_constant(n: int) -> np.ndarray:
    return np.full(n, 5.0, dtype=np.float32)


def signal_linear(n: int) -> np.ndarray:
    return np.linspace(0, 40, n).astype(np.float32)



def signal_sine(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    return (5.0 * np.sin(2 * np.pi * t / 200)).astype(np.float32)


def signal_sawtooth(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    period = 200
    return (10.0 * (t % period) / period).astype(np.float32)


def signal_square(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    return (5.0 * np.sign(np.sin(2 * np.pi * t / 200))).astype(np.float32)



SIGNALS = {
    "constant": ("Constant", signal_constant),
    "linear": ("Linear trend", signal_linear),
    "sine": ("Sine wave", signal_sine),
    "sawtooth": ("Sawtooth wave", signal_sawtooth),
    "square": ("Square wave", signal_square),
}


def run_one(name, label, gen_fn, model, cfg, device, context_length, prediction_length,
            output_dir, flip_invariance=False):
    total_len = context_length + prediction_length
    signal = gen_fn(total_len)
    context_np = signal[:context_length]
    ground_truth = signal[context_length:]

    context_tensor = torch.tensor(context_np, device=device).unsqueeze(0).unsqueeze(-1)
    pred_pos = forecast(
        model, context_tensor, prediction_length,
        seq_len=cfg.seq_len, output_token_len=cfg.output_token_len,
    )
    if flip_invariance:
        pred_neg = forecast(
            model, -context_tensor, prediction_length,
            seq_len=cfg.seq_len, output_token_len=cfg.output_token_len,
        )
        preds_tensor = 0.5 * (pred_pos - pred_neg)
    else:
        preds_tensor = pred_pos
    preds = preds_tensor[0, :, 0].float().cpu().numpy()

    ctx_t = np.arange(context_length)
    pred_t = np.arange(context_length, total_len)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ctx_t, context_np, color="steelblue", label="Context")
    ax.plot(pred_t, ground_truth, color="gray", linestyle="--", label="Ground truth")
    ax.plot(pred_t, preds, color="tomato", label="Forecast")
    ax.axvline(context_length, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title(f"Reverso: {label}")
    ax.legend()
    fig.tight_layout()
    out_path = f"{output_dir}/{name}_forecast.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {label:25s} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Reverso forecast demo on synthetic signals")
    parser.add_argument("--signal", type=str, default="all",
                        help="Signal name, or 'all' to run every signal")
    parser.add_argument("--list", action="store_true", help="List available signals and exit")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/reverso_small/checkpoint.pth")
    parser.add_argument("--args-json", type=str,
                        default="checkpoints/reverso_small/args.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--prediction-length", type=int, default=480)
    parser.add_argument("--output-dir", type=str, default="example")
    parser.add_argument("--flip-invariance", action="store_true",
                        help="Average f(x) with -f(-x) for flip invariance")
    args = parser.parse_args()

    if args.list:
        for name, (label, _) in SIGNALS.items():
            print(f"  {name:15s}  {label}")
        return

    model, cfg = load_model(args.checkpoint, args.args_json, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    if args.signal == "all":
        to_run = list(SIGNALS.items())
    else:
        if args.signal not in SIGNALS:
            print(f"Unknown signal '{args.signal}'. Use --list to see options.")
            return
        to_run = [(args.signal, SIGNALS[args.signal])]

    for name, (label, gen_fn) in to_run:
        run_one(name, label, gen_fn, model, cfg, args.device,
                args.context_length, args.prediction_length, args.output_dir,
                args.flip_invariance)


if __name__ == "__main__":
    main()
