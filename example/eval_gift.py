"""
GiftEval evaluation script for Reverso.
"""
import os
import json
import math
import argparse
import csv
from types import SimpleNamespace
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import pandas as pd

from reverso.forecast import load_checkpoint

try:
    from torch.cuda.amp import autocast as autocast_fp
except Exception:
    autocast_fp = None

def numpy_fill(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


class ReversoPredictor:
    """GiftEval predictor for reverso.Model."""

    def __init__(
        self,
        prediction_length: int,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        seq_len: int = 2048,
        input_token_len: int = 2048,
        output_token_len: int = 48,
        e_layers: int = 8,
        d_model: int = 128,
        d_intermediate: int = 512,
        output_bottleneck_dim: int = 48,
        expand_v: float = 1.0,
        state_weaving: int = 1,
        gating_kernel_size: int = 3,
        main_module: str = "conv,attn,conv,attn,conv,attn,conv,attn",
        num_samples: int = 100,
        batch_size: int = 256,
        use_amp: int = 1,
        downsample_factor: int = 1,
        force_flip_invariance: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.prediction_length = int(prediction_length)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.input_token_len = int(input_token_len)
        self.output_token_len = int(output_token_len)
        self.use_amp = int(use_amp)
        self.downsample_factor = int(downsample_factor)
        self.force_flip_invariance = bool(force_flip_invariance)

        args = SimpleNamespace(
            input_token_len=self.input_token_len,
            output_token_len=self.output_token_len,
            seq_len=self.seq_len,
            d_model=int(d_model),
            d_intermediate=int(d_intermediate),
            use_norm=True,
            learn_bias=1,
            output_bottleneck_dim=int(output_bottleneck_dim),
            expand_v=float(expand_v),
            state_weaving=int(state_weaving),
            gating_kernel_size=int(gating_kernel_size),
            main_module=str(main_module),
        )

        from reverso import model as model_impl
        try:
            self.model = model_impl.Model(args).to(self.device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA not usable ({e}); falling back to CPU.")
                self.device = torch.device("cpu")
                self.use_amp = 0
                self.model = model_impl.Model(args).to(self.device)
            else:
                raise
        self.model.eval()

        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print("Warning: checkpoint_path not provided or file not found. Using randomly initialized weights.")

    def _load_checkpoint(self, ckpt_path: str):
        load_checkpoint(self.model, ckpt_path, device=str(self.device))

    def _downsample_if_needed(self, series: torch.Tensor) -> Tuple[torch.Tensor, int]:
        cur = series
        if self.downsample_factor > 1:
            cur = cur[::self.downsample_factor]
        return cur, self.downsample_factor

    def _left_pad_to_len(self, arr: np.ndarray, target_len: int) -> Tuple[np.ndarray, int]:
        if arr.shape[0] >= target_len:
            return arr[-target_len:], 0
        pad_len = target_len - arr.shape[0]
        fill_value = arr[0] if arr.shape[0] > 0 else 0.0
        padding = np.full((pad_len,), fill_value, dtype=arr.dtype)
        return np.concatenate([padding, arr], axis=0), pad_len

    def _prepare_context_matrix(self, context: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        xs = []
        downsample_factors = []

        for c in context:
            cur, downsample_factor = self._downsample_if_needed(c)
            downsample_factors.append(downsample_factor)

            cur_np = cur.detach().cpu().float().numpy()
            cur_np, _ = self._left_pad_to_len(cur_np, self.seq_len)

            x2d = cur_np[None, :]
            x_interp = np.copy(x2d)
            series = x2d[0]
            if np.any(np.isnan(series)):
                valid_mask = ~np.isnan(series)
                if np.sum(valid_mask) >= 2:
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = series[valid_mask]
                    x_interp[0] = np.interp(np.arange(len(series)), valid_indices, valid_values)
                else:
                    x_interp = numpy_fill(x2d)
            ff = numpy_fill(x_interp)
            bf = np.flip(numpy_fill(np.flip(x_interp, axis=1)), axis=1)
            x_imp = np.where(np.isnan(ff), bf, ff)
            x_imp = np.where(np.isnan(x_imp), 0.0, x_imp)
            xs.append(x_imp[0])

        x = torch.tensor(np.stack(xs), device=self.device, dtype=torch.float32).unsqueeze(-1)
        return x, downsample_factors

    def _decode_autoregressive(self, init_ctx: torch.Tensor, use_bf16: bool, downsample_factors: List[int]) -> torch.Tensor:
        B, _, C = init_ctx.shape
        roll_len = int(self.output_token_len)

        target_pred_lens = [int(self.prediction_length) // int(max(1, df)) for df in downsample_factors]
        max_target_pred_len = max(target_pred_lens)
        steps = math.ceil(max_target_pred_len / roll_len)
        preds: List[torch.Tensor] = []
        batch_ctx = init_ctx

        y_mark = torch.zeros(B, self.output_token_len, C, device=self.device, dtype=init_ctx.dtype)

        for _ in range(steps):
            x_in = batch_ctx[:, -self.seq_len:, :]
            x_mark = torch.zeros_like(x_in)

            if autocast_fp is not None and self.use_amp and use_bf16:
                try:
                    with autocast_fp(dtype=torch.bfloat16):
                        outputs = self.model(x_in, x_mark, y_mark)
                except Exception:
                    outputs = self.model(x_in, x_mark, y_mark)
            else:
                outputs = self.model(x_in, x_mark, y_mark)

            out_chunk = outputs[:, -self.output_token_len:, :]
            take_chunk = out_chunk[:, :roll_len, :]
            preds.append(take_chunk)
            batch_ctx = torch.cat([batch_ctx, take_chunk], dim=1)

        return torch.cat(preds, dim=1)

    @torch.no_grad()
    def predict(self, test_data_input, use_bf16_if_available: bool = True):
        from gluonts.itertools import batcher
        from gluonts.model.forecast import SampleForecast

        forecasts = []
        use_bf16 = bool(
            use_bf16_if_available
            and self.device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

        for batch in batcher(test_data_input, batch_size=self.batch_size):
            targets = [torch.tensor(entry["target"], dtype=torch.float32) for entry in batch]
            batch_ctx, downsample_factors = self._prepare_context_matrix(targets)

            pred_pos = self._decode_autoregressive(batch_ctx, use_bf16, downsample_factors)
            if self.force_flip_invariance:
                pred_neg = self._decode_autoregressive(-batch_ctx, use_bf16, downsample_factors)
                pred_full = 0.5 * (pred_pos - pred_neg)
            else:
                pred_full = pred_pos

            if torch.isnan(pred_full).any():
                pf_2d = pred_full.squeeze(-1).detach().cpu().numpy()
                pf_2d = numpy_fill(pf_2d)
                pred_full = torch.tensor(pf_2d, device=pred_full.device, dtype=pred_full.dtype).unsqueeze(-1)

            pred_full_np = pred_full.float().squeeze(-1).detach().cpu().numpy()
            pred_list = []
            for i in range(len(downsample_factors)):
                df = downsample_factors[i]
                target_pred_len = int(self.prediction_length) // int(max(1, df))
                seq = pred_full_np[i, :target_pred_len]
                if df > 1:
                    old_len = len(seq)
                    new_len = int(self.prediction_length)
                    seq = np.interp(np.linspace(0, 1, new_len), np.linspace(0, 1, old_len), seq)
                pred_list.append(seq)
            pred_full_np = np.array(pred_list)

            for i, ts in enumerate(batch):
                start_date = ts["start"] + len(ts["target"])
                samples = np.repeat(pred_full_np[i][None, :], self.num_samples, axis=0)
                forecasts.append(SampleForecast(samples=samples, start_date=start_date))

        return forecasts


# ==========================
# GiftEval evaluation script
# ==========================
from gluonts.ev.metrics import (
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
    MeanWeightedSumQuantileLoss,
)

METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/W ett2/D jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"


def main():
    parser = argparse.ArgumentParser(description="Run Reverso GiftEval across datasets")
    parser.add_argument("--checkpoint", default='checkpoints/reverso_small/checkpoint.pth', help="Path to model checkpoint")
    parser.add_argument("--json_path", default='checkpoints/reverso_small/args.json', help="Path to JSON file with model config overrides")
    parser.add_argument("--output-dir", dest="output_dir", default='results/reverso_small', help="Output directory for results")
    parser.add_argument("--dataset", default=None, help="Filter to specific dataset (substring match)")
    parser.add_argument("--term", default=None, choices=["short", "medium", "long"], help="Filter to specific term")
    parser.add_argument("--force-flip-invariance", dest="force_flip_invariance", action="store_true",
                        help="Average f(x) with -f(-x) for flip invariance")
    parser.add_argument("--downsample-json", dest="downsample_json",
                        default="config/downsample_factors.json",
                        help="Path to JSON with downsample factors per dataset/term")
    args = parser.parse_args()

    # Load model config from JSON if provided
    json_cfg = {}
    if args.json_path and os.path.isfile(args.json_path):
        with open(args.json_path, "r") as f:
            json_cfg = json.load(f)

    # Model hyperparameters
    SEQ_LEN = int(json_cfg.get("seq_len", 2048))
    INPUT_TOKEN_LEN = int(json_cfg.get("input_token_len", 2048))
    OUTPUT_TOKEN_LEN = int(json_cfg.get("output_token_len", 48))
    E_LAYERS = int(json_cfg.get("e_layers", 8))
    D_MODEL = int(json_cfg.get("d_model", 128))
    D_INTERMEDIATE = int(json_cfg.get("d_intermediate", 512))
    OUTPUT_BOTTLENECK_DIM = int(json_cfg.get("output_bottleneck_dim", 48))
    EXPAND_V = float(json_cfg.get("expand_v", 1.0))
    STATE_WEAVING = int(json_cfg.get("state_weaving", 1))
    GATING_KERNEL_SIZE = int(json_cfg.get("gating_kernel_size", 3))
    MAIN_MODULE = str(json_cfg.get("main_module", "conv,attn,conv,attn,conv,attn,conv,attn"))

    DEVICE = "cuda"
    NUM_SAMPLES = 100
    BATCH_SIZE = 256
    USE_AMP = 1

    downsample_map = {}
    if os.path.isfile(args.downsample_json):
        with open(args.downsample_json, "r") as f:
            downsample_map = json.load(f)

    # Setup datasets
    all_datasets = sorted(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
    med_long_set = set(MED_LONG_DATASETS.split())
    all_terms = ["short", "medium", "long"]

    with open("config/dataset_properties.json", "r") as f:
        dataset_properties = json.load(f)

    os.environ.setdefault("GIFT_EVAL", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

    if args.dataset:
        all_datasets = [ds for ds in all_datasets if args.dataset in ds]
        if not all_datasets:
            print(f"No datasets found matching '{args.dataset}'")
            return

    if args.term:
        all_terms = [args.term]

    # Setup output
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"all_results_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "model",
            "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
            "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
            "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
            "eval_metrics/MSIS", "eval_metrics/RMSE[mean]",
            "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "domain", "num_variates",
        ])

    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from gift_eval.data import Dataset

    print(f"Evaluating {len(all_datasets)} datasets, terms: {all_terms}")
    print(f"Flip invariance: {args.force_flip_invariance}")

    for ds_num, ds_name in enumerate(all_datasets):
        if "/" in ds_name:
            ds_key = PRETTY_NAMES.get(ds_name.split("/")[0].lower(), ds_name.split("/")[0].lower())
            ds_freq = ds_name.split("/")[1]
        else:
            ds_key = PRETTY_NAMES.get(ds_name.lower(), ds_name.lower())
            ds_freq = dataset_properties[ds_key]["frequency"]

        print(f"[{ds_num + 1}/{len(all_datasets)}] {ds_name}")

        for term in all_terms:
            if term in ("medium", "long") and ds_name not in med_long_set:
                continue

            ds_config = f"{ds_key}/{ds_freq}/{term}"
            probe = Dataset(name=ds_name, term=term, to_univariate=False)
            to_univariate = probe.target_dim != 1
            dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)

            downsample_key = f"{ds_key}/{ds_freq}/{term}".lower()
            downsample_factor = downsample_map.get(downsample_key, 1)

            info = f"  {term}: {len(dataset.test_data)} instances"
            if downsample_factor > 1:
                info += f", downsample={downsample_factor}"
            print(info)

            predictor = ReversoPredictor(
                prediction_length=dataset.prediction_length,
                checkpoint_path=args.checkpoint,
                device=DEVICE,
                seq_len=SEQ_LEN,
                input_token_len=INPUT_TOKEN_LEN,
                output_token_len=OUTPUT_TOKEN_LEN,
                e_layers=E_LAYERS,
                d_model=D_MODEL,
                d_intermediate=D_INTERMEDIATE,
                output_bottleneck_dim=OUTPUT_BOTTLENECK_DIM,
                expand_v=EXPAND_V,
                state_weaving=STATE_WEAVING,
                gating_kernel_size=GATING_KERNEL_SIZE,
                main_module=MAIN_MODULE,
                num_samples=NUM_SAMPLES,
                batch_size=BATCH_SIZE,
                use_amp=USE_AMP,
                downsample_factor=downsample_factor,
                force_flip_invariance=args.force_flip_invariance,
            )

            res = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=METRICS,
                batch_size=BATCH_SIZE,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ds_config, "reverso",
                    res["MSE[mean]"][0], res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0], res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0], res["sMAPE[0.5]"][0],
                    res["MSIS"][0], res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0], res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    dataset_properties[ds_key]["domain"],
                    dataset_properties[ds_key]["num_variates"],
                ])

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
