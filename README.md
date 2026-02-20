
<h1 align="center">Reverso</h1>

<h3 align="center">
  Efficient time-series foundation models for zero-shot forecasting.
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2602.17634">Paper</a> •
  <a href="https://github.com/shinfxh/reverso">GitHub</a> •
  <a href="https://huggingface.co/shinfxh/reverso">Hugging Face</a>
</p>

<p align="center">
  By combining long convolutions with linear RNN layers, Reverso matches the performance of transformer-based models that are over <b>100x larger</b>.
</p>

## Key Results

<p align="center">
  <img src="figures/gift_eval_pareto_overall.png" width="800">
</p>

Evaluated on [Gift-Eval](https://github.com/SalesforceAIResearch/gift-eval), a comprehensive time-series forecasting benchmark spanning 97 tasks within 23 datasets across 7 domains.

| Model | Params | Gift-Eval MASE |
|---|---|---|
| **Reverso** | 2.6M | **0.711** |
| Reverso-Small | 550K | 0.726 |
| Reverso-Nano | 200K | 0.760 |

For reference, Xihe-Max (1.5B params) achieves 0.711 and TimesFM-2.5 (200M params) achieves 0.705 on the same benchmark.

## Installation

```bash
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv
pip install --no-build-isolation git+https://github.com/HazyResearch/flash-fft-conv.git
pip install -e .
```

### Requirements

- Python >= 3.11
- PyTorch 2.6.0
- CUDA-compatible GPU
- [FlashFFTConv](https://github.com/HazyResearch/flash-fft-conv)
- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)

## Model Architecture

<p align="center">
  <img src="figures/new_arch.png" width="800">
</p>

Reverso uses a hybrid architecture that interleaves:
1. **Long convolution layers** ([FlashFFTConv](https://github.com/HazyResearch/flash-fft-conv)) with gated short convolutions
2. **DeltaNet layers** for modeling sequential dependencies
3. **MLP layers** for channel mixing
4. **Attention-based decoder head** for producing the final forecast

Input sequences are normalized to [0, 1] and processed point-wise (no patching). The model predicts 48 time steps at a time and rolls out autoregressively for longer horizons.

| Config | Params | Layers | d_model |
|---|---|---|---|
| Reverso | 2.6M | 8 | 128 |
| Reverso-Small | 550K | 4 | 64 |
| Reverso-Nano | 200K | 2 | 32 |

The modeling code is in [`reverso/`](reverso/).

## Quick Start

```python
import torch
from reverso import load_model, forecast

model, cfg = load_model(
    "checkpoints/reverso_small/checkpoint.pth",
    "checkpoints/reverso_small/args.json",
    device="cuda",
)

context = torch.full((1, 2048, 1), 5.0, device="cuda")  # (batch, seq_len, 1)
predictions = forecast(
    model, context,
    prediction_length=96,
    seq_len=cfg.seq_len,
    output_token_len=cfg.output_token_len,
)
print(predictions.shape)  # (1, 96, 1)
```

## Examples

Install the example dependencies first:

```bash
pip install -r example/requirements.txt
```

### Forecast Demo

Run Reverso on synthetic signals (constant, linear, sine, sawtooth, square):

```bash
python example/forecast_demo.py --signal all
```

Use `--signal sine` to run a single signal, or `--list` to see all options.

### Gift-Eval Benchmark

To reproduce the benchmark results, first follow the [Gift-Eval setup instructions](https://github.com/SalesforceAIResearch/gift-eval) to install the package and download the data. By default the script looks for the data in `data/` at the repository root. You can override this by setting the `GIFT_EVAL` environment variable:

```bash
export GIFT_EVAL=/path/to/gift-eval-data
```

Then run:

```bash
python example/eval_gift.py \
    --checkpoint checkpoints/reverso_small/checkpoint.pth \
    --output-dir results/ \
    --force-flip-invariance
```

> **Note:** Dependencies within Gift-Eval may conflict with those in Reverso. If you encounter issues, try upgrading `huggingface_hub`:
> ```bash
> pip install --upgrade huggingface_hub
> ```
> **Note:** While running this benchmark, it is recommended to use flip invariance, but this requires two forward passes of the model. The inference speed is also not fully optimized and could be further sped up. 

## Available Checkpoints

| Model | Status | Path |
|---|---|---|
| Reverso-Small (550K) | Available | `checkpoints/reverso_small/` |
| Reverso (2.6M) | Coming soon | — |
| Reverso-Nano (200K) | Coming soon | — |

## Citation

```bibtex
@misc{fu2026reversoefficienttimeseries,
      title={Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting},
      author={Xinghong Fu and Yanhong Li and Georgios Papaioannou and Yoon Kim},
      year={2026},
      eprint={2602.17634},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.17634},
}
```

## License

This project is licensed under the [MIT License](LICENSE).
