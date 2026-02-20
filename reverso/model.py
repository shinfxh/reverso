"""
Reverso: conv-attention hybrid for time series forecasting.
"""
import torch
from torch import nn
import torch.nn.functional as F
from flashfftconv import FlashFFTConv
from fla.layers import DeltaNet
from typing import Any


class Gating(nn.Module):
    def __init__(self, channels, temporal_kernel=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2, groups=channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, d_intermediate=0):
        super().__init__()
        self.norm = nn.LayerNorm(d_out)
        if d_intermediate and d_intermediate > 0:
            self.linear = nn.Linear(d_in, d_intermediate)
            self.linear_final = nn.Linear(d_intermediate, d_out)
        else:
            self.linear = nn.Linear(d_in, d_out)
            self.linear_final = nn.Identity()
        self.activation = nn.ReLU()
        self.skip_linear = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        if x.ndim == 3:
            x = x.permute(0, 2, 1)
        residual = self.skip_linear(x)
        y = self.linear(x)
        y = self.activation(y)
        y = self.linear_final(y)
        y = self.norm(y)
        y = residual + y
        if y.ndim == 3:
            y = y.permute(0, 2, 1)
        return y


class CNNBlock(nn.Module):
    def __init__(self, channels, seq_len, flashfftconv, gating_kernel_size=3):
        super().__init__()
        self.flashfftconv = flashfftconv
        self.k = nn.Parameter(torch.randn(channels, seq_len, dtype=torch.float32))
        self.pregate = Gating(channels, gating_kernel_size)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        residual = x
        x_conv = x.contiguous().to(torch.bfloat16)
        pregate = self.pregate(x_conv.float()).to(x_conv.dtype)
        postgate = torch.ones_like(x_conv)
        out = self.flashfftconv(x_conv, self.k, pregate=pregate, postgate=postgate)
        out = self.activation(out)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = out + residual
        return out


class AttentionBlock(nn.Module):
    def __init__(self, d_model, expand_v, state_weaving=False, is_intermediate=False):
        super().__init__()
        self.state_weaving = state_weaving
        self.is_intermediate = is_intermediate
        self.attention = DeltaNet(
            mode='chunk',
            d_model=d_model,
            expand_k=1.0,
            expand_v=expand_v,
            num_heads=4,
            use_beta=True,
            use_gate=False,
            use_short_conv=True,
            conv_size=4,
            allow_neg_eigval=False,
            qk_activation='silu',
            qk_norm='l2',
            layer_idx=0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        residual = x_t
        if self.state_weaving and self.is_intermediate:
            x_t = x_t.clone()
            x_t[:, 0:1, :] = x_t[:, 0:1, :] + x_t[:, -1:, :]
        attn_out = self.attention(hidden_states=x_t, attention_mask=None)
        if isinstance(attn_out, tuple):
            out = attn_out[0]
        else:
            out = attn_out
        out = self.norm(out)
        out = out + residual
        out = out.transpose(1, 2)
        return out


class Model(nn.Module):
    """
    Reverso: conv-deltanet hybrid for time series forecasting.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.input_token_len = configs.input_token_len
        self.output_token_len = configs.output_token_len
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm

        self.embedding = nn.Linear(1, self.d_model, bias=False)
        self.shared_flashfftconv = FlashFFTConv(self.seq_len, dtype=torch.bfloat16)

        d_intermediate = configs.d_intermediate
        expand_v = getattr(configs, 'expand_v', 1.0)
        state_weaving = getattr(configs, 'state_weaving', False)
        gating_kernel_size = getattr(configs, 'gating_kernel_size', 3)
        module_list = [m.strip() for m in configs.main_module.split(',')]
        e_layers = len(module_list)

        layers = []
        for i, layer_type in enumerate(module_list):
            if layer_type == 'conv':
                layers.append(CNNBlock(
                    self.d_model, self.seq_len, self.shared_flashfftconv, gating_kernel_size,
                ))
            elif layer_type == 'attn':
                is_intermediate = (i > 0) and (i < e_layers - 1)
                layers.append(AttentionBlock(
                    self.d_model, expand_v, state_weaving, is_intermediate,
                ))
            else:
                raise ValueError(f'Invalid layer type: {layer_type}')
            layers.append(MLPBlock(self.d_model, self.d_model, d_intermediate))
        self.layers = nn.Sequential(*layers)

        output_bottleneck_dim = getattr(configs, 'output_bottleneck_dim', self.output_token_len)
        self.head = nn.Linear(self.input_token_len, output_bottleneck_dim, bias=configs.learn_bias)
        self.simple_q_proj = nn.Linear(self.d_model, self.d_model)
        self.key_proj = nn.Linear(self.d_model, self.d_model)
        self.value_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, 1)

    def forward(self, x, x_mark=None, y_mark=None, **kwargs: Any):
        B, L, C = x.shape

        if self.use_norm:
            x_min = x.min(1, keepdim=True)[0].detach()
            x_max = x.max(1, keepdim=True)[0].detach()
            x_range = torch.clamp(x_max - x_min, min=1e-5).detach()
            x = (x - x_min) / x_range
            means = x_min
            stdev = x_range

        x = self.embedding(x).transpose(1, 2)

        dec_out = self.layers(x)

        temp_out = self.head(dec_out).permute(0, 2, 1)
        q = self.simple_q_proj(temp_out)

        dec_out_perm = dec_out.permute(0, 2, 1)
        k = self.key_proj(dec_out_perm)
        v = self.value_proj(dec_out_perm)

        attn = F.scaled_dot_product_attention(q, k, v)
        dec_out = self.out_proj(attn)

        if self.use_norm:
            dec_out = dec_out * stdev + means

        return dec_out

    def forecast(self, x, x_mark=None, y_mark=None, **kwargs):
        return self.forward(x, x_mark, y_mark, **kwargs)
