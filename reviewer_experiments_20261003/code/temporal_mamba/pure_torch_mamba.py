"""Pure PyTorch Mamba block with an explicit selective scan.

This implementation intentionally uses no mamba_ssm, Triton, causal-conv1d,
or custom CUDA extension.  Sequence length is only nine for the gaze task, so
the transparent recurrent scan is preferable to a fused kernel here.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PureTorchMamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        params = self.x_proj(x)
        dt_raw, B, C = torch.split(
            params, (self.dt_rank, self.d_state, self.d_state), dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log.float()).to(dtype=x.dtype)
        state = x.new_zeros(batch, self.d_inner, self.d_state)
        outputs = []
        for step in range(length):
            dt = delta[:, step]
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt.unsqueeze(-1) * B[:, step].unsqueeze(1)
            state = dA * state + dB * x[:, step].unsqueeze(-1)
            y = (state * C[:, step].unsqueeze(1)).sum(dim=-1)
            outputs.append(y + self.D * x[:, step])
        return torch.stack(outputs, dim=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x, gate = self.in_proj(hidden).chunk(2, dim=-1)
        length = x.shape[1]
        x = self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = F.silu(x)
        y = self.selective_scan(x)
        return self.out_proj(y * F.silu(gate))


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = PureTorchMamba(d_model=d_model, d_state=d_state)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.dropout(self.mixer(self.norm(hidden)))

