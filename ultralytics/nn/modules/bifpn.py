import torch
import torch.nn as nn

from .conv import Conv


class BiFPN(nn.Module):
    """Lightweight BiFPN-style weighted fusion for multi-scale feature inputs."""

    def __init__(self, in_channels, out_channels, eps=1e-4):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) < 2:
            raise ValueError("BiFPN expects at least two input feature maps.")

        self.eps = eps
        self.align = nn.ModuleList(
            nn.Identity() if c == out_channels else Conv(c, out_channels, k=1, s=1)
            for c in in_channels
        )
        self.w = nn.Parameter(torch.ones(len(in_channels), dtype=torch.float32))
        # self.act = nn.ReLU(inplace=False)
        self.out_conv = Conv(out_channels, out_channels, k=3, s=1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != len(self.align):
            raise ValueError(f"BiFPN expects {len(self.align)} inputs, but got {type(x)} with length {len(x) if isinstance(x, (list, tuple)) else 'N/A'}.")

        # Normalize non-negative fusion weights so the feature sum is scale-stable.
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)

        y = 0
        for i, xi in enumerate(x):
            y = y + w[i] * self.align[i](xi)
        return self.out_conv(y)

