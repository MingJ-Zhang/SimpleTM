import torch
import torch.nn as nn


class SeasonalTrendBlock(nn.Module):
    """Separate a time series into seasonal and trend components using
    a moving average filter.

    Args:
        kernel_size (int): Size of the moving average window. Must be odd.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) // 2
        # AvgPool1d performs a simple moving average over the temporal dimension
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=pad, count_include_pad=False)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x (Tensor): Input sequence of shape [B, L, N].

        Returns:
            seasonal (Tensor): Seasonal component [B, L, N].
            trend (Tensor): Trend component [B, L, N].
        """
        # Rearrange to [B, N, L] for pooling along the temporal dimension
        x_t = x.transpose(1, 2)
        trend = self.avg(x_t)
        seasonal = x_t - trend
        # Restore original shape [B, L, N]
        return seasonal.transpose(1, 2), trend.transpose(1, 2)
