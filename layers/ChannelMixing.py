import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedChannelMixing(nn.Module):
    """Apply channel-wise attention to adaptively reweight features.

    This block first summarises each channel via global average pooling
    and then uses a small gating network to produce channel-wise gates.
    The input is rescaled by these gates, enabling dynamic feature fusion.
    """

    def __init__(self, channels: int, reduction: int = 16, dropout: float = 0.0):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x (Tensor): Input sequence [B, L, N].
        Returns:
            Tensor: Rescaled sequence of the same shape.
        """
        # Global average pooling over the temporal dimension: [B, N]
        w = x.mean(dim=1)
        w = self.fc1(w)
        w = F.relu(w)
        w = self.dropout(w)
        w = self.fc2(w)
        w = self.sigmoid(w).unsqueeze(1)  # [B, 1, N]
        return x * w
