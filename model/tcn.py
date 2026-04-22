# tcn.py
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        """
        TCN residual block with dilated causal convolution.
        Adapted from ST-BDP paper Experiment_01/TCN_15/tcn_15.py (TF → PyTorch).

        Args:
            in_channels:  number of input channels
            out_channels: number of output channels
            kernel_size:  convolution kernel size
            dilation:     dilation rate (doubles each block: 1, 2, 4, 8...)
            dropout:      dropout rate
        """
        super().__init__()

        # Padding to maintain sequence length with causal convolution
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels,
                                           kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels,
                                           kernel_size, padding=padding, dilation=dilation))

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Skip connection: 1x1 conv if channel dimensions differ
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.final_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, channels, time)

        Returns:
            out: tensor of shape (batch, out_channels, time)
        """
        # Causal conv: trim the extra padding on the right
        pad = (self.conv1.weight.shape[-1] - 1) * self.conv1.dilation[0]

        out = self.conv1(x)[:, :, :-pad] if pad > 0 else self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout1(out)

        out = self.conv2(out)[:, :, :-pad] if pad > 0 else self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout2(out)

        # Residual connection
        skip = self.skip(x) if self.skip is not None else x
        return self.final_activation(out + skip)


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 num_layers: int = 4, dropout: float = 0.2):
        """
        Stacked TCN residual blocks with exponentially increasing dilation rates.
        Mirrors TCN class in ST-BDP paper tcn_15.py.

        Args:
            in_channels:  input feature dimension
            out_channels: output feature dimension (same for all layers)
            kernel_size:  convolution kernel size (paper uses 3)
            num_layers:   number of residual blocks (paper uses 4)
            dropout:      dropout rate
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i           # dilation: 1, 2, 4, 8
            in_ch = in_channels if i == 0 else out_channels
            layers.append(ResidualBlock(in_ch, out_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input of shape (batch, in_channels, time)

        Returns:
            out: shape (batch, out_channels, time)
        """
        return self.network(x)


if __name__ == "__main__":
    # Quick sanity check
    batch, time, features = 16, 72, 6
    x = torch.randn(batch, features, time)  # PyTorch: (batch, channels, time)

    tcn = TCNBlock(in_channels=features, out_channels=64)
    out = tcn(x)
    print(f"Input:  {x.shape}")   # (16, 6, 72)
    print(f"Output: {out.shape}") # (16, 64, 72)