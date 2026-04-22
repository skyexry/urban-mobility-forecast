# stgnn.py
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.attention import STConv


class STGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        K: int,
        input_window: int = 72,
        output_window: int = 72,
        dropout: float = 0.2
    ):
        """
        Spatio-Temporal GNN for bike-sharing demand forecasting.
        Version 1: STConv encoder + Linear output (simplified).
        Based on ST-BDP paper architecture (Zhou et al., 2024).

        Args:
            num_nodes:      number of stations (N)
            in_channels:    input feature dimension (1 for demand only)
            hidden_channels: intermediate feature dimension
            out_channels:   STConv output feature dimension
            kernel_size:    temporal convolution kernel size (paper uses 3)
            K:              Chebyshev polynomial order (paper uses 3)
            input_window:   number of input time steps (default 72)
            output_window:  number of output time steps to predict (default 72)
            dropout:        dropout rate
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.input_window = input_window
        self.output_window = output_window

        # STConv block: extracts spatio-temporal features from demand graph
        self.stconv = STConv(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            K=K
        )

        self.dropout = nn.Dropout(dropout)

        # Output layer: project from STConv output to prediction window
        # STConv reduces time dim by 2*(kernel_size-1), so we account for that
        reduced_time = input_window - 2 * (kernel_size - 1)
        self.output_proj = nn.Linear(out_channels * reduced_time, output_window)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:           demand tensor of shape (batch, N, input_window, 1)
            edge_index:  graph connectivity of shape (2, E)
            edge_weight: edge weights of shape (E,)

        Returns:
            y_hat: predicted demand of shape (batch, N, output_window)
        """
        batch, N, T, F = x.shape

        # STConv expects (batch, N, F, T)
        x = x.permute(0, 1, 3, 2)  # (batch, N, 1, T)

        # Apply STConv: output is (batch, N, out_channels, reduced_T)
        out = self.stconv(x, edge_index, edge_weight)

        out = self.dropout(out)

        # Flatten time and channel dims for linear projection
        out = out.reshape(batch, N, -1)  # (batch, N, out_channels * reduced_T)

        # Project to output window
        y_hat = self.output_proj(out)   # (batch, N, output_window)

        return y_hat


if __name__ == "__main__":
    # Sanity check
    batch, N, T = 4, 438, 72

    x = torch.randn(batch, N, T, 1)
    edge_index = torch.randint(0, N, (2, 1000))
    edge_weight = torch.rand(1000)

    model = STGNN(
        num_nodes=N,
        in_channels=1,
        hidden_channels=32,
        out_channels=64,
        kernel_size=3,
        K=3
    )

    y_hat = model(x, edge_index, edge_weight)
    print(f"Input:  {x.shape}")
    print(f"Output: {y_hat.shape}")  # (4, 438, 72)