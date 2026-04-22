# stgnn.py
import torch
import torch.nn as nn
from model.stconv import STConvBlock, build_laplacian
from model.tcn import TCNBlock


class STGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
        K: int = 3,
        time_channels: int = 6,
        tcn_channels: int = 32,
        input_window: int = 72,
        output_window: int = 72,
        dropout: float = 0.2
    ):
        """
        Spatio-Temporal GNN for bike-sharing demand forecasting.
        Version 2: STConv (pure PyTorch) + TCN (time features) + Linear output.
        Based on ST-BDP paper (Zhou et al., 2024).

        Args:
            num_nodes:       number of stations (N)
            in_channels:     input feature dimension (1 for demand)
            hidden_channels: STConv intermediate channels
            out_channels:    STConv output channels
            kernel_size:     temporal conv kernel size (paper uses 3)
            K:               Chebyshev order (paper uses 3)
            time_channels:   number of time encoding features (6 sin/cos signals)
            tcn_channels:    TCN output channels
            input_window:    input time steps (default 72)
            output_window:   output time steps to predict (default 72)
            dropout:         dropout rate
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.input_window = input_window
        self.output_window = output_window

        # STConv block: processes demand graph (spatial + temporal)
        self.stconv = STConvBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            K=K
        )

        # TCN block: processes time encoding features
        self.tcn = TCNBlock(
            in_channels=time_channels,
            out_channels=tcn_channels,
            kernel_size=kernel_size,
            num_layers=4,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        # Reduced time after STConv
        reduced_time = input_window - 2 * (kernel_size - 1)  # 72 - 4 = 68

        # Feature fusion: concatenate STConv and TCN outputs, project
        stconv_flat = out_channels * reduced_time             # 64 * 68 = 4352
        tcn_flat = tcn_channels * input_window                # 32 * 72 = 2304
        fusion_dim = stconv_flat + tcn_flat                   # 6656

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_window)
        )

    def forward(
        self,
        x_demand: torch.Tensor,
        x_time: torch.Tensor,
        L_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_demand: demand tensor of shape (batch, N, input_window, 1)
            x_time:   time features of shape (batch, input_window, 6)
            L_hat:    normalized Laplacian of shape (N, N)

        Returns:
            y_hat: predicted demand of shape (batch, N, output_window)
        """
        batch, N, T, F = x_demand.shape

        # STConv: (batch, N, T, 1) → (batch, out_channels, N, reduced_T)
        x = x_demand.permute(0, 3, 1, 2)           # (batch, 1, N, T)
        stconv_out = self.stconv(x, L_hat)          # (batch, out_ch, N, reduced_T)
        stconv_out = self.dropout(stconv_out)
        stconv_flat = stconv_out.permute(0, 2, 1, 3).reshape(batch, N, -1)  # (batch, N, out_ch*reduced_T)

        # TCN: (batch, T, 6) → (batch, tcn_channels, T)
        x_t = x_time.permute(0, 2, 1)              # (batch, 6, T)
        tcn_out = self.tcn(x_t)                     # (batch, tcn_ch, T)
        tcn_flat = tcn_out.reshape(batch, -1)       # (batch, tcn_ch*T)

        # Expand TCN output to match N nodes
        tcn_flat = tcn_flat.unsqueeze(1).expand(-1, N, -1)  # (batch, N, tcn_ch*T)

        # Fusion: concatenate and project
        fused = torch.cat([stconv_flat, tcn_flat], dim=-1)   # (batch, N, fusion_dim)
        y_hat = self.fusion(fused)                            # (batch, N, output_window)

        return y_hat


if __name__ == "__main__":
    # Sanity check
    batch, N, T = 4, 20, 72

    x_demand = torch.randn(batch, N, T, 1)
    x_time = torch.randn(batch, T, 6)
    W = torch.rand(N, N)
    W = (W + W.T) / 2
    L_hat = build_laplacian(W)

    model = STGNN(num_nodes=N)
    y_hat = model(x_demand, x_time, L_hat)
    print(f"x_demand: {x_demand.shape}")
    print(f"x_time:   {x_time.shape}")
    print(f"y_hat:    {y_hat.shape}")   # (4, 20, 72)