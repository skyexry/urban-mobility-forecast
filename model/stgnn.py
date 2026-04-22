# stgnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.stconv import STConvBlock, STGCNBlock, build_laplacian
from model.tcn import TCNBlock


class SelfAttentionDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Self-Attention based decoder.

        Args:
            d_model:   feature dimension
            num_heads: number of attention heads
            dropout:   dropout rate
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input of shape (batch*N, T, d_model)

        Returns:
            out: shape (batch*N, T, d_model)
        """
        # Self-attention + residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Conv1D + residual
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.dropout(conv_out))

        return x


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
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Spatio-Temporal GNN for bike-sharing demand forecasting.
        STConv Encoder + TCN Encoder + Feature Fusion + Self-Attention Decoder.

        Args:
            num_nodes:       number of stations N
            in_channels:     input feature dim (1 for demand)
            hidden_channels: STConv intermediate channels
            out_channels:    STConv output channels
            kernel_size:     temporal conv kernel size (paper: 3)
            K:               Chebyshev order (paper: 3)
            time_channels:   time encoding features (6 sin/cos signals)
            tcn_channels:    TCN output channels
            input_window:    input time steps (paper: 72)
            output_window:   output time steps (paper: 72)
            num_heads:       Self-Attention heads
            dropout:         dropout rate
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.input_window = input_window
        self.output_window = output_window

        # STConv block: demand graph encoder
        self.stconv = STGCNBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            K=K
        )

        # TCN block: time feature encoder
        self.tcn = TCNBlock(
            in_channels=time_channels,
            out_channels=tcn_channels,
            kernel_size=kernel_size,
            num_layers=4,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        # Reduced time after STConv: T - 2*(Kt-1)
        reduced_time = input_window - 4 * (kernel_size - 1)  # 72-8=64

        # Feature fusion: project to common d_model
        # STConv output: (batch, out_channels, N, reduced_T)
        # TCN output:    (batch, tcn_channels, input_T)
        # Fuse per time step: out_channels + tcn_channels → d_model
        d_model = out_channels + tcn_channels                 # 64+32=96
        self.fusion_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.fusion_norm = nn.LayerNorm(d_model)

        # Residual projection for demand features (paper: Add block)
        self.residual_proj = nn.Linear(out_channels, d_model)

        # Self-Attention decoder
        self.decoder = SelfAttentionDecoder(d_model, num_heads, dropout)

        # Output projection: d_model → output_window
        self.output_proj = nn.Conv1d(d_model, output_window, kernel_size=1)

    def forward(
        self,
        x_demand: torch.Tensor,
        x_time: torch.Tensor,
        L_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_demand: (batch, N, input_window, 1)
            x_time:   (batch, input_window, 6)
            L_hat:    (N, N) normalized Laplacian

        Returns:
            y_hat: (batch, N, output_window)
        """
        batch, N, T, F = x_demand.shape

        # ── Encoder: STConv branch ──────────────────────────────────────────
        x = x_demand.permute(0, 3, 1, 2)           # (batch, 1, N, T)
        stconv_out = self.stconv(x, L_hat)          # (batch, out_ch, N, reduced_T)
        stconv_out = self.dropout(stconv_out)
        # → (batch, N, reduced_T, out_ch)
        stconv_out = stconv_out.permute(0, 2, 3, 1)

        # ── Encoder: TCN branch ─────────────────────────────────────────────
        x_t = x_time.permute(0, 2, 1)              # (batch, 6, T)
        tcn_out = self.tcn(x_t)                     # (batch, tcn_ch, T)
        # Trim TCN output to match reduced_T
        reduced_T = stconv_out.shape[2]
        tcn_out = tcn_out[:, :, :reduced_T]         # (batch, tcn_ch, reduced_T)
        # → (batch, N, reduced_T, tcn_ch)
        tcn_out = tcn_out.unsqueeze(1).expand(-1, N, -1, -1).permute(0, 1, 3, 2)
        tcn_out = tcn_out.permute(0, 1, 2, 3)      # (batch, N, reduced_T, tcn_ch)

        # ── Feature Fusion ──────────────────────────────────────────────────
        # Concatenate STConv and TCN outputs
        fused = torch.cat([stconv_out, tcn_out], dim=-1)  # (batch, N, reduced_T, d_model)

        # Conv1D fusion + LayerNorm
        b, n, t, d = fused.shape
        fused = fused.reshape(b*n, t, d)
        fused = self.fusion_conv(fused.transpose(1,2)).transpose(1,2)  # (b*n, t, d)
        fused = self.fusion_norm(fused)

        # Residual connection with STConv output (Add block, per paper)
        stconv_flat = stconv_out.reshape(b*n, t, -1)      # (b*n, t, out_ch)
        residual = self.residual_proj(stconv_flat)          # (b*n, t, d_model)
        fused = fused + residual

        # ── Self-Attention Decoder ──────────────────────────────────────────
        decoded = self.decoder(fused)                       # (b*n, t, d_model)

        # ── Output projection ───────────────────────────────────────────────
        out = self.output_proj(decoded.transpose(1,2))      # (b*n, output_window, t)
        # Take mean over reduced_T dimension
        out = out.mean(dim=-1)                              # (b*n, output_window)
        out = out.reshape(batch, N, self.output_window)     # (batch, N, output_window)

        return out


if __name__ == "__main__":
    batch, N, T = 4, 20, 72

    x_demand = torch.randn(batch, N, T, 1)
    x_time   = torch.randn(batch, T, 6)
    W = torch.rand(N, N)
    W = (W + W.T) / 2
    L_hat = build_laplacian(W)

    model = STGNN(num_nodes=N)
    y_hat = model(x_demand, x_time, L_hat)
    print(f"x_demand: {x_demand.shape}")
    print(f"x_time:   {x_time.shape}")
    print(f"y_hat:    {y_hat.shape}")   # (4, 20, 72)