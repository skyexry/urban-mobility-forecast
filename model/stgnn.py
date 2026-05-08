import torch
import torch.nn as nn
import torch.nn.functional as F
from model.stconv import STGCNBlock, build_edge_index, build_laplacian
from model.tcn import TCNBlock


class SelfAttentionDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Self-Attention based decoder.
        Implements Eq.(17)(18) from ST-BDP paper.

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
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
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

        # Two Conv1D + residual (per paper Figure 9 decoder)
        residual = x
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + residual)

        return x


class TransformerDecoderHead(nn.Module):
    def __init__(self, d_model: int, output_window: int, num_heads: int = 4,
                 dropout: float = 0.1, num_layers: int = 2):
        """
        Transformer decoder: learned queries (one per output step) attend to
        encoder memory via cross-attention, replacing the Conv1d mean-pool output.
        """
        super().__init__()
        self.query_embed = nn.Embedding(output_window, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        # memory: (b*n, reduced_T, d_model)
        b_n = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(b_n, -1, -1)
        decoded = self.decoder(queries, memory)          # (b*n, output_window, d_model)
        return self.out_proj(decoded).squeeze(-1)        # (b*n, output_window)


class STGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 1,
        hidden_channels: int = 16,
        out_channels: int = 64,
        kernel_size: int = 3,
        K: int = 3,
        time_channels: int = 6,
        tcn_channels: int = 64,
        input_window: int = 72,
        output_window: int = 72,
        num_heads: int = 4,
        dropout: float = 0.2,
        weather_channels: int = 0,
        use_transformer_decoder: bool = False
    ):
        """
        Spatio-Temporal GNN for bike-sharing demand forecasting.
        Implements ST-BDP architecture (Zhou et al., 2024):
        STGCNBlock Encoder + TCN Encoder + Feature Fusion + Self-Attention Decoder.
        Uses PyG ChebConv for efficient sparse graph convolution.

        Args:
            num_nodes:       number of stations N
            in_channels:     input feature dim (1 for demand)
            hidden_channels: STConv intermediate channels (paper: 16)
            out_channels:    STConv output channels (paper: 64)
            kernel_size:     temporal conv kernel size (paper: 3)
            K:               Chebyshev order (paper: 3)
            time_channels:   time encoding features (6 sin/cos signals)
            tcn_channels:    TCN output channels (paper: 64)
            input_window:    input time steps (paper: 72)
            output_window:   output time steps (paper: 72)
            num_heads:       Self-Attention heads
            dropout:         dropout rate
            weather_channels: weather feature dim (0 = no weather, 9 = full weather branch)
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.input_window = input_window
        self.output_window = output_window
        self.weather_channels = weather_channels
        self.use_transformer_decoder = use_transformer_decoder

        # STGCNBlock: two stacked ST-Conv blocks for demand graph encoding
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

        # Optional weather TCN branch (matches paper's 4th input branch)
        if weather_channels > 0:
            self.weather_tcn = TCNBlock(
                in_channels=weather_channels,
                out_channels=tcn_channels,
                kernel_size=kernel_size,
                num_layers=4,
                dropout=dropout
            )

        self.dropout = nn.Dropout(dropout)

        # Time dim after two STConv blocks: T - 4*(Kt-1)
        reduced_time = input_window - 4 * (kernel_size - 1)  # 72-8=64

        # d_model: 128 without weather, 192 with weather branch
        n_branches = 3 if weather_channels > 0 else 2
        d_model = out_channels + tcn_channels * (n_branches - 1)

        self.fusion_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.fusion_norm = nn.LayerNorm(d_model)

        # Residual projection for STConv output (Add block per paper)
        self.residual_proj = nn.Linear(out_channels, d_model)

        # Self-Attention decoder
        self.decoder = SelfAttentionDecoder(d_model, num_heads, dropout)

        # Output head: Transformer decoder (cross-attention) or Conv1d mean-pool
        if use_transformer_decoder:
            self.output_head = TransformerDecoderHead(
                d_model, output_window, num_heads, dropout, num_layers=2)
        else:
            self.output_proj = nn.Conv1d(d_model, output_window, kernel_size=1)

    def forward(
        self,
        x_demand: torch.Tensor,
        x_time: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        x_weather: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x_demand:    (batch, N, input_window, 1)
            x_time:      (batch, input_window, 6)
            edge_index:  (2, E) graph connectivity
            edge_weight: (E,) edge weights
            x_weather:   (batch, input_window, 9) — optional weather features

        Returns:
            y_hat: (batch, N, output_window)
        """
        batch, N, T, F = x_demand.shape

        # ── Encoder: STConv branch ──────────────────────────────────────────
        x = x_demand.permute(0, 3, 1, 2)                    # (batch, 1, N, T)
        stconv_out = self.stconv(x, edge_index, edge_weight) # (batch, out_ch, N, reduced_T)
        stconv_out = self.dropout(stconv_out)
        stconv_out = stconv_out.permute(0, 2, 3, 1)         # (batch, N, reduced_T, out_ch)

        # ── Encoder: TCN branch ─────────────────────────────────────────────
        x_t = x_time.permute(0, 2, 1)                       # (batch, 6, T)
        tcn_out = self.tcn(x_t)                              # (batch, tcn_ch, T)

        # Trim TCN to match reduced_T
        reduced_T = stconv_out.shape[2]
        tcn_out = tcn_out[:, :, :reduced_T]                  # (batch, tcn_ch, reduced_T)
        tcn_out = tcn_out.unsqueeze(1).expand(-1, N, -1, -1) # (batch, N, tcn_ch, reduced_T)
        tcn_out = tcn_out.permute(0, 1, 3, 2)               # (batch, N, reduced_T, tcn_ch)

        # ── Encoder: Weather TCN branch (optional) ──────────────────────────
        branches = [stconv_out, tcn_out]
        if self.weather_channels > 0 and x_weather is not None:
            x_w = x_weather.permute(0, 2, 1)                        # (batch, 9, T)
            weather_out = self.weather_tcn(x_w)                      # (batch, tcn_ch, T)
            weather_out = weather_out[:, :, :reduced_T]              # (batch, tcn_ch, reduced_T)
            weather_out = weather_out.unsqueeze(1).expand(-1, N, -1, -1)
            weather_out = weather_out.permute(0, 1, 3, 2)           # (batch, N, reduced_T, tcn_ch)
            branches.append(weather_out)

        # ── Feature Fusion ──────────────────────────────────────────────────
        fused = torch.cat(branches, dim=-1)                  # (batch, N, reduced_T, d_model)

        b, n, t, d = fused.shape
        fused = fused.reshape(b * n, t, d)
        fused = self.fusion_conv(fused.transpose(1, 2)).transpose(1, 2)
        fused = self.fusion_norm(fused)

        # Residual connection (Add block per paper)
        stconv_flat = stconv_out.reshape(b * n, t, -1)
        residual = self.residual_proj(stconv_flat)
        fused = fused + residual

        # ── Self-Attention Decoder ──────────────────────────────────────────
        decoded = self.decoder(fused)                         # (b*n, t, d_model)

        # ── Output head ─────────────────────────────────────────────────────
        if self.use_transformer_decoder:
            out = self.output_head(decoded)                   # (b*n, output_window)
        else:
            out = self.output_proj(decoded.transpose(1, 2))   # (b*n, output_window, t)
            out = out.mean(dim=-1)                            # (b*n, output_window)
        out = out.reshape(batch, N, self.output_window)       # (batch, N, output_window)

        return out


if __name__ == "__main__":
    batch, N, T = 4, 20, 72

    x_demand = torch.randn(batch, N, T, 1)
    x_time   = torch.randn(batch, T, 6)
    W = torch.rand(N, N)
    W = (W + W.T) / 2
    W_tensor = torch.FloatTensor(W)
    edge_index, edge_weight = build_edge_index(W_tensor)

    model = STGNN(num_nodes=N)
    y_hat = model(x_demand, x_time, edge_index, edge_weight)
    print(f"x_demand: {x_demand.shape}")
    print(f"x_time:   {x_time.shape}")
    print(f"y_hat:    {y_hat.shape}")  # (4, 20, 72)