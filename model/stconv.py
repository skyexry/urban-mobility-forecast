import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, Batch


class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        Temporal gated convolution (GLU activation).
        Ref: ST-BDP paper Figure 11C.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 2 * out_channels,
            kernel_size=(1, kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        p, q = out.chunk(2, dim=1)
        return p * torch.sigmoid(q)


class STConvBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, kernel_size: int, K: int):
        """
        Spatio-Temporal Convolutional Block.
        Structure: Temporal Gated Conv → PyG ChebConv → Temporal Gated Conv.
        Uses PyG batch processing for efficient graph convolution.
        Ref: ST-BDP paper Figure 11B.
        """
        super().__init__()
        self.tgconv1 = TemporalGatedConv(in_channels, hidden_channels, kernel_size)
        self.graph_conv = ChebConv(hidden_channels, hidden_channels, K=K)
        self.tgconv2 = TemporalGatedConv(hidden_channels, out_channels, kernel_size)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (batch, in_channels, N, T)
            edge_index:  (2, E)
            edge_weight: (E,)

        Returns:
            out: (batch, out_channels, N, T - 2*(kernel_size-1))
        """
        # First temporal conv
        out = self.tgconv1(x)                            # (batch, hidden, N, T')

        batch_size, C, N, T = out.shape

        # Reshape for graph conv: (batch*T, N, C)
        out = out.permute(0, 3, 2, 1).reshape(batch_size * T, N, C)

        # Build PyG batch: repeat edge_index for each graph in batch*T
        # This allows processing all graphs in parallel
        batch_size_total = batch_size * T
        offset = torch.arange(batch_size_total, device=x.device).unsqueeze(1) * N
        edge_index_batch = (edge_index.unsqueeze(0) + offset.unsqueeze(-1))  # (batch*T, 2, E)
        edge_index_batch = edge_index_batch.view(2, -1)                       # (2, batch*T*E)
        edge_weight_batch = edge_weight.unsqueeze(0).expand(batch_size_total, -1).reshape(-1)

        # Flatten all nodes: (batch*T*N, C)
        out_flat = out.reshape(batch_size_total * N, C)

        # Single PyG ChebConv call for all graphs
        out_flat = self.graph_conv(out_flat, edge_index_batch, edge_weight_batch)

        # Reshape back: (batch, C, N, T')
        out = out_flat.reshape(batch_size, T, N, C).permute(0, 3, 2, 1)
        out = F.relu(out)

        # Second temporal conv
        out = self.tgconv2(out)

        # Layer norm
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, kernel_size: int, K: int):
        """
        Full STGCN: two stacked ST-Conv blocks + output conv.
        Ref: ST-BDP paper Figure 11(a).
        """
        super().__init__()
        self.stconv1 = STConvBlock(in_channels, hidden_channels, out_channels, kernel_size, K)
        self.stconv2 = STConvBlock(out_channels, hidden_channels, out_channels, kernel_size, K)
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        out = self.stconv1(x, edge_index, edge_weight)
        out = self.stconv2(out, edge_index, edge_weight)
        out = self.output_conv(out)
        return out


def build_edge_index(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dense adjacency matrix to PyG sparse edge_index format.

    Args:
        W: adjacency matrix of shape (N, N)

    Returns:
        edge_index:  (2, E) LongTensor
        edge_weight: (E,) FloatTensor
    """
    src, dst = torch.nonzero(W, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).long()
    edge_weight = W[src, dst].float()
    return edge_index, edge_weight


def build_laplacian(W: torch.Tensor) -> torch.Tensor:
    """Keep for backward compatibility with old code."""
    N = W.shape[0]
    D = W.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
    I = torch.eye(N, device=W.device)
    L = I - D_inv_sqrt @ W @ D_inv_sqrt
    lambda_max = torch.linalg.eigvalsh(L).max()
    L_hat = (2 * L / lambda_max) - I
    return L_hat