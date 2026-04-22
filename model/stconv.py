import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int):
        """
        Chebyshev spectral graph convolution.
        Approximates graph convolution using K-order Chebyshev polynomials.
        Ref: ST-BDP paper Eq.(12), Graph-Conv layer.

        Args:
            in_channels:  input feature dimension per node
            out_channels: output feature dimension per node
            K:            Chebyshev polynomial order (paper uses 3)
        """
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, L_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     node features of shape (batch, N, in_channels)
            L_hat: normalized Laplacian of shape (N, N)

        Returns:
            out: shape (batch, N, out_channels)
        """
        # Chebyshev recurrence: T_0=x, T_1=L_hat*x, T_k=2*L_hat*T_{k-1} - T_{k-2}
        Tx_0 = x                                    # (batch, N, F)
        out = torch.matmul(Tx_0, self.weight[0])    # (batch, N, out)

        if self.K > 1:
            Tx_1 = torch.matmul(L_hat, x)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.K):
            Tx_2 = 2 * torch.matmul(L_hat, Tx_1) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out + self.bias


class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        Temporal gated convolution (GLU activation).
        Implements the Temporal Gated-Conv layer in ST-Conv block.
        Ref: ST-BDP paper Figure 11C.

        Args:
            in_channels:  input channels
            out_channels: output channels
            kernel_size:  temporal kernel size Kt (paper uses 3)
        """
        super().__init__()
        # Output 2*out_channels for GLU gating
        self.conv = nn.Conv2d(
            in_channels, 2 * out_channels,
            kernel_size=(1, kernel_size)   # (node dim, time dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input of shape (batch, in_channels, N, T)

        Returns:
            out: shape (batch, out_channels, N, T - kernel_size + 1)
        """
        out = self.conv(x)                          # (batch, 2*out, N, T-Kt+1)
        p, q = out.chunk(2, dim=1)                  # each (batch, out, N, T-Kt+1)
        return p * torch.sigmoid(q)                 # GLU gating


class STConvBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, kernel_size: int, K: int):
        """
        Spatio-Temporal Convolutional Block.
        Structure: Temporal Gated Conv → Graph Conv → Temporal Gated Conv.
        Ref: ST-BDP paper Figure 11B, Eq.(12).

        Args:
            in_channels:     input feature channels
            hidden_channels: intermediate channels after first temporal conv
            out_channels:    output channels
            kernel_size:     temporal kernel size Kt (paper uses 3)
            K:               Chebyshev order (paper uses 3)
        """
        super().__init__()
        self.tgconv1 = TemporalGatedConv(in_channels, hidden_channels, kernel_size)
        self.graph_conv = ChebConv(hidden_channels, hidden_channels, K)
        self.tgconv2 = TemporalGatedConv(hidden_channels, out_channels, kernel_size)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, L_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     input of shape (batch, in_channels, N, T)
            L_hat: normalized Laplacian of shape (N, N)

        Returns:
            out: shape (batch, out_channels, N, T - 2*(kernel_size-1))
        """
        # First temporal conv: (batch, hidden, N, T-Kt+1)
        out = self.tgconv1(x)

        # Graph conv: reshape for matmul, then back
        batch, C, N, T = out.shape
        out = out.permute(0, 3, 2, 1)               # (batch, T, N, C)
        out = self.graph_conv(out.reshape(-1, N, C), L_hat)  # (batch*T, N, C)
        out = out.reshape(batch, T, N, C).permute(0, 3, 2, 1)  # (batch, C, N, T)
        out = F.relu(out)

        # Second temporal conv: (batch, out_channels, N, T-2*(Kt-1))
        out = self.tgconv2(out)

        # Layer norm over channel dim
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out


def build_laplacian(W: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized graph Laplacian from adjacency matrix.
    L_hat = I - D^{-1/2} A D^{-1/2}, scaled to [-1, 1].

    Args:
        W: adjacency matrix of shape (N, N)

    Returns:
        L_hat: normalized Laplacian of shape (N, N)
    """
    N = W.shape[0]
    D = W.sum(dim=1)                                # degree vector
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
    I = torch.eye(N, device=W.device)
    L = I - D_inv_sqrt @ W @ D_inv_sqrt             # normalized Laplacian
    # Scale to [-1, 1] for Chebyshev approximation
    lambda_max = torch.linalg.eigvalsh(L).max()
    L_hat = (2 * L / lambda_max) - I
    return L_hat


if __name__ == "__main__":
    # Sanity check
    batch, N, T = 4, 10, 72
    in_ch, hidden_ch, out_ch = 1, 16, 32
    Kt, K = 3, 3

    x = torch.randn(batch, in_ch, N, T)
    W = torch.rand(N, N)
    W = (W + W.T) / 2                               # symmetric
    L_hat = build_laplacian(W)

    block = STConvBlock(in_ch, hidden_ch, out_ch, Kt, K)
    out = block(x, L_hat)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # (4, 32, 10, 68)