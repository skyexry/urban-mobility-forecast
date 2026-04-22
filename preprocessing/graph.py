# graph.py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def build_adjacency_matrix(
    df: pd.DataFrame,
    sigma2: float = 0.1,
    theta: float = 0.5
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build adjacency matrix using Gaussian kernel on inter-station distances.
    Implements Eq.(1) from ST-BDP paper:
        w_ij = exp(-d_ij^2 / sigma^2)  if i != j and value >= theta
        w_ij = 0                        otherwise

    Args:
        df:     DataFrame with columns [start_station_id, start_lat, start_lng]
        sigma2: variance of the Gaussian kernel (controls distance sensitivity)
        theta:  sparsity threshold (edges below this value are set to 0)

    Returns:
        W:        adjacency matrix of shape (N, N)
        stations: DataFrame of station metadata, index-aligned with W
    """
    # Get unique stations with mean coordinates
    stations = (df.groupby('start_station_id')[['start_lat', 'start_lng']]
                .mean()
                .reset_index()
                .reset_index(drop=True))

    coords = stations[['start_lat', 'start_lng']].values  # (N, 2)

    # Compute pairwise Euclidean distances between station coordinates
    dist_matrix = cdist(coords, coords, metric='euclidean')  # (N, N)

    # Apply Gaussian kernel: w_ij = exp(-d^2 / sigma^2)
    W = np.exp(-dist_matrix ** 2 / sigma2)

    # Apply sparsity threshold and zero out self-loops
    W[W < theta] = 0
    np.fill_diagonal(W, 0)

    print(f"Stations     : {len(stations)}")
    print(f"Non-zero edges: {np.count_nonzero(W)}")
    print(f"Sparsity     : {1 - np.count_nonzero(W) / (len(stations) ** 2):.2%}")

    return W, stations


def adjacency_to_edge_index(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert dense adjacency matrix to PyG-compatible edge_index and edge_weight.

    Args:
        W: adjacency matrix of shape (N, N)

    Returns:
        edge_index:  LongTensor of shape (2, E) — source and target node indices
        edge_weight: FloatTensor of shape (E,)  — edge weights
    """
    src, dst = np.nonzero(W)
    edge_index = np.stack([src, dst], axis=0)   # (2, E)
    edge_weight = W[src, dst]                    # (E,)

    return edge_index, edge_weight