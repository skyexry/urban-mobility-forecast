# features.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_time_features(hours: pd.Series) -> np.ndarray:
    """
    Encode timestamps as cyclic sin/cos signals.
    Implements temporal feature construction from ST-BDP paper (Eq. 5-10).
    Six signals: day_sin, day_cos, week_sin, week_cos, year_sin, year_cos

    Args:
        hours: Series of datetime values (hourly granularity)

    Returns:
        time_features: array of shape (T, 6)
    """
    ts = hours.astype(np.int64) // 10**9  # convert to Unix timestamp (seconds)

    day_sin  = np.sin(ts * (2 * np.pi / (60 * 60 * 24)))
    day_cos  = np.cos(ts * (2 * np.pi / (60 * 60 * 24)))
    week_sin = np.sin(ts * (2 * np.pi / (60 * 60 * 24 * 7)))
    week_cos = np.cos(ts * (2 * np.pi / (60 * 60 * 24 * 7)))
    year_sin = np.sin(ts * (2 * np.pi / (60 * 60 * 24 * 365.2425)))
    year_cos = np.cos(ts * (2 * np.pi / (60 * 60 * 24 * 365.2425)))

    return np.stack([day_sin, day_cos, week_sin, week_cos, year_sin, year_cos], axis=1)  # (T, 6)


def build_demand_matrix(
    df: pd.DataFrame,
    station_ids: list
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Pivot long-format demand data into a (T, N) matrix.
    Missing values (station not active at that hour) are filled with 0.

    Args:
        df:          DataFrame with columns [start_station_id, hour, demand]
        station_ids: ordered list of N station IDs (must match adjacency matrix)

    Returns:
        demand_matrix: array of shape (T, N)
        hours:         DatetimeIndex of length T
    """
    pivot = (df.pivot_table(index='hour', columns='start_station_id',
                            values='demand', aggfunc='sum')
               .reindex(columns=station_ids)
               .fillna(0)
               .sort_index())

    return pivot.values.astype(np.float32), pivot.index  # (T, N)

def normalize_demand(
    demand_matrix: np.ndarray
) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize demand values using log1p transform + MinMax scaling to [-1, 1].
    log1p compresses the skewed distribution before scaling,
    improving prediction accuracy for low-demand time steps.

    Args:
        demand_matrix: array of shape (T, N)

    Returns:
        normalized:    array of shape (T, N)
        scaler:        fitted MinMaxScaler for inverse transform
    """
    T, N = demand_matrix.shape

    # log1p transform to compress skewed distribution
    # log1p(0) = 0, preserves zero demand
    log_demand = np.log1p(demand_matrix)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized = scaler.fit_transform(log_demand.reshape(-1, 1)).reshape(T, N)

    return normalized, scaler


def build_sliding_windows(
    demand_matrix: np.ndarray,
    time_features: np.ndarray,
    input_window: int = 72,
    output_window: int = 72
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice demand matrix and time features into sliding window samples.
    Adapted from ST-BDP paper DataProcessing.py (single-station → multi-station).

    Args:
        demand_matrix: normalized demand array of shape (T, N)
        time_features: time encoding array of shape (T, 6)
        input_window:  number of input hours (default 72, per paper)
        output_window: number of output hours to predict (default 72, per paper)

    Returns:
        x_demand: array of shape (samples, N, input_window, 1)
        x_time:   array of shape (samples, input_window, 6)
        y:        array of shape (samples, N, output_window, 1)
    """
    T, N = demand_matrix.shape
    total_window = input_window + output_window

    x_demand, x_time, y = [], [], []

    # Slide window by 1 hour at a time (same as paper)
    for i in range(total_window, T + 1):
        x_start = i - total_window
        x_end   = i - output_window
        y_end   = i

        x_demand.append(demand_matrix[x_start:x_end, :])   # (input_window, N)
        x_time.append(time_features[x_start:x_end, :])     # (input_window, 6)
        y.append(demand_matrix[x_end:y_end, :])             # (output_window, N)

    x_demand = np.array(x_demand)                          # (samples, input_window, N)
    x_time   = np.array(x_time)                            # (samples, input_window, 6)
    y        = np.array(y)                                  # (samples, output_window, N)

    # Reshape to (samples, N, window, 1) to match PyG Temporal STConv input
    x_demand = x_demand.transpose(0, 2, 1)[:, :, :, np.newaxis]  # (samples, N, input_window, 1)
    y        = y.transpose(0, 2, 1)[:, :, :, np.newaxis]         # (samples, N, output_window, 1)

    print(f"Samples  : {len(x_demand)}")
    print(f"x_demand : {x_demand.shape}")
    print(f"x_time   : {x_time.shape}")
    print(f"y        : {y.shape}")

    return x_demand, x_time, y

def inverse_transform_demand(
    normalized: np.ndarray,
    scaler: MinMaxScaler
) -> np.ndarray:
    """
    Inverse transform normalized predictions back to original demand scale.
    Reverses log1p + MinMax normalization.

    Args:
        normalized: array of any shape
        scaler:     fitted MinMaxScaler from normalize_demand

    Returns:
        demand: array in original scale
    """
    shape = normalized.shape
    # Reverse MinMax scaling
    log_demand = scaler.inverse_transform(normalized.reshape(-1, 1)).reshape(shape)
    # Reverse log1p
    return np.expm1(log_demand)