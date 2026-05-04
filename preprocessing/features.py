# features.py
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
    output_window: int = 72,
    weather_features: np.ndarray = None
) -> tuple:
    """
    Slice demand matrix and time features into sliding window samples.
    Adapted from ST-BDP paper DataProcessing.py (single-station → multi-station).

    Args:
        demand_matrix:    normalized demand array of shape (T, N)
        time_features:    time encoding array of shape (T, 6)
        input_window:     number of input hours (default 72, per paper)
        output_window:    number of output hours to predict (default 72, per paper)
        weather_features: optional normalized weather array of shape (T, 9)

    Returns:
        x_demand: array of shape (samples, N, input_window, 1)
        x_time:   array of shape (samples, input_window, 6)
        x_weather (only if weather_features is not None): (samples, input_window, 9)
        y:        array of shape (samples, N, output_window, 1)
    """
    T, N = demand_matrix.shape
    total_window = input_window + output_window

    x_demand, x_time, x_weather_list, y = [], [], [], []

    for i in range(total_window, T + 1):
        x_start = i - total_window
        x_end   = i - output_window
        y_end   = i

        x_demand.append(demand_matrix[x_start:x_end, :])
        x_time.append(time_features[x_start:x_end, :])
        y.append(demand_matrix[x_end:y_end, :])
        if weather_features is not None:
            x_weather_list.append(weather_features[x_start:x_end, :])

    x_demand = np.array(x_demand)
    x_time   = np.array(x_time)
    y        = np.array(y)

    x_demand = x_demand.transpose(0, 2, 1)[:, :, :, np.newaxis]  # (samples, N, input_window, 1)
    y        = y.transpose(0, 2, 1)[:, :, :, np.newaxis]         # (samples, N, output_window, 1)

    print(f"Samples  : {len(x_demand)}")
    print(f"x_demand : {x_demand.shape}")
    print(f"x_time   : {x_time.shape}")
    print(f"y        : {y.shape}")

    if weather_features is not None:
        x_weather = np.array(x_weather_list)                     # (samples, input_window, 9)
        print(f"x_weather: {x_weather.shape}")
        return x_demand, x_time, x_weather, y

    return x_demand, x_time, y

def fetch_weather_data(
    start_date: str,
    end_date: str,
    lat: float = 40.71,
    lon: float = -74.01
) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo archive API for NYC (no API key required).

    Args:
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'

    Returns:
        DataFrame with columns: hour, temperature_2m, dew_point_2m,
        relative_humidity_2m, surface_pressure, wind_speed_10m,
        wind_direction_10m, wind_gusts_10m, precipitation
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,"
        "surface_pressure,wind_speed_10m,wind_direction_10m,"
        "wind_gusts_10m,precipitation"
        "&timezone=America%2FNew_York"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()['hourly']
    df = pd.DataFrame(data)
    df['hour'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    return df.drop(columns='time')


def build_weather_features(
    weather_df: pd.DataFrame,
    hours: pd.DatetimeIndex
) -> np.ndarray:
    """
    Align weather to demand timestamps and decompose wind into (u, v) components.
    Matches ST-BDP paper input structure: Wind_X, Wind_Y, Wind_Gust_X, Wind_Gust_Y.

    Returns raw (T, 9) array — call normalize_weather_features() before use.

    Feature order (matches paper):
        [0] temperature_2m          (z-score)
        [1] dew_point_2m            (z-score)
        [2] relative_humidity_2m    (z-score)
        [3] surface_pressure        (z-score)
        [4] wind_u  = speed*sin(dir)(z-score)
        [5] wind_v  = speed*cos(dir)(z-score)
        [6] gust_u  = gust*sin(dir) (z-score)
        [7] gust_v  = gust*cos(dir) (z-score)
        [8] precipitation           (raw, no normalization per paper)
    """
    w = weather_df.set_index('hour').reindex(hours).ffill().fillna(0)

    dir_rad = np.deg2rad(w['wind_direction_10m'].values)
    speed   = w['wind_speed_10m'].values
    gust    = w['wind_gusts_10m'].values

    raw = np.stack([
        w['temperature_2m'].values,
        w['dew_point_2m'].values,
        w['relative_humidity_2m'].values,
        w['surface_pressure'].values,
        speed * np.sin(dir_rad),
        speed * np.cos(dir_rad),
        gust  * np.sin(dir_rad),
        gust  * np.cos(dir_rad),
        w['precipitation'].values,
    ], axis=1).astype(np.float32)  # (T, 9)

    return raw


def normalize_weather_features(
    weather_raw: np.ndarray,
    scaler: StandardScaler = None
) -> tuple[np.ndarray, StandardScaler]:
    """
    Z-score normalize first 8 weather features; keep precipitation (col 8) raw.
    Matches ST-BDP paper normalization strategy.

    Args:
        weather_raw: (T, 9) raw array from build_weather_features
        scaler:      pass fitted scaler for val/test transforms;
                     None to fit on this data (use training split only)

    Returns:
        normalized: (T, 9)
        scaler:     fitted StandardScaler (on first 8 columns)
    """
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(weather_raw[:, :8])

    normalized = weather_raw.copy()
    normalized[:, :8] = scaler.transform(weather_raw[:, :8])
    return normalized, scaler


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