# ml_utils.py

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"

BASE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD']
EXTRA_FEATURES = [
    'Return_1', 'Return_5', 'Volatility_10', 'Volume_Change',
    'High_Low_Range', 'ATR', 'OBV',
    'Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 'Lag_Close_5'
]
FEATURE_COLUMNS = BASE_FEATURES + EXTRA_FEATURES
MIN_FULL_ROWS = 120
MIN_LITE_ROWS = 60


def search_tickers(query: str, limit: int = 5) -> list[dict[str, str]]:
    query = (query or '').strip()
    if not query:
        return []

    params = {"q": query, "lang": "ru-RU", "region": "US"}
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; stockpredict/1.0)",
        "Accept": "application/json",
    }
    try:
        response = requests.get(YAHOO_SEARCH_URL, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    results: list[dict[str, str]] = []
    for item in data.get('quotes', []):
        symbol = item.get('symbol')
        if not symbol:
            continue
        name = item.get('shortname') or item.get('longname') or item.get('name') or ''
        exchange = item.get('exchange') or item.get('exchDisp') or ''
        results.append({'symbol': symbol, 'name': name, 'exchange': exchange})
        if len(results) >= limit:
            break
    return results


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None
    return flatten_yf_columns(df)


def add_technical_indicators(df: pd.DataFrame, *, lite: bool = False) -> pd.DataFrame:
    df = df.copy()

    ma_short = 10 if not lite else 5
    ma_long = 50 if not lite else 15
    rsi_period = 14 if not lite else 6
    atr_period = 14 if not lite else 5
    ema_fast = 12 if not lite else 5
    ema_slow = 26 if not lite else 10
    ret_long = 5 if not lite else 3
    vol_window = 10 if not lite else 5

    close = df['Close']

    df['MA_10'] = close.rolling(ma_short).mean()
    df['MA_50'] = close.rolling(ma_long).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_fast_series = close.ewm(span=ema_fast, adjust=False).mean()
    ema_slow_series = close.ewm(span=ema_slow, adjust=False).mean()
    df['MACD'] = ema_fast_series - ema_slow_series

    df['Return_1'] = close.pct_change()
    df['Return_5'] = close.pct_change(ret_long)
    df['Volatility_10'] = df['Return_1'].rolling(vol_window).std()
    df['Volume_Change'] = df['Volume'].replace(0, np.nan).pct_change()
    df['High_Low_Range'] = (df['High'] - df['Low']) / close.replace(0, np.nan)

    tr1 = df['High'] - df['Low']
    prev_close = close.shift()
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(atr_period).mean()

    volume = df['Volume'].fillna(0)
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    for lag in [1, 2, 3, 5]:
        df[f'Lag_Close_{lag}'] = close.shift(lag)

    return df


def create_sequences(X: np.ndarray, y: np.ndarray | None = None, time_steps: int = 5):
    X_seq = []
    y_seq = []
    has_target = y is not None
    for i in range(time_steps, len(X)):
        X_seq.append(X[i - time_steps : i])
        if has_target:
            y_seq.append(y[i])
    X_array = np.array(X_seq)
    y_array = np.array(y_seq) if has_target else None
    return X_array, y_array


def _sanitize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    critical_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in critical_cols:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan
    if "Volume" in df.columns:
        df.loc[df["Volume"] <= 0, "Volume"] = np.nan
    return df


def _prepare_feature_frame(
    df: pd.DataFrame,
    *,
    time_steps: int,
    lite: bool,
    min_rows: int,
) -> pd.DataFrame | None:
    frame = add_technical_indicators(df, lite=lite)
    close = frame["Close"].replace(0, np.nan)
    frame["Return_Target"] = close.shift(-1) / close - 1
    frame = _sanitize_numeric_frame(frame)
    frame.dropna(inplace=True)
    required_rows = max(min_rows, time_steps + 5)
    if len(frame) < required_rows:
        return None
    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    cast_cols = [col for col in FEATURE_COLUMNS + ["Return_Target"] if col in frame.columns]
    frame[cast_cols] = frame[cast_cols].astype(np.float64)
    return frame


def prepare_dataframe_for_model(df: pd.DataFrame, time_steps: int, *, allow_fallback: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No market data returned for the selected dates.")

    attempts: list[tuple[bool, int]] = [(False, MIN_FULL_ROWS)]
    if allow_fallback:
        attempts.append((True, MIN_LITE_ROWS))

    for lite, min_rows in attempts:
        frame = _prepare_feature_frame(df.copy(), time_steps=time_steps, lite=lite, min_rows=min_rows)
        if frame is not None:
            return frame
    raise ValueError("Not enough clean data for this ticker after feature engineering. Try another period or ticker.")


def prepare_multiticker_data(tickers, start_date: str, end_date: str, time_steps: int = 5):
    ticker_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    all_features = []
    all_targets = []

    for ticker in tickers:
        df = download_stock_data(ticker, start_date, end_date)
        if df is None or df.empty:
            continue
        try:
            processed = prepare_dataframe_for_model(df, time_steps, allow_fallback=True)
        except ValueError as exc:
            print(f"[WARN] Skipping {ticker}: {exc}")
            continue
        if len(processed) <= time_steps:
            continue

        features = processed[FEATURE_COLUMNS].values
        target = processed["Return_Target"].values.reshape(-1, 1)
        ticker_arrays.append((features, target))
        all_features.append(features)
        all_targets.append(target)

    if not ticker_arrays:
        raise ValueError("No sufficient data fetched for the requested tickers.")

    stacked_features = np.vstack(all_features)
    stacked_targets = np.vstack(all_targets)

    scaler_X = MinMaxScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(stacked_features)
    scaler_y.fit(stacked_targets)

    X_sequences = []
    y_sequences = []
    for features, target in ticker_arrays:
        X_scaled = scaler_X.transform(features)
        y_scaled = scaler_y.transform(target)
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
        if X_seq.size and y_seq is not None and y_seq.size:
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

    if not X_sequences:
        raise ValueError("Not enough sequence data after processing tickers.")

    X_all = np.vstack(X_sequences)
    y_all = np.concatenate(y_sequences).ravel()

    split_idx = int(len(X_all) * 0.8)
    X_train_seq = X_all[:split_idx]
    X_test_seq = X_all[split_idx:]
    y_train_seq = y_all[:split_idx]
    y_test_seq = y_all[split_idx:]

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler_X, scaler_y


def train_lstm(X_train_seq, y_train_seq, epochs: int = 40, batch_size: int = 32):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
    ]

    model.fit(
        X_train_seq,
        y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    return model


def evaluate_lstm(model, X_test_seq, y_test_seq, scaler_y):
    y_pred = model.predict(X_test_seq)
    y_pred_inv = scaler_y.inverse_transform(y_pred).ravel()
    y_test_inv = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()

    metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))),
        'MAE': float(mean_absolute_error(y_test_inv, y_pred_inv)),
        'R2': float(r2_score(y_test_inv, y_pred_inv)),
    }
    return metrics, y_pred_inv, y_test_inv


def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, linestyle='--', label='Predicted')
    plt.title(f'{model_name} Prediction')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()


def compute_error_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return None
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    denom_mask = y_true != 0
    if denom_mask.sum():
        mape = float(np.mean(np.abs(residual[denom_mask] / y_true[denom_mask])) * 100.0)
    else:
        mape = None
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }

