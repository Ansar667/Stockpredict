# model.py

import json
import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model

from ml_utils import FEATURE_COLUMNS, compute_error_metrics, prepare_dataframe_for_model, flatten_yf_columns

MODELS_DIR = "models"
HISTORY_PADDING_DAYS = 200
PREDICTION_COLUMNS = ["Pred_LSTM", "Pred_XGB", "Pred_RF"]

scaler_X = joblib.load(os.path.join(MODELS_DIR, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(MODELS_DIR, "scaler_y.pkl"))
lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_model.h5"), compile=False)

meta_path = os.path.join(MODELS_DIR, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
else:
    meta = {}

TIME_STEPS = meta.get("time_steps", 5)


def _ensure_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in PREDICTION_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df


def predict_stock(ticker: str, start_date: str, end_date: str):
    if not ticker:
        raise ValueError("Ticker is required.")
    if not start_date or not end_date:
        raise ValueError("Both start and end dates are required.")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if start_ts >= end_ts:
        raise ValueError("Start date must be earlier than end date.")

    padded_start = (start_ts - pd.DateOffset(days=HISTORY_PADDING_DAYS)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=padded_start, end=end_date, progress=False)
    if df.empty:
        raise ValueError("No market data returned for the selected dates.")
    df = flatten_yf_columns(df)

    df = prepare_dataframe_for_model(df, TIME_STEPS, allow_fallback=True)
    features = df[FEATURE_COLUMNS].values
    X_scaled = scaler_X.transform(features)
    if len(df) <= TIME_STEPS:
        raise ValueError("Time range is too short: need more rows for LSTM sequences.")

    sequences = []
    target_indexes = []
    for i in range(TIME_STEPS, len(X_scaled)):
        sequences.append(X_scaled[i - TIME_STEPS : i])
        target_indexes.append(i)
    X_seq = np.array(sequences)
    if X_seq.size == 0:
        raise ValueError("Not enough sequence data for LSTM prediction.")

    lstm_pred_scaled = lstm_model.predict(X_seq, verbose=0)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).ravel()

    pred_prices = [None] * len(df)
    closes = df["Close"].values
    for idx, row_idx in enumerate(target_indexes):
        if row_idx >= len(closes):
            continue
        next_idx = row_idx + 1
        if next_idx >= len(closes):
            break
        base_price = closes[row_idx]
        pred_price = base_price * (1 + lstm_pred[idx])
        pred_prices[next_idx] = float(pred_price)

    result_df = df.copy()
    result_df["Pred_LSTM"] = pred_prices
    result_df = _ensure_prediction_columns(result_df)
    result_df = result_df.loc[(result_df.index >= start_ts) & (result_df.index <= end_ts)]
    result_df.index.name = "Date"

    if result_df.empty:
        raise ValueError("No rows available for the requested window after preprocessing.")

    last_close = result_df["Close"].iloc[-1]
    last_pred = result_df["Pred_LSTM"].dropna().iloc[-1] if result_df["Pred_LSTM"].notna().any() else None

    summary = f"Latest close: {last_close:.2f}"
    if last_pred is not None:
        summary += f" | LSTM: {last_pred:.2f}"

    metrics = {}
    actual = result_df["Close"].to_numpy(dtype=float)
    for column in PREDICTION_COLUMNS:
        if column in result_df:
            values = result_df[column].to_numpy(dtype=float)
            stats = compute_error_metrics(actual, values)
            if stats:
                metrics[column] = stats

    return result_df, summary, metrics
