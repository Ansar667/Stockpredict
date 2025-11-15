import json
import os
import joblib
from datetime import date, timedelta

from ml_utils import (
    prepare_multiticker_data,
    train_lstm,
    evaluate_lstm,
)

LOOKBACK_YEARS = 8
TRAIN_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "META", "AMZN"]
END_TRAIN = date.today().isoformat()
START_TRAIN = (date.today() - timedelta(days=LOOKBACK_YEARS * 365)).isoformat()
TIME_STEPS = 20
MODELS_DIR = "models"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[INFO] Training on tickers: {', '.join(TRAIN_TICKERS)}")
    (
        X_train_seq,
        X_test_seq,
        y_train_seq,
        y_test_seq,
        scaler_X,
        scaler_y,
    ) = prepare_multiticker_data(
        TRAIN_TICKERS,
        START_TRAIN,
        END_TRAIN,
        time_steps=TIME_STEPS,
    )

    print(f"[INFO] Train sequences: {len(X_train_seq)}, Test sequences: {len(X_test_seq)}")

    lstm_model = train_lstm(X_train_seq, y_train_seq, epochs=60, batch_size=64)

    metrics, *_ = evaluate_lstm(lstm_model, X_test_seq, y_test_seq, scaler_y)
    print(metrics)

    lstm_path = os.path.join(MODELS_DIR, "lstm_model.h5")
    lstm_model.save(lstm_path)
    scaler_x_path = os.path.join(MODELS_DIR, "scaler_X.pkl")
    scaler_y_path = os.path.join(MODELS_DIR, "scaler_y.pkl")
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

    meta = {
        "train_tickers": TRAIN_TICKERS,
        "start": START_TRAIN,
        "end": END_TRAIN,
        "time_steps": TIME_STEPS,
        "target": "Return",
        "best_model": "LSTM",
        "metrics": metrics,
    }
    meta_path = os.path.join(MODELS_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
