# predictor.py
# Loads saved models and makes a prediction for the latest available data.
# Two models:
#   1. Classifier  -> predicts UP or DOWN (direction)
#   2. Regressor   -> predicts percentage price change for next day

import os
import joblib
import numpy as np
import pandas as pd

from data_fetcher import fetch_stock_data
from feature_engineering import add_technical_indicators

MODELS_DIR = "models"

FEATURE_COLS = [
    "MA_7", "MA_21", "MA_50",
    "EMA_12", "EMA_26",
    "MACD", "MACD_Signal",
    "RSI",
    "BB_Upper", "BB_Lower", "BB_Width",
    "Return_1d", "Return_5d",
    "Vol_Ratio", "HL_Range",
]


def predict_next_day(ticker: str) -> dict:
    """
    Returns a dict with:
      - prediction      : 'UP' or 'DOWN'
      - confidence_up   : probability price goes UP (0-100)
      - confidence_down  : probability price goes DOWN (0-100)
      - latest_close     : today's closing price
      - predicted_pct    : predicted % change for tomorrow
      - predicted_price  : predicted price for tomorrow
      - change_rupees    : predicted change in rupees
      - indicators       : dict of latest indicator values
    """
    safe_name     = ticker.replace(".", "_")
    model_path     = os.path.join(MODELS_DIR, f"{safe_name}_model.pkl")
    scaler_path    = os.path.join(MODELS_DIR, f"{safe_name}_scaler.pkl")
    regressor_path = os.path.join(MODELS_DIR, f"{safe_name}_regressor.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found for {ticker}. "
            f"Please run  python train_model.py  first."
        )

    # Load all three: classifier, scaler, regressor
    model     = joblib.load(model_path)
    scaler    = joblib.load(scaler_path)
    regressor = joblib.load(regressor_path) if os.path.exists(regressor_path) else None

    # Fetch recent data (6 months is enough to compute all indicators)
    df = fetch_stock_data(ticker, period="6mo")
    df = add_technical_indicators(df)

    latest_close = float(df["Close"].squeeze().iloc[-1])
    latest = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    latest_sc = scaler.transform(latest)

    # --- Classifier: UP or DOWN? ---
    prob_up    = model.predict_proba(latest_sc)[0][1]
    prob_down  = 1 - prob_up

    # --- Classifier owns the direction, always ---
    # The classifier was trained specifically for UP/DOWN and its confidence
    # is what the confidence bars display. Never let the regressor override it.
    prediction = "UP" if prob_up >= 0.5 else "DOWN"

    # --- Regressor: magnitude of % change ---
    # Use the regressor for the size of the move, but force its sign to agree
    # with the classifier so the banner, confidence bars, and text are consistent.
    predicted_pct = 0.0
    if regressor is not None:
        raw_pct = float(regressor.predict(latest_sc)[0])
        # Align sign with classifier direction
        if prediction == "UP":
            predicted_pct = abs(raw_pct) if raw_pct != 0.0 else 0.1
        else:
            predicted_pct = -abs(raw_pct) if raw_pct != 0.0 else -0.1

    predicted_price = latest_close * (1 + predicted_pct / 100)
    change_rupees   = predicted_price - latest_close

    indicators = {col: round(float(df[col].squeeze().iloc[-1]), 4)
                  for col in FEATURE_COLS}

    return {
        "prediction":       prediction,
        "confidence_up":    round(prob_up * 100, 2),
        "confidence_down":  round(prob_down * 100, 2),
        "latest_close":     round(latest_close, 2),
        "predicted_pct":    round(predicted_pct, 2),
        "predicted_price":  round(predicted_price, 2),
        "change_rupees":    round(change_rupees, 2),
        "indicators":       indicators,
    }


if __name__ == "__main__":
    result = predict_next_day("RELIANCE.NS")
    print(f"\nPrediction        : {result['prediction']}")
    print(f"Confidence        : UP={result['confidence_up']}%  DOWN={result['confidence_down']}%")
    print(f"Today's Close     : Rs.{result['latest_close']}")
    print(f"Predicted Change  : {result['predicted_pct']}%")
    print(f"Predicted Price   : Rs.{result['predicted_price']}")
    print(f"Change in Rs      : Rs.{result['change_rupees']}")
