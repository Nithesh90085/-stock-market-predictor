# feature_engineering.py
# Creates technical indicators as features for the ML model

import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common technical indicators to the dataframe.
    These are the 'signals' traders use — we use them as ML features.
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # ── Moving Averages ──────────────────────────────────────────────
    df["MA_7"]  = close.rolling(window=7).mean()
    df["MA_21"] = close.rolling(window=21).mean()
    df["MA_50"] = close.rolling(window=50).mean()

    # ── Exponential Moving Average ───────────────────────────────────
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # ── MACD (Moving Average Convergence Divergence) ─────────────────
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ── RSI (Relative Strength Index) ────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_mid          = close.rolling(20).mean()
    bb_std          = close.rolling(20).std()
    df["BB_Upper"]  = bb_mid + 2 * bb_std
    df["BB_Lower"]  = bb_mid - 2 * bb_std
    df["BB_Width"]  = df["BB_Upper"] - df["BB_Lower"]

    # ── Price momentum ────────────────────────────────────────────────
    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)

    # ── Volume indicators ─────────────────────────────────────────────
    df["Vol_MA_10"]   = volume.rolling(10).mean()
    df["Vol_Ratio"]   = volume / (df["Vol_MA_10"] + 1e-10)

    # ── High-Low range ────────────────────────────────────────────────
    df["HL_Range"] = high - low

    df.dropna(inplace=True)
    return df


def create_target(df: pd.DataFrame, days_ahead: int = 1) -> pd.DataFrame:
    """
    Creates two targets:
      - Target     : 1 if price goes UP after `days_ahead` days, else 0 (for classifier)
      - Target_Pct  : percentage price change for next day (for regressor)
    """
    df = df.copy()
    close = df["Close"].squeeze()
    future_close = close.shift(-days_ahead)
    df["Target"] = (future_close > close).astype(int)
    df["Target_Pct"] = ((future_close - close) / close * 100).round(4)
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    from data_fetcher import fetch_stock_data
    df = fetch_stock_data("TCS.NS", period="2y")
    df = add_technical_indicators(df)
    df = create_target(df)
    print(df[["Close", "RSI", "MACD", "Target"]].tail(10))
