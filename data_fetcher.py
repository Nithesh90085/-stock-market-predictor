# data_fetcher.py
# Fetches historical stock data from Yahoo Finance for Indian stocks

import yfinance as yf
import pandas as pd

# Popular Indian stocks (NSE symbols with .NS suffix for Yahoo Finance)
INDIAN_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "SBI": "SBIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "HUL": "HINDUNILVR.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
}

def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker.
    period: '1y', '2y', '5y', 'max'
    """
    print(f"Fetching data for {ticker} ...")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    df.dropna(inplace=True)
    print(f"  Got {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df


if __name__ == "__main__":
    # Quick test
    df = fetch_stock_data("RELIANCE.NS", period="1y")
    print(df.tail())
