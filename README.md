# 📈 Indian Stock Market Predictor

### Final Year CSE Project

A Machine Learning-based web application that predicts whether an Indian stock's price will go **UP** or **DOWN** the next day, using technical indicators and a Random Forest classifier.

---

## 🎯 What This Project Does

1. Downloads 5 years of historical stock data from Yahoo Finance
2. Calculates 15 technical indicators (RSI, MACD, Moving Averages, Bollinger Bands, etc.)
3. Trains a Random Forest ML model to predict next-day price direction (UP/DOWN)
4. Shows predictions in a web interface with charts and confidence scores

---

## 🛠️ Tech Stack

| Component       | Technology              |
|-----------------|------------------------|
| Language         | Python 3               |
| Data source      | Yahoo Finance (yfinance)|
| ML Model         | scikit-learn (Random Forest) |
| Web Interface    | Streamlit              |
| Charts           | Plotly, matplotlib      |

---

## 📂 Project Files

```
major kiro/
├── data_fetcher.py        # Downloads stock data from Yahoo Finance
├── feature_engineering.py # Calculates technical indicators + creates targets
├── train_model.py         # Trains 2 ML models (classifier + regressor)
├── predictor.py           # Loads saved models and makes predictions
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── HOW_IT_WORKS.md        # Detailed step-by-step explanation
├── REPORT.md              # Formal project report (formulas, model info, references)
└── models/                # Saved models + charts (created after training)
```

---

## 🚀 How to Run (Step by Step)

### Step 1: Install Python packages
```bash
pip install -r requirements.txt
```

### Step 2: Train the models
```bash
python train_model.py
```
This downloads data for 10 Indian stocks, trains models, and saves them in the `models/` folder.  
*(Takes about 5-10 minutes. You only need to do this once.)*

### Step 3: Launch the web app
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 📊 Stocks Included

| Stock Name         | Ticker          |
|--------------------|-----------------|
| Reliance Industries| RELIANCE.NS      |
| TCS               | TCS.NS           |
| Infosys           | INFY.NS          |
| HDFC Bank         | HDFCBANK.NS      |
| ICICI Bank        | ICICIBANK.NS     |
| Wipro             | WIPRO.NS         |
| SBI               | SBIN.NS          |
| Bajaj Finance     | BAJFINANCE.NS    |
| HUL               | HINDUNILVR.NS    |
| Bharti Airtel    | BHARTIARTL.NS    |

---

## 🔑 Technical Indicators Used

| Indicator         | What it measures                        |
|-------------------|----------------------------------------|
| MA_7, MA_21, MA_50| Moving Averages (short, mid, long term) |
| EMA_12, EMA_26    | Exponential Moving Averages             |
| MACD              | Trend direction & momentum              |
| RSI               | Overbought/oversold conditions (0-100)  |
| Bollinger Bands   | Volatility & price range                |
| Return_1d, 5d    | Price change percentage                  |
| Vol_Ratio         | Volume vs average volume                 |
| HL_Range          | High-Low price range                     |

---

## ❓ Questions Your Teacher Might Ask

**Q: Why Random Forest?**
A: Random Forest works well with tabular data, handles non-linear relationships, doesn't need much tuning, and gives feature importance so we can understand which indicators matter most.

**Q: What is the target variable?**
A: Binary — 1 if price goes UP next day, 0 if it goes DOWN.

**Q: Why no shuffling in train/test split?**
A: Stock data is time-series. We train on older data and test on recent data to avoid looking into the future.

**Q: What data do you use?**
A: 5 years of daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.

**Q: What metrics do you use to evaluate the model?**
A: We use multiple metrics — classification accuracy, ROC-AUC (Area Under the ROC Curve), RMSE for the regressor, and a trading strategy backtest comparing cumulative returns of the model's predictions versus buy-and-hold.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is NOT financial advice. Stock markets are influenced by many factors beyond technical indicators. Do not make real investment decisions based on this tool.
