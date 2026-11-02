# 📖 How It Works — Complete Beginner's Guide

## Indian Stock Market Predictor — Step by Step Explanation

---

Let's say you pick **Reliance Industries** in the app and click "Predict Next Trading Day".

Here is exactly what happens, from start to finish, in simple language.

---

## STEP 1: Data Fetching (data_fetcher.py)

### What happens?
The app contacts **Yahoo Finance** (a free website) and downloads 5 years of daily stock data for Reliance.

### What data does it get?
Every single trading day from November 2021 to November 2026 — about **1,236 rows**. Each row has 5 numbers:

| Date       | Open   | High   | Low    | Close  | Volume    |
|------------|--------|--------|--------|--------|-----------|
| 2024-01-15 | 2450.0 | 2480.0 | 2440.0 | 2470.0 | 5,200,000 |
| 2024-01-16 | 2470.0 | 2500.0 | 2465.0 | 2495.0 | 6,100,000 |
| 2024-01-17 | 2495.0 | 2510.0 | 2480.0 | 2480.0 | 4,800,000 |

**What do these mean?**
- **Open** = Price when market opens at 9:15 AM
- **High** = Highest price reached during the day
- **Low** = Lowest price reached during the day
- **Close** = Price when market closes at 3:30 PM (this is the most important number)
- **Volume** = How many shares were bought and sold that day

### How does the code do it?
```python
import yfinance as yf
df = yf.download("RELIANCE.NS", period="5y")
```
That's it. One line of code downloads 5 years of data. `RELIANCE.NS` means Reliance on the NSE (National Stock Exchange of India). The `.NS` suffix tells Yahoo Finance it's an Indian stock.

---

## STEP 2: Feature Engineering (feature_engineering.py)

### What happens?
The raw data (Open, High, Low, Close, Volume) is not enough for the ML model. We need to calculate **15 technical indicators** — these are mathematical formulas that help us understand the stock's behavior.

### Why do we need indicators?
Imagine someone asks you "Will Reliance go up tomorrow?" You wouldn't just look at today's price. You'd ask:
- Is the price trending up or down recently? → Moving Averages
- Is the stock overbought or oversold? → RSI
- Is the trend changing? → MACD
- Is the stock volatile right now? → Bollinger Bands

These indicators answer those questions with numbers.

### The 15 Indicators Explained (with Reliance example)

Let's say today's Reliance data is:
- Close = ₹1,250
- Date = November 2, 2026

#### Indicator 1, 2, 3: Moving Averages (MA_7, MA_21, MA_50)

**What it is:** The average closing price over the last N days.

**Example:**
```
MA_7 = average of last 7 days closing prices
       = (1250 + 1245 + 1260 + 1255 + 1248 + 1252 + 1249) / 7
       = 1251.28
```

**Why it matters:** If today's price (₹1,250) is BELOW the 7-day average (₹1,251.28), the stock is slightly dipping. If MA_7 is above MA_50, the short-term trend is upward ("Golden Cross").

#### Indicator 4, 5: Exponential Moving Averages (EMA_12, EMA_26)

**What it is:** Same as moving average, but gives MORE weight to recent days. Reacts faster to price changes.

**Example:**
```
EMA_12 = weighted average of last 12 days
         (today's price gets more weight than 12 days ago)
```

#### Indicator 6, 7: MACD and MACD Signal

**What it is:** MACD shows the difference between the 12-day and 26-day EMAs. It tells us if the short-term trend is stronger or weaker than the long-term trend.

**Example:**
```
MACD = EMA_12 - EMA_26
     = 1252 - 1248
     = +4.0

MACD_Signal = 9-day average of MACD values = +3.5
```

**How to read it:**
- MACD > MACD_Signal → upward momentum (price might rise)
- MACD < MACD_Signal → downward momentum (price might fall)

In our example: MACD (+4.0) > Signal (+3.5) → slightly bullish

#### Indicator 8: RSI (Relative Strength Index)

**What it is:** A score from 0 to 100 that tells if a stock is overbought or oversold.

**How it's calculated:**
```
Step 1: Look at last 14 days
Step 2: Calculate average gain on UP days = ₹15/day (example)
Step 3: Calculate average loss on DOWN days = ₹10/day (example)
Step 4: RS = Average Gain / Average Loss = 15/10 = 1.5
Step 5: RSI = 100 - (100 / (1 + RS)) = 100 - (100 / 2.5) = 60
```

**How to read it:**
- RSI > 70 → Overbought (price too high, might FALL soon)
- RSI < 30 → Oversold (price too low, might RISE soon)
- RSI = 50 → Neutral

In our example: RSI = 60 → slightly bullish but not extreme

#### Indicator 9, 10, 11: Bollinger Bands (BB_Upper, BB_Lower, BB_Width)

**What it is:** Upper and lower limits that show the expected price range.

**How it's calculated:**
```
BB_Middle = 20-day moving average = ₹1,248
BB_Std    = standard deviation of last 20 days = ₹15 (how much price varies)
BB_Upper  = 1248 + (2 × 15) = ₹1,278
BB_Lower  = 1248 - (2 × 15) = ₹1,218
BB_Width  = 1278 - 1218 = ₹60
```

**How to read it:**
- If price touches BB_Upper → stock might be overpriced
- If price touches BB_Lower → stock might be underpriced
- Narrow BB_Width → low volatility (price not moving much)
- Wide BB_Width → high volatility (price swinging a lot)

#### Indicator 12, 13: Returns (Return_1d, Return_5d)

**What it is:** Percentage change in price over 1 day and 5 days.

**Example:**
```
Return_1d = (Today's Close - Yesterday's Close) / Yesterday's Close × 100
          = (1250 - 1245) / 1245 × 100
          = +0.40%

Return_5d = (Today's Close - 5 Days Ago Close) / 5 Days Ago Close × 100
          = (1250 - 1230) / 1230 × 100
          = +1.63%
```

**Why it matters:** Shows recent momentum. Positive return = stock is going up.

#### Indicator 14: Volume Ratio (Vol_Ratio)

**What it is:** Today's trading volume compared to the 10-day average volume.

**Example:**
```
Vol_MA_10 = average volume of last 10 days = 5,000,000
Vol_Ratio = Today's Volume / Vol_MA_10
          = 7,500,000 / 5,000,000
          = 1.5
```

**How to read it:**
- Vol_Ratio > 1.5 → Unusual activity (something is happening, big players are trading)
- Vol_Ratio < 0.5 → Low interest (quiet day)

#### Indicator 15: High-Low Range (HL_Range)

**What it is:** Difference between the highest and lowest price of the day.

**Example:**
```
HL_Range = High - Low = 1270 - 1235 = ₹35
```

**Why it matters:** Shows how much the price swung during the day. Large range = high volatility.

### After Step 2, our data looks like:

| Date       | Close  | MA_7   | RSI | MACD | BB_Upper | BB_Lower | Return_1d | Vol_Ratio | ... |
|------------|--------|--------|-----|------|----------|----------|-----------|-----------|-----|
| 2024-01-15 | 2470.0 | 2465.2 | 55  | +2.1 | 2500     | 2440     | +0.40%    | 1.2       | ... |
| 2024-01-16 | 2495.0 | 2471.5 | 62  | +3.8 | 2510     | 2450     | +1.01%    | 1.5       | ... |
| 2024-01-17 | 2480.0 | 2475.0 | 58  | +2.5 | 2505     | 2455     | -0.60%    | 0.9       | ... |

Each row now has **15 indicator values** instead of just 5 raw prices.

---

## STEP 3: Creating the Target (What We Want to Predict)

### What happens?
For each historical day, we look at what happened the NEXT day and label it:

**For the Classifier (UP/DOWN):**
```
If tomorrow's Close > today's Close → Target = 1 (UP)
If tomorrow's Close < today's Close → Target = 0 (DOWN)
```

**For the Regressor (Price Change):**
```
Target_Pct = (Tomorrow's Close - Today's Close) / Today's Close × 100
```

### Example with Reliance:

| Date       | Today's Close | Tomorrow's Close | Target | Target_Pct |
|------------|--------------|------------------|--------|------------|
| 2024-01-15 | ₹2,470       | ₹2,495           | 1 (UP) | +1.01%     |
| 2024-01-16 | ₹2,495       | ₹2,480           | 0 (DOWN) | -0.60%   |
| 2024-01-17 | ₹2,480       | ₹2,510           | 1 (UP) | +1.21%     |
| 2024-01-18 | ₹2,510       | ₹2,496           | 0 (DOWN) | -0.56%   |

The model's job: **Given today's 15 indicators, predict the Target (UP/DOWN) and Target_Pct (% change).**

---

## STEP 4: Training the Model (train_model.py)

### What happens?
We feed all 1,236 rows of historical data (indicators + targets) into a Machine Learning algorithm called **Random Forest**. The algorithm learns patterns like "when RSI is below 30 and volume is high, the price usually goes up the next day."

### How is the data split?

```
Total data: 1,236 days (2021-2026)
├── Training: 988 days (80%) → used to teach the model
└── Testing:  248 days (20%) → used to check if the model actually learned
```

**Important:** We DON'T shuffle the data. We train on older data (2021-2024) and test on newer data (2025-2026). This is because in real life, we predict the future using the past — not the other way around.

### What is StandardScaler and why do we need it?

Different indicators have different scales:
- Close price = ₹1,250 (large number)
- RSI = 60 (small number, 0-100)
- Return_1d = 0.40 (tiny number, percentage)

If we don't scale them, the model might think Close price is more important just because the number is bigger. **StandardScaler** converts all values to the same scale (average = 0, range = -1 to +1) so the model treats all indicators fairly.

```
Before scaling:  Close = 1250,  RSI = 60,  Return_1d = 0.40
After scaling:   Close = +0.8,  RSI = -0.3, Return_1d = +0.5
```

---

## STEP 5: How Random Forest Works (The Core Algorithm)

### What is a Decision Tree?

A decision tree is like a flowchart of Yes/No questions. The model learns these questions from the data.

**Example of ONE decision tree (simplified):**

```
                   Is RSI > 50?
                  /            \
               YES              NO
               /                  \
    Is MACD > Signal?       Is Volume_Ratio > 1.5?
       /        \              /        \
     YES        NO            YES        NO
      |          |              |          |
  Predict UP  Predict DOWN  Predict DOWN  Predict UP
  (82% were   (61% were     (70% were     (55% were
   actually     actually     actually      actually
    UP)         DOWN)       DOWN)          UP)
```

**How the tree was built:**
1. The algorithm looks at all 988 training days
2. It tries every indicator and every possible split point
3. It picks the split that best separates UP days from DOWN days
4. In our example, "RSI > 50?" was the best first split — it separated 82% of UP days to one side
5. Then it repeats for each branch, asking more questions

### What is a Random Forest?

A **Random Forest** is a collection of **200 decision trees**.

But here's the twist — each tree is slightly different:
1. Each tree is trained on a **random subset** of the data (not all 988 days)
2. Each tree only sees a **random subset** of the 15 indicators at each split

This makes each tree unique, like asking 200 different analysts for their opinion.

### How the final prediction is made:

```
Tree 1:    Predicts UP
Tree 2:    Predicts DOWN
Tree 3:    Predicts UP
Tree 4:    Predicts UP
Tree 5:    Predicts DOWN
...
Tree 200:  Predicts UP

Total votes: 124 UP, 76 DOWN

Final prediction: UP (62% voted UP, 38% voted DOWN)
```

**This is exactly what the "Confidence" shows in the app:**
- Confidence UP = 62% (124 trees voted UP)
- Confidence DOWN = 38% (76 trees voted DOWN)

### Why Random Forest and not just one tree?

One tree can **overfit** — it memorizes the training data but does badly on new data. Random Forest solves this by averaging 200 trees, making the prediction more stable and reliable.

---

## STEP 6: Two Models (Classifier + Regressor)

We actually train TWO separate Random Forests:

### Model 1: Classifier (predicts UP or DOWN)
- **Input:** 15 indicators for today
- **Output:** 1 (UP) or 0 (DOWN)
- **How:** 200 trees vote, majority wins
- **Example output:** UP with 62% confidence

### Model 2: Regressor (predicts percentage change)
- **Input:** Same 15 indicators for today
- **Output:** A number like +1.05% or -2.70%
- **How:** 200 trees predict a number, average of all 200 is the final answer
- **Example output:** -2.70%

### How the predicted price is calculated:

```
Today's Close     = ₹1,250
Predicted % Change = -2.70%

Predicted Price   = Today's Close × (1 + Predicted % / 100)
                  = 1250 × (1 + (-2.70 / 100))
                  = 1250 × 0.973
                  = ₹1,216.25

Change in Rupees = Predicted Price - Today's Close
                 = 1216.25 - 1250
                 = -₹33.75
```

---

## STEP 7: Making a Prediction (predictor.py)

### What happens when you click "Predict" in the app?

```
1. App calls predictor.py
2. predictor.py downloads latest 6 months of Reliance data from Yahoo Finance
3. Calculates today's 15 indicators (same formulas as Step 2)
4. Takes ONLY today's row (the latest day's indicators)
5. Scales the values using the saved StandardScaler
6. Feeds the 15 scaled values into the saved Classifier model
   → 200 trees vote → 51% say UP, 49% say DOWN
7. Feeds the same 15 scaled values into the saved Regressor model
   → 200 trees predict numbers → average = -2.70%
8. Calculates predicted price:
   → 1250 × (1 + (-2.70/100)) = ₹1,216.25
9. Returns everything to the app:
   - Prediction: DOWN (because -2.70% is negative)
   - Confidence: UP=51%, DOWN=49%
   - Today's Close: ₹1,250
   - Predicted Change: -2.70%
   - Predicted Price: ₹1,216.25
   - Change in Rs: -₹33.75
   - All 15 indicator values
```

---

## COMPLETE SYSTEM FLOW DIAGRAM

```
USER opens app.py in browser
         |
         v
USER selects "Reliance Industries" and clicks "Predict"
         |
         v
┌─────────────────────────────────────────────────────────┐
│  app.py (Streamlit Web Interface)                        │
│  Receives the request, calls predictor.py                │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  predictor.py                                            │
│  Step A: Load saved model files from models/ folder      │
│          - RELIANCE_NS_model.pkl     (classifier)        │
│          - RELIANCE_NS_regressor.pkl (regressor)          │
│          - RELIANCE_NS_scaler.pkl    (scaler)            │
│                                                          │
│  Step B: Call data_fetcher.py to download latest data    │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  data_fetcher.py                                         │
│  Downloads 6 months of Reliance data from Yahoo Finance  │
│  Returns DataFrame with Open, High, Low, Close, Volume  │
│  (about 126 rows)                                       │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  feature_engineering.py                                  │
│  Calculates 15 technical indicators from the raw data    │
│  MA_7, MA_21, MA_50, EMA_12, EMA_26, MACD, MACD_Signal,  │
│  RSI, BB_Upper, BB_Lower, BB_Width, Return_1d, Return_5d,│
│  Vol_Ratio, HL_Range                                    │
│                                                          │
│  Returns enriched DataFrame with 15+ columns             │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  predictor.py (continued)                                │
│  Takes the LAST ROW (today's 15 indicators)             │
│  Scales them using the saved StandardScaler              │
│                                                          │
│  Feeds into Classifier → 200 trees vote → UP/DOWN        │
│  Feeds into Regressor  → 200 trees predict → % change    │
│                                                          │
│  Calculates: predicted_price, change_in_rupees          │
│  Returns all results to app.py                           │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  app.py (Streamlit Web Interface)                        │
│  Displays:                                               │
│  - Prediction: UP or DOWN (green/red banner)             │
│  - Today's Close: ₹1,250                                 │
│  - Predicted Change: -2.70%                              │
│  - Predicted Price: ₹1,216.25                            │
│  - Change in Rs: -₹33.75                                 │
│  - Confidence: UP=51%, DOWN=49%                          │
│  - 15 technical indicator values in a table              │
│  - Interactive candlestick price chart                   │
│  - ROC curve and trading strategy backtest charts         │
└─────────────────────────────────────────────────────────┘
```

---

## TRAINING FLOW (What happened when you ran train_model.py)

This only needs to be done ONCE. The saved models are reused every time.

```
train_model.py
         |
         v
For EACH of the 10 stocks (Reliance, TCS, Infosys, etc.):
         |
         v
┌─────────────────────────────────────────────────────────┐
│  1. DOWNLOAD 5 years of daily data from Yahoo Finance    │
│     → 1,236 rows of Open, High, Low, Close, Volume      │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  2. CALCULATE 15 technical indicators                   │
│     → Each row now has 15 feature columns                │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  3. CREATE TARGETS                                      │
│     → Target = 1 (UP) or 0 (DOWN) for each day          │
│     → Target_Pct = % price change for next day          │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  4. SPLIT DATA (time-ordered, no shuffling)              │
│     → First 80% (988 days) = Training data               │
│     → Last 20% (248 days) = Testing data                 │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  5. SCALE FEATURES using StandardScaler                 │
│     → All 15 indicators converted to same scale          │
│     → Scaler saved as .pkl file for later use            │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  6. TRAIN CLASSIFIER (RandomForestClassifier)           │
│     → 200 decision trees learn from 988 training days   │
│     → Each tree sees random subset of data + indicators  │
│     → Tree learns: "If RSI<30 & Vol>1.5 → usually UP"    │
│     → Model saved as .pkl file                           │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  7. TRAIN REGRESSOR (RandomForestRegressor)              │
│     → Same 200 trees but predict % change (a number)     │
│     → Tree learns: "If RSI<30 & Vol>1.5 → +1.2% change" │
│     → Model saved as .pkl file                           │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  8. EVALUATE on the 248 testing days (unseen data)       │
│     → Classifier: what % did it predict correctly?       │
│     → Regressor: how far off were the % predictions?    │
│     → Print accuracy, classification report, RMSE       │
└─────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────┐
│  9. SAVE CHARTS                                          │
│     → ROC curve .png (model discrimination ability)    │
│     → Trading backtest .png (strategy vs buy & hold)    │
│     → Feature importance .png (which indicators matter)  │
└─────────────────────────────────────────────────────────┘
         |
         v
Move to next stock...
```

---

## WHY IS ACCURACY ONLY ~50-53%?

### The honest answer:

Stock market prediction is one of the hardest problems in data science. Here's why:

**1. The market is influenced by things we can't measure:**
```
Our model sees:              The market also reacts to:
- Price history              - Company earnings reports
- Volume                     - Government policies
- Technical indicators        - Global news (wars, elections)
                             - Investor emotions (fear, greed)
                             - Insider trading
                             - Social media rumors
                             - Interest rate changes
                             - Oil prices
                             - Currency exchange rates
```

**2. Random guessing = 50%:**
```
If you flip a coin to predict UP/DOWN:
- You'll be right 50% of the time

Our model gets ~51-53%:
- It's only 1-3% better than a coin flip
- But in the stock market, even 1% edge over millions of trades = huge profit
- Professional hedge funds are happy with 52-55%
```

**3. The market is somewhat efficient:**
If there was an easy pattern, everyone would use it and it would stop working. The fact that it's hard to predict is actually a sign that the market is working correctly.

**4. We're only using technical indicators:**
Real stock predictions combine:
- Technical analysis (what we do) → 50-53% accuracy
- Fundamental analysis (company financials) → adds maybe 2-3%
- Sentiment analysis (news, Twitter) → adds maybe 1-2%
- Even all combined → still only 55-60%

### What to tell your teacher:

"The model achieves 50-53% accuracy which is consistent with published research on daily stock price direction prediction using technical indicators. The Efficient Market Hypothesis suggests that historical price patterns are already reflected in current prices, making consistent outperformance difficult. Our goal was to demonstrate the complete ML pipeline — data collection, feature engineering, model training, evaluation, and deployment — rather than to achieve high accuracy."

---

## FILE SUMMARY

```
major kiro/
│
├── data_fetcher.py        # STEP 1: Downloads stock data from Yahoo Finance
│                          #   Input:  Stock ticker (e.g., "RELIANCE.NS")
│                          #   Output: DataFrame with Open, High, Low, Close, Volume
│
├── feature_engineering.py # STEP 2: Calculates 15 technical indicators
│                          #   Input:  Raw OHLCV data
│                          #   Output: DataFrame with 15 indicator columns + targets
│                          #   Also creates: Target (UP/DOWN) and Target_Pct (% change)
│
├── train_model.py         # STEP 3: Trains the ML models (run ONCE)
│                          #   Input:  Historical data + indicators + targets
│                          #   Process: Train 200-tree Random Forest (classifier + regressor)
│                          #   Output: Saved .pkl files + performance charts
│
├── predictor.py           # STEP 4: Makes predictions using saved models
│                          #   Input:  Stock ticker
│                          #   Process: Fetch latest data → calculate indicators → predict
│                          #   Output: Direction, confidence, predicted price, % change
│
├── app.py                 # STEP 5: Web interface (Streamlit)
│                          #   Connects all the pieces and shows results in the browser
│
├── models/                # Folder where trained models are saved
│                          #   Contains: .pkl files + ROC curve + trading backtest + feature importance
│
├── requirements.txt       # List of Python packages needed
├── README.md              # Quick start guide
└── HOW_IT_WORKS.md        # This file — detailed explanation
```

---

## QUICK REFERENCE: Key Terms for Your Presentation

| Term | Simple Meaning |
|------|---------------|
| OHLCV | Open, High, Low, Close, Volume — the 5 daily stock numbers |
| Technical Indicator | Mathematical formula applied to stock prices to find patterns |
| Moving Average (MA) | Average price over last N days — shows trend direction |
| RSI | Score 0-100 showing if stock is overbought (>70) or oversold (<30) |
| MACD | Compares short and long term trends — shows momentum |
| Bollinger Bands | Upper/lower price limits based on volatility |
| Decision Tree | Flowchart of Yes/No questions that splits data into groups |
| Random Forest | 200 decision trees voting together for a more reliable prediction |
| Classifier | ML model that predicts categories (UP or DOWN) |
| Regressor | ML model that predicts numbers (% price change) |
| StandardScaler | Converts all indicators to the same scale so none dominates |
| Training Data | 80% of historical data used to teach the model |
| Testing Data | 20% of historical data used to check if the model learned correctly |
| Confusion Matrix | Table showing how many predictions were right vs wrong |
| Feature Importance | Ranking of which indicators the model found most useful |
| ROC Curve       | Graph showing model's ability to separate UP vs DOWN across thresholds |
| AUC             | Area Under ROC Curve; 0.5 = random, 1.0 = perfect |
| Trading Backtest | Simulated profit from following the model's predictions |
| .pkl file | Saved ML model that can be loaded later without retraining |
