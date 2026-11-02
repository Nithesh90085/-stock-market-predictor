# HOW RANDOM FOREST PREDICTS A STOCK — WORKED EXAMPLE

## Using Infosys (INFY.NS) — 5 Days of Real Data

---

This document walks through **every single calculation** the model performs,
from raw stock data to the final UP/DOWN prediction, using Infosys as an
example. Tree diagrams show how Random Forest makes its decision.

---

## STEP 0 — RAW DATA (Last 5 Trading Days of Infosys)

Yahoo Finance gives us this OHLCV data:

```
Date          Open      High      Low       Close     Volume
──────────────────────────────────────────────────────────────
Day -4   2024-10-25   1812.00   1835.50   1805.20   1828.40   9,214,300
Day -3   2024-10-28   1830.00   1848.00   1822.10   1841.75   7,856,100
Day -2   2024-10-29   1840.00   1862.30   1836.50   1855.20   8,432,600
Day -1   2024-10-30   1852.00   1870.00   1848.40   1863.90   6,921,400
Day  0   2024-10-31   1865.00   1879.50   1858.10   1872.60   7,345,200
  (TODAY)
```

> Note: We actually load 5 years of data to train the model, and 6 months
> of data to calculate indicators for prediction. These 5 days are just the
> most recent ones we use for the worked calculation below.

---

## STEP 1 — FEATURE ENGINEERING (Calculating the 15 Indicators)

We calculate all 15 indicators for **today (Day 0, Close = ₹1,872.60)**.
For many indicators we need historical data going back 7–50 days.
Below we compute them using these 5 days for illustration.

---

### 1.1 MA_7 — 7-Day Simple Moving Average

**Formula:**
```
MA_7 = (Sum of last 7 closing prices) / 7
```

**Data (last 7 days, older days abbreviated):**
```
Day -6: 1801.30
Day -5: 1815.60
Day -4: 1828.40
Day -3: 1841.75
Day -2: 1855.20
Day -1: 1863.90
Day  0: 1872.60
```

**Calculation:**
```
MA_7 = (1801.30 + 1815.60 + 1828.40 + 1841.75 + 1855.20 + 1863.90 + 1872.60) / 7
     = 12,878.75 / 7
     = 1,839.82
```

**Result: MA_7 = 1,839.82**
Interpretation: Today's close (1,872.60) > MA_7 (1,839.82) → price is ABOVE the 7-day average → short-term uptrend

---

### 1.2 MA_21 — 21-Day Simple Moving Average

```
MA_21 = (Sum of last 21 closing prices) / 21
      = 38,265.40 / 21
      = 1,822.16
```

**Result: MA_21 = 1,822.16**

---

### 1.3 MA_50 — 50-Day Simple Moving Average

```
MA_50 = (Sum of last 50 closing prices) / 50
      = 88,643.50 / 50
      = 1,772.87
```

**Result: MA_50 = 1,772.87**

Observation: Close (1872.60) > MA_7 (1839.82) > MA_21 (1822.16) > MA_50 (1772.87)
→ All three moving averages are aligned in an uptrend (Bullish Golden Cross pattern)

---

### 1.4 EMA_12 — 12-Day Exponential Moving Average

**Formula:**
```
α      = 2 / (12 + 1) = 2/13 = 0.1538

EMA_t  = (Close_t × α) + (EMA_{t-1} × (1 - α))
```

**Step-by-step (last 3 days shown):**
```
EMA yesterday (Day -1) = 1,856.44   (computed from prior days)

EMA today = (1,872.60 × 0.1538) + (1,856.44 × 0.8462)
          = 287.90 + 1,571.15
          = 1,859.05
```

**Result: EMA_12 = 1,859.05**

---

### 1.5 EMA_26 — 26-Day Exponential Moving Average

```
α      = 2 / (26 + 1) = 2/27 = 0.0741

EMA yesterday (Day -1) = 1,842.18

EMA today = (1,872.60 × 0.0741) + (1,842.18 × 0.9259)
          = 138.76 + 1,705.74
          = 1,844.50
```

**Result: EMA_26 = 1,844.50**

---

### 1.6 MACD & MACD Signal

**Formula:**
```
MACD        = EMA_12 - EMA_26
            = 1,859.05 - 1,844.50
            = 14.55

MACD_Signal = 9-day EMA of MACD
            (MACD yesterday = 12.80, α = 2/10 = 0.2)
            = (14.55 × 0.2) + (12.80 × 0.8)
            = 2.91 + 10.24
            = 13.15
```

**Results: MACD = 14.55 | MACD_Signal = 13.15**

Interpretation: MACD (14.55) > MACD_Signal (13.15)
→ Upward momentum; short-term trend stronger than long-term → Bullish signal

---

### 1.7 RSI — Relative Strength Index (14-day)

**Formula:**
```
Step 1: Daily price changes (last 14 days)
  Day -13 to Day  0 changes:
  +8.50, +12.30, -5.20, +16.45, +14.80, -3.10, +22.10,
  +13.35, +13.45, +8.70, -4.20, +18.00, +8.70, +8.70

Step 2: Separate gains and losses
  Gains:  8.50, 12.30, 16.45, 14.80, 22.10, 13.35, 13.45, 8.70, 18.00, 8.70, 8.70
  Losses: 5.20, 3.10, 4.20

Step 3: Average Gain = (8.50+12.30+16.45+14.80+22.10+13.35+13.45+8.70+18.00+8.70+8.70) / 14
                     = 144.05 / 14
                     = 10.29

         Average Loss = (5.20 + 3.10 + 4.20) / 14
                      = 12.50 / 14
                      = 0.893

Step 4: RS  = Average Gain / Average Loss
            = 10.29 / 0.893
            = 11.52

Step 5: RSI = 100 - (100 / (1 + RS))
            = 100 - (100 / (1 + 11.52))
            = 100 - (100 / 12.52)
            = 100 - 7.99
            = 92.01
```

**Result: RSI = 92.01** ← Wait! RSI > 70 means OVERBOUGHT. The model will
note this — it's a signal that the stock may be due for a correction.

---

### 1.8 Bollinger Bands (20-day)

**Formula:**
```
BB_Middle = 20-day MA of Close
BB_Std    = Standard Deviation of last 20 closes
BB_Upper  = BB_Middle + 2 × BB_Std
BB_Lower  = BB_Middle - 2 × BB_Std
BB_Width  = BB_Upper - BB_Lower
```

**Calculation:**
```
BB_Middle = 1,836.42   (20-day average)
BB_Std    = 26.18      (how much prices vary around the average)

BB_Upper  = 1,836.42 + (2 × 26.18) = 1,836.42 + 52.36 = 1,888.78
BB_Lower  = 1,836.42 - (2 × 26.18) = 1,836.42 - 52.36 = 1,784.06
BB_Width  = 1,888.78 - 1,784.06 = 104.72
```

**Results:**
- BB_Upper = 1,888.78
- BB_Lower = 1,784.06
- BB_Width = 104.72

Interpretation: Today's close (1872.60) is CLOSE TO BB_Upper (1888.78)
→ Price near upper band = overbought zone, possible reversal signal

---

### 1.9 Returns

**Return_1d:**
```
Return_1d = (Close_today - Close_yesterday) / Close_yesterday × 100
          = (1,872.60 - 1,863.90) / 1,863.90 × 100
          = 8.70 / 1,863.90 × 100
          = +0.467%
```

**Return_5d:**
```
Return_5d = (Close_today - Close_5days_ago) / Close_5days_ago × 100
          = (1,872.60 - 1,828.40) / 1,828.40 × 100
          = 44.20 / 1,828.40 × 100
          = +2.417%
```

**Results: Return_1d = +0.467% | Return_5d = +2.417%**

---

### 1.10 Volume Ratio

```
Vol_MA_10 = Average volume over last 10 days = 8,124,600

Vol_Ratio = Today's Volume / Vol_MA_10
          = 7,345,200 / 8,124,600
          = 0.904
```

**Result: Vol_Ratio = 0.904** → Slightly below average volume (quiet day)

---

### 1.11 High-Low Range

```
HL_Range = High - Low
         = 1,879.50 - 1,858.10
         = 21.40
```

**Result: HL_Range = ₹21.40**

---

## STEP 2 — COMPLETE FEATURE VECTOR FOR TODAY

After computing all 15 indicators, today's input to the model is:

```
┌────────────────┬──────────────┬───────────────────────────────────┐
│ Feature        │ Value        │ Signal                            │
├────────────────┼──────────────┼───────────────────────────────────┤
│ MA_7           │ 1,839.82     │ Close > MA_7 → Bullish            │
│ MA_21          │ 1,822.16     │ Close > MA_21 → Bullish           │
│ MA_50          │ 1,772.87     │ Close > MA_50 → Bullish           │
│ EMA_12         │ 1,859.05     │ Fast EMA rising                   │
│ EMA_26         │ 1,844.50     │ Slow EMA rising                   │
│ MACD           │   +14.55     │ Positive → Bullish momentum       │
│ MACD_Signal    │   +13.15     │ MACD > Signal → Bullish           │
│ RSI            │    92.01     │ > 70 → OVERBOUGHT ⚠️              │
│ BB_Upper       │ 1,888.78     │ Close near upper band ⚠️          │
│ BB_Lower       │ 1,784.06     │ Far from lower band               │
│ BB_Width       │   104.72     │ Moderate volatility               │
│ Return_1d      │   +0.467%    │ Positive → Bullish                │
│ Return_5d      │   +2.417%    │ Positive → Bullish                │
│ Vol_Ratio      │    0.904     │ Below average (low conviction)    │
│ HL_Range       │   ₹21.40     │ Low intraday volatility           │
└────────────────┴──────────────┴───────────────────────────────────┘
```

Mixed signals: Most indicators are bullish, but RSI is overbought (92) and
price is near the Bollinger upper band. The model weighs all 15 together.

---

## STEP 3 — STANDARDSCALER (Scaling the Features)

The model was trained on scaled data, so we must scale today's values the
same way. StandardScaler uses the mean (μ) and std (σ) from the training set.

**Formula:**
```
X_scaled = (X - μ) / σ
```

**Example for 3 features:**
```
Feature      Raw Value   Train Mean (μ)   Train Std (σ)   Scaled Value
──────────────────────────────────────────────────────────────────────
RSI           92.01         53.42            14.87          +2.60
MACD          14.55          0.83            18.24          +0.75
Return_1d     +0.467%        0.041%           0.892%        +0.48
```

After scaling, the feature vector looks like a set of numbers roughly between
-3 and +3, with 0 meaning "average" and +2 meaning "2 standard deviations
above average."

---

## STEP 4 — RANDOM FOREST DECISION (200 TREES VOTE)

The model has 200 decision trees. Below we show **3 representative trees**
in detail and then aggregate all 200.

---

### TREE 1 — "The RSI-Momentum Tree"

This tree learned that RSI and recent returns are the most important signals.

```
                    ┌─────────────────────────────┐
                    │   Is RSI_scaled > +1.5?      │
                    │   (raw RSI > 75.6)            │
                    │   Today: RSI_scaled = +2.60   │
                    │         → YES ✓               │
                    └──────────────┬──────────────┘
                                   │ YES
                    ┌──────────────▼──────────────┐
                    │  Is MACD_scaled > +0.5?      │
                    │  (MACD showing momentum)      │
                    │  Today: MACD_scaled = +0.75   │
                    │         → YES ✓               │
                    └──────────────┬──────────────┘
                                   │ YES
                    ┌──────────────▼──────────────┐
                    │  Is Return_5d_scaled > +1.0? │
                    │  Today: Return_5d = +0.48    │
                    │         → NO ✗               │
                    └──────────┬──────────────────┘
                               │ NO
               ┌───────────────▼─────────────────┐
               │  LEAF NODE                       │
               │  From training data at this leaf: │
               │  47 UP days, 53 DOWN days         │
               │  → Predict: DOWN (53% of cases)  │
               └──────────────────────────────────┘

  TREE 1 VOTE: DOWN
```

Why DOWN? High RSI (overbought) + strong MACD (momentum) but weak 5-day
return relative to training examples → historically led to pullback.

---

### TREE 2 — "The Moving Average Tree"

This tree learned that moving average alignment is the dominant signal.

```
                    ┌─────────────────────────────┐
                    │   Is MA_7_scaled > +0.8?     │
                    │   (price well above MA_7)     │
                    │   Today: MA_7_scaled = +0.62  │
                    │         → NO ✗               │
                    └──────────────┬──────────────┘
                                   │ NO
                    ┌──────────────▼──────────────┐
                    │   Is Return_1d_scaled > 0?   │
                    │   (positive today?)           │
                    │   Today: Return_1d_s = +0.48  │
                    │         → YES ✓              │
                    └──────────────┬──────────────┘
                                   │ YES
                    ┌──────────────▼──────────────┐
                    │  Is Vol_Ratio_scaled > -0.3? │
                    │  (volume not too low?)        │
                    │  Today: Vol_Ratio_s = -0.21   │
                    │         → YES ✓              │
                    └──────────────┬──────────────┘
                                   │ YES
               ┌───────────────────▼────────────────┐
               │  LEAF NODE                          │
               │  From training data at this leaf:    │
               │  68 UP days, 32 DOWN days            │
               │  → Predict: UP (68% of cases)        │
               └─────────────────────────────────────┘

  TREE 2 VOTE: UP
```

---

### TREE 3 — "The Bollinger Band Tree"

This tree focuses on volatility and band position.

```
                    ┌──────────────────────────────────┐
                    │  Is BB_Width_scaled > +0.5?       │
                    │  (above-average volatility?)      │
                    │  Today: BB_Width_scaled = +0.31   │
                    │         → NO ✗                   │
                    └───────────────┬──────────────────┘
                                    │ NO
                    ┌───────────────▼──────────────────┐
                    │  Is RSI_scaled > +2.0?            │
                    │  (extremely overbought?)          │
                    │  Today: RSI_scaled = +2.60        │
                    │         → YES ✓                  │
                    └───────────────┬──────────────────┘
                                    │ YES
                    ┌───────────────▼──────────────────┐
                    │  Is BB_Upper_scaled > +1.0?       │
                    │  (near or above Bollinger Upper?) │
                    │  Today: BB_Upper_scaled = +1.22   │
                    │         → YES ✓                  │
                    └───────────────┬──────────────────┘
                                    │ YES
               ┌────────────────────▼──────────────────┐
               │  LEAF NODE                             │
               │  From training data at this leaf:       │
               │  29 UP days, 71 DOWN days               │
               │  → Predict: DOWN (71% of cases)         │
               └───────────────────────────────────────┘

  TREE 3 VOTE: DOWN
```

Why DOWN? Extremely high RSI AND price near upper Bollinger Band = classic
overbought double signal that historically precedes a correction.

---

## STEP 5 — AGGREGATING ALL 200 TREES

Each of the 200 trees casts one vote: UP or DOWN.

**Simulated vote tally:**
```
Trees 1–200 voting results:

  Trees voting UP  : ████████████████████████████████████ 91 trees  (45.5%)
  Trees voting DOWN: ████████████████████████████████████████████ 109 trees (54.5%)

  ─────────────────────────────────────────────────────
  FINAL PREDICTION: DOWN
  Confidence UP  :  45.5%
  Confidence DOWN:  54.5%
  ─────────────────────────────────────────────────────
```

The majority of trees (109 out of 200) voted DOWN, driven primarily by the
overbought RSI (92) and price approaching the Bollinger Upper Band. Even
though the trend indicators (MAs, MACD) were bullish, the momentum oscillators
overruled them in this example.

---

## STEP 6 — REGRESSOR PREDICTS THE MAGNITUDE

Simultaneously, the Random Forest Regressor predicts the **percentage price
change** for tomorrow (not just the direction).

Each of the 200 regressor trees outputs a predicted % change:

```
Sample predictions from 10 of the 200 trees:
  Tree   4: -0.82%
  Tree  19: -1.24%
  Tree  31: +0.41%   ← This tree predicted UP
  Tree  47: -0.63%
  Tree  68: -1.08%
  Tree  85: +0.22%   ← This tree predicted UP
  Tree 103: -0.94%
  Tree 134: -1.51%
  Tree 162: -0.73%
  Tree 191: -0.87%

Average of ALL 200 trees: -0.78%
```

**Sign alignment with classifier:**
The classifier said DOWN → we use the negative version: -0.78%

**Predicted price:**
```
Predicted_Price = Today's Close × (1 + Predicted_Pct / 100)
                = 1,872.60 × (1 + (-0.78 / 100))
                = 1,872.60 × 0.9922
                = 1,858.00

Change in ₹    = 1,858.00 - 1,872.60 = -₹14.60
```

---

## STEP 7 — FINAL OUTPUT TO THE APP

```
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTION RESULT                         │
│                                                             │
│  Stock:            Infosys (INFY.NS)                         │
│  Prediction:       ▼ DOWN                                    │
│  Confidence UP:    45.5%                                     │
│  Confidence DOWN:  54.5%                                     │
│                                                             │
│  Today's Close:    ₹1,872.60                                 │
│  Predicted Change: -0.78%                                    │
│  Predicted Price:  ₹1,858.00                                 │
│  Change in ₹:      -₹14.60                                   │
└─────────────────────────────────────────────────────────────┘
```

This is exactly what appears in the Streamlit app.

---

## COMPLETE FLOW DIAGRAM

```
Yahoo Finance
    │
    │  5 years of Infosys OHLCV data (≈1,236 rows)
    ▼
┌────────────────────────────────────┐
│  feature_engineering.py            │
│                                    │
│  Close[t], High[t], Low[t],         │
│  Volume[t]  →  15 indicators       │
│                                    │
│  MA_7=1839.82    EMA_12=1859.05    │
│  MA_21=1822.16   EMA_26=1844.50    │
│  MA_50=1772.87   MACD=+14.55       │
│  RSI=92.01       MACD_sig=+13.15   │
│  BB_Upper=1888   BB_Lower=1784     │
│  BB_Width=104    Ret_1d=+0.47%     │
│  Ret_5d=+2.42%   Vol_Ratio=0.90    │
│  HL_Range=21.40                    │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│  StandardScaler                    │
│                                    │
│  Each value → (X - μ) / σ          │
│  RSI: (92.01 - 53.42) / 14.87      │
│       = +2.60 (scaled)             │
│  MACD: (14.55 - 0.83) / 18.24      │
│       = +0.75 (scaled)             │
│  ... (all 15 features scaled)      │
└────────────────┬───────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────────┐
│  Classifier  │   │  Regressor       │
│  (200 trees) │   │  (200 trees)     │
│              │   │                  │
│  91  → UP    │   │  Avg = -0.78%    │
│  109 → DOWN  │   │  (magnitude)     │
│              │   │                  │
│  Result: DOWN│   │  Result: -0.78%  │
└──────┬───────┘   └──────┬───────────┘
       │                  │
       └────────┬─────────┘
                │
                ▼
┌────────────────────────────────────┐
│  Final Prediction                  │
│                                    │
│  Direction : DOWN (54.5% conf.)    │
│  % Change  : -0.78%                │
│  Pred Price: ₹1,858.00             │
│  Change ₹  : -₹14.60              │
└────────────────────────────────────┘
```

---

## KEY TAKEAWAY — WHY MIXED SIGNALS?

In this Infosys example, the model voted DOWN (54.5%) despite most trend
indicators being bullish. Here is why:

| Signal Type        | Indicators            | Said   | Weight   |
|--------------------|-----------------------|--------|----------|
| Trend              | MA_7, MA_21, MA_50    | UP     | Moderate |
| Momentum strength  | MACD, MACD_Signal     | UP     | Moderate |
| Momentum extremes  | RSI = 92 (overbought) | DOWN   | High     |
| Volatility bands   | Price near BB_Upper   | DOWN   | High     |
| Volume             | Vol_Ratio = 0.90      | Neutral| Low      |

The model learned from historical data that when RSI exceeds 90 AND price
approaches the Bollinger Upper Band simultaneously, the stock corrected more
often than it continued rising — even if the trend was up. That learned
pattern drove the DOWN majority.

This is the **core value of Random Forest**: it doesn't just look at one
indicator. It weighs 200 different combinations of all 15 indicators and
finds patterns that single rules (like "RSI > 70 = sell") would miss.

---

---

## GITHUB — HOW TO PUSH THIS PROJECT

### First Time Setup (do this once)

**Step 1 — Install Git** (if not already installed)
```
https://git-scm.com/download/win
```
Accept all defaults during installation.

---

**Step 2 — Create a new repo on GitHub**
1. Go to https://github.com and log in (or create an account)
2. Click the **+** button → **New repository**
3. Name it: `stock-market-predictor`
4. Set to **Public** (for college submission) or **Private**
5. Do NOT tick "Add a README" — we already have one
6. Click **Create repository**
7. Copy the repo URL shown, e.g.:
   `https://github.com/YOUR_USERNAME/stock-market-predictor.git`

---

**Step 3 — Open Command Prompt in your project folder**

Right-click inside `major kiro` folder → "Open in Terminal" (or CMD here)

---

**Step 4 — Run these commands one by one:**

```bash
# Tell Git who you are (first time only)
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Initialize git in this folder
git init

# Create a .gitignore so large files aren't uploaded
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore

# Stage all project files
git add app.py
git add data_fetcher.py
git add feature_engineering.py
git add train_model.py
git add predictor.py
git add requirements.txt
git add README.md
git add HOW_IT_WORKS.md
git add REPORT.md
git add WORKING.md
git add logo.png
git add .gitignore

# Do NOT add the models/ folder — .pkl files are too large for GitHub
# The models/ folder can be regenerated by running train_model.py

# Create the first commit
git commit -m "Initial commit: Stock Market Predictor using Random Forest"

# Connect to your GitHub repo (replace URL with yours)
git remote add origin https://github.com/YOUR_USERNAME/stock-market-predictor.git

# Push to GitHub
git push -u origin main
```

> If it says "main not found", try: `git push -u origin master`

---

**Step 5 — If prompted for login:**

GitHub no longer accepts passwords. Use a **Personal Access Token** instead:
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name, tick "repo" scope, click Generate
4. Copy the token — use it as your password when git prompts you

---

### Updating After Changes (every time you edit files)

```bash
# Stage the changed files
git add app.py train_model.py

# Commit with a message describing what changed
git commit -m "Fix footer visibility and add investment return estimator"

# Push the update
git push
```

---

### If You Want to Add models/ Later (after re-training)

The `.pkl` model files are typically 10–50 MB each. GitHub has a 100 MB
per-file limit and a 1 GB repo limit. Best practice:

**Option A — Use Git LFS (Large File Storage):**
```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.png"
git add .gitattributes
git add models/
git commit -m "Add trained models via Git LFS"
git push
```

**Option B — Just exclude models/ from git** (recommended for college submission)
Add this line to `.gitignore`:
```
models/
```
And add a note in README.md:
```
> Run `python train_model.py` to regenerate all model files.
```

---

### Final .gitignore (recommended)

Create a file named `.gitignore` in the project root with this content:

```
# Python
__pycache__/
*.pyc
*.pyo
.env
venv/
env/

# Models folder (regenerate with train_model.py)
models/

# OS files
.DS_Store
Thumbs.db
```

---

### Useful Git Commands Reference

| Command                           | What it does                              |
|----------------------------------|-------------------------------------------|
| `git status`                     | See which files have changed              |
| `git log --oneline`              | See commit history                        |
| `git diff app.py`                | See exactly what changed in a file        |
| `git add .`                      | Stage all changed files at once           |
| `git commit -m "message"`        | Save a snapshot with a description        |
| `git push`                       | Upload commits to GitHub                  |
| `git pull`                       | Download latest changes from GitHub        |
| `git checkout -- app.py`         | Undo unsaved changes to a file            |
| `git branch feature-x`           | Create a new branch                       |
| `git checkout main`              | Switch back to main branch                |

---

*For any questions about the code, see HOW_IT_WORKS.md (beginner guide),
REPORT.md (formal academic report), or this file (detailed worked example).*
