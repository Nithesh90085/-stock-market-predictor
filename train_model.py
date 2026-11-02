# train_model.py
# Trains a Random Forest classifier and saves the model to disk.
# Run this ONCE to generate the saved model files.

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error,
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — fixes tkinter crash on Python 3.13
import matplotlib.pyplot as plt

from data_fetcher import fetch_stock_data, INDIAN_STOCKS
from feature_engineering import add_technical_indicators, create_target

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
PERIOD       = "5y"       # how much historical data to use
DAYS_AHEAD   = 1          # predict next-day direction
TEST_SIZE    = 0.2        # 20% data for testing
RANDOM_STATE = 42

FEATURE_COLS = [
    "MA_7", "MA_21", "MA_50",
    "EMA_12", "EMA_26",
    "MACD", "MACD_Signal",
    "RSI",
    "BB_Upper", "BB_Lower", "BB_Width",
    "Return_1d", "Return_5d",
    "Vol_Ratio", "HL_Range",
]

os.makedirs(MODELS_DIR, exist_ok=True)


def train_for_stock(ticker: str, stock_name: str):
    print(f"\n{'='*55}")
    print(f"  Training model for {stock_name} ({ticker})")
    print(f"{'='*55}")

    # 1. Fetch data
    df = fetch_stock_data(ticker, period=PERIOD)

    # 2. Feature engineering
    df = add_technical_indicators(df)
    df = create_target(df, days_ahead=DAYS_AHEAD)

    # 3. Prepare X and y
    X = df[FEATURE_COLS].values
    y = df["Target"].values          # 0 = DOWN, 1 = UP
    y_pct = df["Target_Pct"].values  # percentage price change (for regressor)

    # 4. Train / Test split (time-ordered — no shuffling!)
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    y_pct_train, y_pct_test = y_pct[:split], y_pct[split:]

    # 5. Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 6. Train Random Forest Classifier (predicts UP/DOWN direction)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    # 6b. Train Random Forest Regressor (predicts % price change)
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    regressor.fit(X_train_sc, y_pct_train)

    # 7. Evaluate classifier
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Classifier Test Accuracy : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # 7b. Evaluate regressor
    y_pct_pred = regressor.predict(X_test_sc)
    rmse = np.sqrt(mean_squared_error(y_pct_test, y_pct_pred))
    mae = mean_absolute_error(y_pct_test, y_pct_pred)
    print(f"\n  Regressor RMSE: {rmse:.4f}%  |  MAE: {mae:.4f}%")
    print(f"  (RMSE = avg error in % price change prediction)")

    # 8. ROC Curve + AUC score
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_sc)[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"\n  ROC-AUC Score : {roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#0ea5e9", lw=2.2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--",
            label="Random Classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#0ea5e9")
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{stock_name} — ROC Curve\n"
                 f"AUC = {roc_auc:.3f}  |  Accuracy = {acc*100:.1f}%",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = ticker.replace(".", "_")
    plt.savefig(os.path.join(MODELS_DIR, f"{safe_name}_roc.png"), dpi=100)
    plt.close()

    # 8b. Cumulative Trading Returns (backtest)
    # Simulate a simple strategy: buy if model predicts UP, sell/short if DOWN
    # Each day's return = actual % change × (+1 if predicted UP, -1 if predicted DOWN)
    strategy_returns = np.where(y_pred == 1, y_pct_test, -y_pct_test)
    cumulative_model   = np.cumsum(strategy_returns)
    cumulative_buyhold = np.cumsum(y_pct_test)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(cumulative_model,   color="#16a34a", lw=2.2, label="Model Strategy")
    ax2.plot(cumulative_buyhold, color="#f59e0b", lw=2.0, linestyle="--",
              label="Buy & Hold")
    ax2.axhline(0, color="#94a3b8", lw=1, linestyle=":")
    ax2.set_xlabel("Trading Days (Test Period)", fontsize=11)
    ax2.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax2.set_title(f"{stock_name} — Trading Strategy Backtest\n"
                  f"Model: {cumulative_model[-1]:+.2f}%  |  "
                  f"Buy&Hold: {cumulative_buyhold[-1]:+.2f}%", fontsize=12)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{safe_name}_returns.png"), dpi=100)
    plt.close()

    print(f"  Strategy Return : {cumulative_model[-1]:+.2f}%")
    print(f"  Buy & Hold      : {cumulative_buyhold[-1]:+.2f}%")

    # 9. Feature importance plot
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
    fi_df.sort_values("Importance", ascending=True, inplace=True)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.barh(fi_df["Feature"], fi_df["Importance"], color="steelblue")
    ax3.set_title(f"{stock_name} — Feature Importances")
    ax3.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{safe_name}_fi.png"), dpi=100)
    plt.close()

    # 10. Persist model + scaler + regressor
    joblib.dump(model,     os.path.join(MODELS_DIR, f"{safe_name}_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODELS_DIR, f"{safe_name}_scaler.pkl"))
    joblib.dump(regressor,  os.path.join(MODELS_DIR, f"{safe_name}_regressor.pkl"))
    print(f"  Saved model, scaler, regressor & charts to '{MODELS_DIR}/'")

    return acc, roc_auc


if __name__ == "__main__":
    results = {}
    for name, ticker in INDIAN_STOCKS.items():
        try:
            acc, roc_auc = train_for_stock(ticker, name)
            results[name] = f"Acc {acc*100:.2f}%  |  AUC {roc_auc:.3f}"
        except Exception as e:
            print(f"  ERROR for {name}: {e}")
            results[name] = "FAILED"

    print("\n\n" + "="*40)
    print("  TRAINING SUMMARY")
    print("="*40)
    for name, acc in results.items():
        print(f"  {name:<25} {acc}")
    print("="*40)
    print("\nAll models saved in the 'models/' folder.")
    print("Now run:  streamlit run app.py")
