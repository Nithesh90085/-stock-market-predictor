# app.py
# Streamlit web app — Stock Market Predictor
# Run with:  streamlit run app.py

import os
import base64
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data_fetcher import fetch_stock_data, INDIAN_STOCKS
from feature_engineering import add_technical_indicators
from predictor import predict_next_day

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────────
# Version stamp — bump this any time prediction logic changes to force a cache clear
APP_VERSION = "v4"
if st.session_state.get("_version") != APP_VERSION:
    for key in ["page", "prediction_result", "selected_stock", "chart_data", "return_data"]:
        st.session_state.pop(key, None)
    st.session_state["_version"] = APP_VERSION

if "page" not in st.session_state:
    st.session_state.page = "home"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None
if "chart_data" not in st.session_state:
    st.session_state.chart_data = None
if "return_data" not in st.session_state:
    st.session_state.return_data = None


# ── Helper: load logo as base64 ────────────────────────────────────────────────
def get_logo_b64(path="logo.png") -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


LOGO_B64 = get_logo_b64()
LOGO_TAG = (
    f'<img src="data:image/png;base64,{LOGO_B64}" '
    'style="width:110px; height:auto; border-radius:14px; margin-bottom:16px;" />'
    if LOGO_B64
    else '<div style="font-size:3rem;">📈</div>'
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Base ── */
.stApp { background-color: #f4f6f9; }

/* ── Kill sidebar ── */
section[data-testid="stSidebar"] { display: none !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Nav bar ── */
.nav-bar {
    display: flex;
    gap: 8px;
    padding: 10px 0 16px 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.nav-link {
    display: inline-block;
    padding: 7px 20px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    color: #475569 !important;
    text-decoration: none !important;
    font-size: 0.84rem;
    font-weight: 500;
}
.nav-link:hover {
    background: #0ea5e9 !important;
    border-color: #0ea5e9 !important;
    color: #ffffff !important;
    text-decoration: none !important;
}

/* ── Prediction banners ── */
.pred-banner {
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    margin: 10px 0 16px 0;
}
.pred-up   { background: linear-gradient(135deg,#dcfce7,#bbf7d0); border:2px solid #16a34a; }
.pred-down { background: linear-gradient(135deg,#fee2e2,#fecaca); border:2px solid #dc2626; }
.pred-icon { font-size: 2.2rem; }
.pred-up   .pred-icon { color: #15803d; }
.pred-down .pred-icon { color: #b91c1c; }
.pred-up-text   { font-size:1.3rem; font-weight:700; color:#15803d; margin-top:6px; }
.pred-down-text { font-size:1.3rem; font-weight:700; color:#b91c1c; margin-top:6px; }

/* ── Custom metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-label  { font-size:0.78rem; color:#64748b; font-weight:500; text-transform:uppercase; letter-spacing:.04em; margin-bottom:6px; }
.metric-value  { font-size:1.45rem; font-weight:700; color:#0f172a; }
.metric-delta-up   { font-size:0.9rem; font-weight:600; color:#15803d; margin-top:4px; }
.metric-delta-down { font-size:0.9rem; font-weight:600; color:#b91c1c; margin-top:4px; }
.metric-delta-neutral { font-size:0.9rem; font-weight:600; color:#64748b; margin-top:4px; }

/* ── Confidence bars ── */
.conf-bar-bg {
    background: #e2e8f0;
    border-radius: 8px;
    height: 28px;
    overflow: hidden;
    margin-top: 4px;
    margin-bottom: 10px;
}
.conf-fill-up {
    background: linear-gradient(90deg,#15803d,#22c55e);
    height:100%; border-radius:8px;
    display:flex; align-items:center; padding-left:12px;
    color:#fff; font-weight:600; font-size:0.8rem;
    min-width: 48px;
}
.conf-fill-down {
    background: linear-gradient(90deg,#b91c1c,#ef4444);
    height:100%; border-radius:8px;
    display:flex; align-items:center; padding-left:12px;
    color:#fff; font-weight:600; font-size:0.8rem;
    min-width: 48px;
}

/* ── Term help ── */
.term-help { font-size:0.78rem; color:#475569; font-style:italic; }

/* ── Investment section labels ── */
.section-desc {
    font-size:0.88rem;
    color:#334155;
    font-weight:500;
    margin-bottom:10px;
}
.invest-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── Stock header ── */
.stock-header { font-size:1.3rem; font-weight:600; color:#0f172a; margin-bottom:2px; }
.stock-sub    { font-size:0.85rem; color:#64748b; margin-bottom:4px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Glossary ──────────────────────────────────────────────────────────────────
TERM_HELP = {
    "RSI (14)":         "Momentum 0-100; >70 overbought, <30 oversold",
    "MACD":             "Trend indicator; positive = upward momentum",
    "MACD Signal":      "Trigger line; MACD crossing above it = bullish",
    "MA 7-day":         "Short-term average price (1 week)",
    "MA 21-day":        "Medium-term average price (~1 month)",
    "MA 50-day":        "Long-term trend benchmark (~2.5 months)",
    "1-Day Return":     "% change vs yesterday's close",
    "5-Day Return":     "% change over last 5 trading days",
    "Bollinger Upper":  "Upper volatility band; price near it = overbought zone",
    "Bollinger Lower":  "Lower volatility band; price near it = oversold zone",
    "Bollinger Width":  "Band spread; wider = more volatile",
    "Volume Ratio":     "Today's volume vs 10-day average; >1 = high activity",
    "High-Low Range":   "Intraday price spread (high minus low in ₹)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # ── Logo + title + description ──
    st.markdown(
        f"""
        <div style="text-align:center; padding:60px 0 20px 0;">
            {LOGO_TAG}
            <h1 style="font-size:2.1rem; font-weight:700; color:#0f172a; margin:0 0 10px 0;">
                Stock Market Predictor
            </h1>
            <p style="color:#64748b; font-size:0.96rem; max-width:500px;
                      margin:0 auto 0 auto; line-height:1.6;">
                Analyses 15 technical indicators using a Random Forest model to estimate
                whether a stock will move <span style="color:#15803d;font-weight:600;">up</span>
                or <span style="color:#b91c1c;font-weight:600;">down</span>
                on the next trading day.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Stock selector ──
    col_l, col_sel, col_r = st.columns([1, 2, 1])
    with col_sel:
        st.markdown(
            '<p style="font-size:0.92rem; font-weight:600; color:#1e293b; '
            'margin-bottom:6px;">Select Stock</p>',
            unsafe_allow_html=True,
        )
        selected_name = st.selectbox(
            label="Select Stock",
            options=list(INDIAN_STOCKS.keys()),
            index=0,
            label_visibility="collapsed",
        )

    st.markdown("")

    # ── Predict button ──
    col_l2, col_btn, col_r2 = st.columns([1, 2, 1])
    with col_btn:
        if st.button("Predict Next Trading Day →", type="primary", use_container_width=True):
            try:
                ticker = INDIAN_STOCKS[selected_name]
                with st.spinner(f"Fetching data for {selected_name}…"):
                    result = predict_next_day(ticker)
                    df_chart = fetch_stock_data(ticker, period="6mo")
                    df_chart = add_technical_indicators(df_chart)
                    df_hist  = fetch_stock_data(ticker, period="5y")
                st.session_state.prediction_result = result
                st.session_state.chart_data = df_chart
                st.session_state.return_data = df_hist
                st.session_state.selected_stock = selected_name
                st.session_state.page = "results"
                st.rerun()
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Run `python train_model.py` first to train the models.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # ── Footer line ──
    st.markdown("")
    st.markdown(
        '<p style="text-align:center; color:#94a3b8; font-size:0.78rem; margin-top:30px;">'
        "Random Forest · 200 trees · 15 technical indicators · Yahoo Finance (NSE)"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    result = st.session_state.prediction_result
    selected_name = st.session_state.selected_stock
    df_chart = st.session_state.chart_data

    if result is None or selected_name is None:
        st.session_state.page = "home"
        st.rerun()

    ticker = INDIAN_STOCKS[selected_name]

    # ── Top bar ──
    bc1, bc2 = st.columns([1, 7])
    with bc1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with bc2:
        st.markdown(
            f'<div class="stock-header">{selected_name}</div>'
            f'<div class="stock-sub">{ticker} &nbsp;·&nbsp; Next trading day prediction</div>',
            unsafe_allow_html=True,
        )

    # ── Nav bar ──
    st.markdown(
        """
        <div class="nav-bar">
            <a href="#prediction"   class="nav-link">Prediction</a>
            <a href="#chart"        class="nav-link">Price Chart</a>
            <a href="#performance"  class="nav-link">Model Performance</a>
            <a href="#returns"      class="nav-link">Investment Returns</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  PREDICTION
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Prediction", anchor="prediction")

    pred          = result["prediction"]
    latest_close  = result["latest_close"]
    pred_price    = result["predicted_price"]
    pred_pct      = result["predicted_pct"]
    change_rs     = result["change_rupees"]
    conf_up       = result["confidence_up"]
    conf_down     = result["confidence_down"]

    # Big banner
    if pred == "UP":
        st.markdown(
            """
            <div class="pred-banner pred-up">
                <div class="pred-icon">▲</div>
                <div class="pred-up-text">Price expected to go UP tomorrow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="pred-banner pred-down">
                <div class="pred-icon">▼</div>
                <div class="pred-down-text">Price expected to go DOWN tomorrow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Custom metric cards (avoids Streamlit's wrong auto-coloring) ──
    def metric_card(label: str, value: str, delta: str, is_up: bool | None) -> str:
        if is_up is True:
            delta_cls = "metric-delta-up"
            arrow = "▲ "
        elif is_up is False:
            delta_cls = "metric-delta-down"
            arrow = "▼ "
        else:
            delta_cls = "metric-delta-neutral"
            arrow = ""
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="{delta_cls}">{arrow}{delta}</div>
        </div>
        """

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            metric_card("Today's Close", f"₹{latest_close:,.2f}", "Current price", None),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            metric_card(
                "Predicted Change",
                f"{pred_pct:+.2f}%",
                f"₹{abs(change_rs):,.2f} {'gain' if change_rs >= 0 else 'loss'}",
                pred_pct >= 0,
            ),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            metric_card(
                "Predicted Price",
                f"₹{pred_price:,.2f}",
                f"{pred_pct:+.2f}% from today",
                pred_pct >= 0,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Confidence bars ──
    st.markdown("##### Confidence Level")
    st.markdown(
        '<span class="term-help">Probability assigned by the model for each direction. '
        "The higher the bar, the more confident the model is.</span>",
        unsafe_allow_html=True,
    )

    st.markdown(f"**▲ UP &nbsp;&nbsp;&nbsp; {conf_up:.1f}%**")
    st.markdown(
        f'<div class="conf-bar-bg">'
        f'<div class="conf-fill-up" style="width:{conf_up}%">{conf_up:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**▼ DOWN &nbsp; {conf_down:.1f}%**")
    st.markdown(
        f'<div class="conf-bar-bg">'
        f'<div class="conf-fill-down" style="width:{conf_down}%">{conf_down:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Summary ──
    st.info(
        f"Based on today's technical indicators, the model expects **{selected_name}** to close "
        f"at approximately **₹{pred_price:,.2f}** tomorrow — a **{pred_pct:+.2f}%** change from "
        f"today's close of ₹{latest_close:,.2f}. "
        f"Model confidence: **{conf_up:.1f}%** UP / **{conf_down:.1f}%** DOWN."
    )

    # ── Technical indicators table ──
    st.markdown("##### Technical Indicators Used for This Prediction")
    st.markdown(
        '<span class="term-help">Values calculated from the latest available trading day.</span>',
        unsafe_allow_html=True,
    )

    ind = result["indicators"]
    key_ind_map = {
        "RSI (14)":         ind["RSI"],
        "MACD":             ind["MACD"],
        "MACD Signal":      ind["MACD_Signal"],
        "MA 7-day":         ind["MA_7"],
        "MA 21-day":        ind["MA_21"],
        "MA 50-day":        ind["MA_50"],
        "1-Day Return":     ind["Return_1d"],
        "5-Day Return":     ind["Return_5d"],
        "Bollinger Upper":  ind["BB_Upper"],
        "Bollinger Lower":  ind["BB_Lower"],
        "Bollinger Width":  ind["BB_Width"],
        "Volume Ratio":     ind["Vol_Ratio"],
        "High-Low Range":   ind["HL_Range"],
    }
    ind_df = pd.DataFrame(
        [
            {
                "Indicator":    name,
                "Value":        round(val, 4),
                "What it means": TERM_HELP.get(name, ""),
            }
            for name, val in key_ind_map.items()
        ]
    )
    st.dataframe(ind_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  PRICE CHART
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("")
    st.subheader("Price Chart — Last 6 Months", anchor="chart")

    # Compute a clean Y-axis range with visible padding
    y_min = float(df_chart["Low"].min())
    y_max = float(df_chart["High"].max())
    y_pad = (y_max - y_min) * 0.05
    y_range = [y_min - y_pad, y_max + y_pad]

    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name="Price",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            increasing_fillcolor="#16a34a",
            decreasing_fillcolor="#dc2626",
        )
    )

    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=df_chart.index, y=df_chart["MA_7"],
            mode="lines", name="MA 7-day",
            line=dict(color="#f59e0b", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_chart.index, y=df_chart["MA_21"],
            mode="lines", name="MA 21-day",
            line=dict(color="#6366f1", width=1.8),
        )
    )

    fig.update_layout(
        # Title left-aligned so it never overlaps the legend
        title=dict(
            text=f"<b>{selected_name}</b>  ·  {ticker}  ·  Daily Candlestick (6 months)",
            font=dict(size=15, color="#1e293b"),
            x=0,          # left-aligned
            xanchor="left",
            pad=dict(l=10),
        ),
        # Legend positioned top-right, inside the plot area but not overlapping title
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,        # push below the chart
            xanchor="left",
            x=0,
            font=dict(size=12, color="#475569"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        yaxis=dict(
            title=dict(text="Price (₹)", font=dict(size=13, color="#475569")),
            tickfont=dict(size=13, color="#1e293b"),
            gridcolor="#f1f5f9",
            showgrid=True,
            tickformat=",.0f",          # e.g. 1,265 instead of 1.265k
            exponentformat="none",
            range=y_range,
            side="right",               # prices on the right side = easier to read
            automargin=True,
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=13, color="#475569")),
            tickfont=dict(size=12, color="#475569"),
            gridcolor="#f1f5f9",
            showgrid=True,
            automargin=True,
        ),
        height=560,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=80, t=70, b=80),  # right margin for Y-axis labels
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<span class="term-help">'
        "🟢 Green candle = closed higher than opened &nbsp;|&nbsp; "
        "🔴 Red candle = closed lower than opened &nbsp;|&nbsp; "
        "— Orange = 7-day MA &nbsp;|&nbsp; — Indigo = 21-day MA"
        "</span>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  MODEL PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("")
    st.subheader("Model Performance", anchor="performance")

    safe_name = ticker.replace(".", "_")
    roc_path     = os.path.join("models", f"{safe_name}_roc.png")
    returns_path = os.path.join("models", f"{safe_name}_returns.png")
    fi_path      = os.path.join("models", f"{safe_name}_fi.png")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**ROC Curve — Model Discrimination**")
        st.markdown(
            '<span class="term-help">Shows how well the model separates UP from DOWN predictions. '
            "AUC (Area Under Curve) closer to 1.0 = better. A random guess = 0.5.</span>",
            unsafe_allow_html=True,
        )
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("Not found. Run `python train_model.py` first.")
    with pcol2:
        st.markdown("**Trading Strategy Backtest**")
        st.markdown(
            '<span class="term-help">Cumulative returns if you followed the model\'s predictions '
            "(green) vs simply buying and holding the stock (orange).</span>",
            unsafe_allow_html=True,
        )
        if os.path.exists(returns_path):
            st.image(returns_path, use_container_width=True)
        else:
            st.info("Not found. Run `python train_model.py` first.")

    st.markdown("")
    pcol3, pcol4 = st.columns([2, 1])
    with pcol3:
        st.markdown("**Feature Importance**")
        st.markdown(
            '<span class="term-help">Which technical indicators the model relies on most.</span>',
            unsafe_allow_html=True,
        )
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Not found. Run `python train_model.py` first.")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    #  INVESTMENT RETURN ESTIMATOR
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Investment Return Estimator", anchor="returns")
    st.markdown(
        '<p class="section-desc">'
        "See how much your investment in this stock could grow based on its "
        "historical performance. Adjust the amount below to see projections for "
        "1, 3, and 5 years."
        "</p>",
        unsafe_allow_html=True,
    )

    df_hist = st.session_state.get("return_data")

    if df_hist is not None and not df_hist.empty:
        close_hist = df_hist["Close"].squeeze()

        # ── Calculate CAGR (Compound Annual Growth Rate) ──
        start_price = float(close_hist.iloc[0])
        end_price   = float(close_hist.iloc[-1])
        n_days      = len(close_hist)
        n_years     = n_days / 252  # ~252 trading days per year

        total_return_pct = ((end_price - start_price) / start_price) * 100

        if n_years > 0 and end_price > 0 and start_price > 0:
            cagr = ((end_price / start_price) ** (1 / n_years) - 1) * 100
        else:
            cagr = 0.0

        # ── Calculate yearly returns for best/worst year ──
        yearly = close_hist.resample("YE").last()
        yearly_ret = yearly.pct_change().dropna() * 100
        if len(yearly_ret) > 0:
            best_year = float(yearly_ret.max())
            worst_year = float(yearly_ret.min())
        else:
            best_year = 0.0
            worst_year = 0.0

        avg_daily_ret = float(close_hist.pct_change().mean() * 100)

        # ── Investment slider with professional label ──
        st.markdown(
            '<p style="font-size:0.92rem; font-weight:600; color:#1e293b; '
            'margin-bottom:6px;">Select Investment Amount</p>',
            unsafe_allow_html=True,
        )
        invest_amount = st.slider(
            "Investment Amount (₹)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            label_visibility="collapsed",
        )
        cagr_color = "#15803d" if cagr >= 0 else "#b91c1c"
        st.markdown(
            f'<p style="font-size:0.85rem; color:#334155; margin-top:-6px;">'
            f"Investing <b>₹{invest_amount:,.0f}</b> today "
            f"at the current CAGR of <b style='color:{cagr_color};'>"
            f"{cagr:+.2f}%/yr</b>"
            f"</p>",
            unsafe_allow_html=True,
        )

        # ── Projected values based on CAGR ──
        cagr_frac = cagr / 100
        val_1y  = invest_amount * (1 + cagr_frac) ** 1
        val_3y  = invest_amount * (1 + cagr_frac) ** 3
        val_5y  = invest_amount * (1 + cagr_frac) ** 5

        profit_1y = val_1y - invest_amount
        profit_3y = val_3y - invest_amount
        profit_5y = val_5y - invest_amount

        profit_color_1 = "#15803d" if profit_1y >= 0 else "#b91c1c"
        profit_color_3 = "#15803d" if profit_3y >= 0 else "#b91c1c"
        profit_color_5 = "#15803d" if profit_5y >= 0 else "#b91c1c"
        arrow_1 = "▲" if profit_1y >= 0 else "▼"
        arrow_3 = "▲" if profit_3y >= 0 else "▼"
        arrow_5 = "▲" if profit_5y >= 0 else "▼"

        # ── Projected returns label ──
        st.markdown(
            '<p style="font-size:0.92rem; font-weight:600; color:#1e293b; '
            'margin-top:18px; margin-bottom:10px;">Projected Returns</p>',
            unsafe_allow_html=True,
        )

        # ── Projected returns cards (using invest-card for better styling) ──
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""
                <div class="invest-card">
                    <div class="metric-label">After 1 Year</div>
                    <div class="metric-value">₹{val_1y:,.0f}</div>
                    <div style="font-size:0.9rem; font-weight:600; color:{profit_color_1}; margin-top:4px;">
                        {arrow_1} ₹{abs(profit_1y):,.0f} ({cagr:+.1f}%/yr)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="invest-card">
                    <div class="metric-label">After 3 Years</div>
                    <div class="metric-value">₹{val_3y:,.0f}</div>
                    <div style="font-size:0.9rem; font-weight:600; color:{profit_color_3}; margin-top:4px;">
                        {arrow_3} ₹{abs(profit_3y):,.0f} ({profit_3y/invest_amount*100:+.1f}% total)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""
                <div class="invest-card">
                    <div class="metric-label">After 5 Years</div>
                    <div class="metric-value">₹{val_5y:,.0f}</div>
                    <div style="font-size:0.9rem; font-weight:600; color:{profit_color_5}; margin-top:4px;">
                        {arrow_5} ₹{abs(profit_5y):,.0f} ({profit_5y/invest_amount*100:+.1f}% total)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Historical stats table with label ──
        st.markdown(
            '<p style="font-size:0.92rem; font-weight:600; color:#1e293b; '
            'margin-top:18px; margin-bottom:10px;">Historical Performance (5 Years)</p>',
            unsafe_allow_html=True,
        )
        hist_stats = pd.DataFrame(
            [
                {"Metric": "Starting Price (5y ago)",  "Value": f"₹{start_price:,.2f}"},
                {"Metric": "Current Price",             "Value": f"₹{end_price:,.2f}"},
                {"Metric": "Total Return (5y)",          "Value": f"{total_return_pct:+.2f}%"},
                {"Metric": "CAGR (Annualised)",          "Value": f"{cagr:+.2f}%/yr"},
                {"Metric": "Avg Daily Return",           "Value": f"{avg_daily_ret:+.4f}%"},
                {"Metric": "Best Year",                 "Value": f"{best_year:+.2f}%"},
                {"Metric": "Worst Year",                 "Value": f"{worst_year:+.2f}%"},
            ]
        )
        st.dataframe(hist_stats, use_container_width=True, hide_index=True)

        st.markdown("")
        st.markdown(
            """
            <div style="background:#fff7ed; border:1px solid #f97316; border-radius:10px;
                        padding:14px 18px; margin-top:8px;">
                <span style="font-size:0.88rem; color:#7c2d12; font-weight:500;">
                    ⚠️ <b>Disclaimer:</b> For educational purposes only — not financial advice.
                    Projections assume the historical CAGR continues, which is unlikely in reality.
                    Stock markets are volatile and past performance does not guarantee future results.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.info("Historical data not available for return calculation.")
