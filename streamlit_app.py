"""
Capstone: Real-Time Predictive Market Dashboard

Satisfies all four capstone requirements:
  1. Automated Data Ingestion  -> fetch_market_snapshot() + background scheduler
  2. In-Memory Analytical Queries -> DuckDB SQL over the ingested table
  3. Visual Uncertainty Forecasts -> historical trend + forecast + confidence band
  4. Live Public Deployment -> deploy this file via Streamlit Community Cloud

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import duckdb
import threading
import time
import os
from datetime import datetime, timezone
import plotly.graph_objects as go


# PAGE CONFIG + CUSTOM CSS  (styling requirement from Week 4)

st.set_page_config(page_title="Live Crypto Market Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    [data-testid="stMetricValue"] {
        font-size: 2.1rem;
        color: #1E3A8A;
        font-weight: 800;
    }
    .metric-card-container {
        border-radius: 8px;
        padding: 15px;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = "market_telemetry.csv"
API_URL = "https://api.coingecko.com/api/v3/coins/markets"
POLL_INTERVAL_SECONDS = 60
COINS_PER_PAGE = 10

PARAMS = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": COINS_PER_PAGE,
    "page": 1,
    "sparkline": "false",
    "price_change_percentage": "1h,24h,7d",
}


# REQUIREMENT 1 — AUTOMATED DATA INGESTION
# A background thread polls a live public API on a schedule and appends timestamped snapshots to a local CSV, independent of the UI thread.


def fetch_market_snapshot() -> pd.DataFrame:
    """Pulls one live snapshot of top-N coins by market cap from CoinGecko."""
    resp = requests.get(API_URL, params=PARAMS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    ingest_ts = datetime.now(timezone.utc).isoformat()
    rows = [{
        "utc_timestamp": ingest_ts,
        "name": coin.get("name"),
        "symbol": coin.get("symbol", "").upper(),
        "current_price": coin.get("current_price"),
        "market_cap": coin.get("market_cap"),
        "total_volume": coin.get("total_volume"),
        "price_change_pct_24h": coin.get("price_change_percentage_24h_in_currency"),
    } for coin in data]
    return pd.DataFrame(rows)


_csv_lock = threading.Lock()

def ingestion_worker():
    """Runs forever in a background thread, polling and appending to CSV."""
    while True:
        try:
            df = fetch_market_snapshot()
            with _csv_lock:
                write_header = not os.path.exists(DATA_FILE)
                df.to_csv(DATA_FILE, mode="a", header=write_header, index=False)
        except Exception as e:
            print(f"Ingestion error: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)


# Start the background ingestion thread exactly ONCE per running process, not once per browser session. Streamlit Cloud serves every visitor from
# the same process, so guarding with st.session_state (per-session) would let each new tab/visitor spawn its own thread, causing concurrent writes
# that corrupt the CSV. A module-level global persists across reruns and sessions within the same process, so it's the correct guard here.

if not globals().get("_ingestion_thread_started"):
    globals()["_ingestion_thread_started"] = True
    thread = threading.Thread(target=ingestion_worker, daemon=True)
    thread.start()
    # Seed the file immediately so the first page load has data to show
    if not os.path.exists(DATA_FILE):
        try:
            with _csv_lock:
                fetch_market_snapshot().to_csv(DATA_FILE, index=False)
        except Exception:
            pass


# REQUIREMENT 2 — IN-MEMORY ANALYTICAL QUERIES (DuckDB)

@st.cache_data(ttl=30)
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()

    try:
        con = duckdb.connect(database=":memory:")

        # Explicit column types instead of read_csv_auto's inference.
        # Root cause of the original BinderException: a stray leftover row (a duplicated CSV header written during the pre-lock race
        # condition) contains the literal text "total_volume" etc. in that position, which made read_csv_auto infer the whole column as
        # VARCHAR instead of numeric.
        con.execute(f"""
            CREATE OR REPLACE VIEW raw_market AS
            SELECT * FROM read_csv(
                '{DATA_FILE}',
                columns = {{
                    'utc_timestamp': 'VARCHAR',
                    'name': 'VARCHAR',
                    'symbol': 'VARCHAR',
                    'current_price': 'DOUBLE',
                    'market_cap': 'DOUBLE',
                    'total_volume': 'DOUBLE',
                    'price_change_pct_24h': 'DOUBLE'
                }},
                ignore_errors = true,
                header = true
            )
        """)

        # ignore_errors=true does NOT drop a malformed row outright — it
        # keeps the row and just sets any column that failed to cast to
        # NULL. Since utc_timestamp/name/symbol are text columns, a stray
        # duplicated-header row still passes through with valid-looking
        # text values (e.g. utc_timestamp = "utc_timestamp" literally),
        # only silently NULLing the numeric columns. That phantom row was
        # what broke MAX(utc_timestamp) filtering. This filter is the
        # actual fix: drop any row where a required numeric field is NULL,
        # which is exactly what a failed cast produces.
        con.execute("""
            CREATE OR REPLACE VIEW market AS
            SELECT * FROM raw_market
            WHERE current_price IS NOT NULL
              AND market_cap IS NOT NULL
              AND total_volume IS NOT NULL
              AND price_change_pct_24h IS NOT NULL
        """)

        row_count = con.execute("SELECT COUNT(*) FROM market").fetchone()[0]
        if row_count == 0:
            con.close()
            return pd.DataFrame()

        latest = con.execute("""
            SELECT * FROM market
            WHERE utc_timestamp = (SELECT MAX(utc_timestamp) FROM market)
            ORDER BY market_cap DESC
        """).df()

        history = con.execute("SELECT * FROM market ORDER BY utc_timestamp").df()

        summary = con.execute("""
            SELECT name, AVG(current_price) AS avg_price, AVG(total_volume) AS avg_volume
            FROM market GROUP BY name ORDER BY avg_volume DESC
        """).df()

        con.close()
        return {"latest": latest, "history": history, "summary": summary}

    except duckdb.Error:
        # Any other malformed-file issue: reset and let ingestion rebuild it.
        try:
            os.remove(DATA_FILE)
        except OSError:
            pass
        return pd.DataFrame()


data = load_data()

st.title("Live Crypto Market Dashboard")
st.caption("Live crypto market data, ingested continuously and queried in-memory with DuckDB.")

if not data or data["history"].empty:
    st.warning("Waiting on the first data snapshot — refresh in about a minute.")
    st.stop()

history = data["history"].copy()
history["utc_timestamp"] = pd.to_datetime(history["utc_timestamp"])
latest = data["latest"]


# KPI ROW (styled metric cards)

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown('<div class="metric-card-container">', unsafe_allow_html=True)
    st.metric("Coins Tracked", latest["name"].nunique())
    st.markdown('</div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="metric-card-container">', unsafe_allow_html=True)
    st.metric("Total Market Cap (top coins)", f"${latest['market_cap'].sum():,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="metric-card-container">', unsafe_allow_html=True)
    st.metric("Snapshots Collected", history["utc_timestamp"].nunique())
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# MARKET OVERVIEW VISUALIZATIONS
# These render from the latest snapshot alone, so they show useful
# content immediately — even before enough history exists to forecast.

st.subheader("Market Overview")

v1, v2 = st.columns(2)

with v1:
    bar_fig = go.Figure(go.Bar(
        x=latest.sort_values("market_cap", ascending=True)["symbol"],
        y=latest.sort_values("market_cap", ascending=True)["market_cap"],
        marker_color="#1D4ED8",
        orientation="v",
    ))
    bar_fig.update_layout(
        title="Market Cap by Coin",
        xaxis_title="Coin", yaxis_title="Market Cap (USD)",
        template="plotly_white",
    )
    st.plotly_chart(bar_fig, width="stretch")

with v2:
    change_sorted = latest.sort_values("price_change_pct_24h")
    colors = ["#DC2626" if v < 0 else "#059669" for v in change_sorted["price_change_pct_24h"]]
    change_fig = go.Figure(go.Bar(
        x=change_sorted["price_change_pct_24h"], y=change_sorted["symbol"],
        orientation="h", marker_color=colors,
    ))
    change_fig.update_layout(
        title="24h % Change by Coin",
        xaxis_title="% Change", template="plotly_white",
    )
    change_fig.add_vline(x=0, line_color="black", line_width=1)
    st.plotly_chart(change_fig, width="stretch")

st.divider()


# Historical trend + point forecast + shaded confidence interval, using a simple linear trend extrapolation as the forecasting model.


st.subheader("Price Forecast with Confidence Interval")

coin_choice = st.selectbox("Select a coin to forecast", sorted(history["name"].unique()))
sigma_multiplier = st.slider("Confidence width (standard deviations)", 1.0, 3.0, 1.96, step=0.1)

coin_hist = history[history["name"] == coin_choice].sort_values("utc_timestamp")

if len(coin_hist) < 3:
    st.info("Not enough snapshots yet for this coin to forecast — check back after a few more polling cycles.")
else:
    # Simple linear regression on time index as the forecasting model
    x = np.arange(len(coin_hist))
    y = coin_hist["current_price"].values
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)

    residuals = y - trend(x)
    resid_std = residuals.std() if len(residuals) > 1 else 0

    forecast_steps = 10
    future_x = np.arange(len(coin_hist), len(coin_hist) + forecast_steps)
    forecast_mean = trend(future_x)
    # Uncertainty grows the further out we project
    growing_std = resid_std * np.sqrt(1 + np.arange(1, forecast_steps + 1) / 5)
    upper_bound = forecast_mean + sigma_multiplier * growing_std
    lower_bound = forecast_mean - sigma_multiplier * growing_std

    last_ts = coin_hist["utc_timestamp"].iloc[-1]
    future_ts = pd.date_range(start=last_ts, periods=forecast_steps + 1, freq=f"{POLL_INTERVAL_SECONDS}s")[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coin_hist["utc_timestamp"], y=coin_hist["current_price"],
        name="Historical Price", line=dict(color="#1E293B"),
    ))
    fig.add_trace(go.Scatter(
        x=future_ts, y=forecast_mean,
        name="Forecast", line=dict(color="#1D4ED8", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=list(future_ts) + list(future_ts[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill="toself", fillcolor="rgba(29, 78, 216, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", name="Confidence Interval",
    ))
    fig.update_layout(
        title=f"{coin_choice} — Price Trend & Forecast",
        xaxis_title="Time", yaxis_title="Price (USD)",
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Forecasting model: linear trend extrapolation over ingested snapshots. "
        "Confidence band widens with forecast horizon, reflecting growing uncertainty."
    )

st.divider()


# SUPPORTING VIEW — latest snapshot table (from the DuckDB query)

st.subheader("Latest Market Snapshot")
st.dataframe(
    latest[["name", "symbol", "current_price", "market_cap", "total_volume", "price_change_pct_24h"]],
    width="stretch",
)

st.caption(
    "Data source: CoinGecko public API, polled automatically every "
    f"{POLL_INTERVAL_SECONDS} seconds by a background thread. "
    "Queried in-memory via DuckDB directly against the ingested CSV."
)
