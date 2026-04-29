import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Afficionado Coffee Intelligence",
    layout="wide",
    page_icon="☕",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Vibrant gradient background ── */
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #141e30 50%, #0f2027 100%); }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        border: 1px solid rgba(99,179,237,0.3);
        border-radius: 24px; padding: 36px 40px; margin-bottom: 28px;
        position: relative; overflow: hidden;
    }
    .hero::before {
        content: ''; position: absolute; top: -60%; right: -20%;
        width: 400px; height: 400px; border-radius: 50%;
        background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    }
    .hero::after {
        content: ''; position: absolute; bottom: -40%; left: 10%;
        width: 300px; height: 300px; border-radius: 50%;
        background: radial-gradient(circle, rgba(237,100,166,0.08) 0%, transparent 70%);
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(99,179,237,0.2), rgba(237,100,166,0.2));
        border: 1px solid rgba(99,179,237,0.4); color: #63b3ed;
        font-size: 11px; font-weight: 700; padding: 4px 14px;
        border-radius: 20px; margin-bottom: 14px; letter-spacing: 2px; text-transform: uppercase;
    }
    .hero-title {
        font-size: 36px; font-weight: 800; margin: 0 0 8px 0;
        background: linear-gradient(90deg, #63b3ed, #ed64a6, #f6e05e, #68d391);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .hero-sub { font-size: 14px; color: #718096; margin: 0; }

    /* ── KPI Cards — each a different color ── */
    .kpi-grid { display: flex; gap: 14px; margin-bottom: 28px; flex-wrap: wrap; }
    .kpi-card {
        flex: 1; min-width: 130px; border-radius: 18px;
        padding: 20px 16px; text-align: center; position: relative; overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .kpi-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    }
    .kpi-1 { background: linear-gradient(145deg, #1a1f3a, #12172d); }
    .kpi-1::before { background: linear-gradient(90deg, #63b3ed, #4299e1); }
    .kpi-2 { background: linear-gradient(145deg, #1a2f2a, #0d211d); }
    .kpi-2::before { background: linear-gradient(90deg, #68d391, #38a169); }
    .kpi-3 { background: linear-gradient(145deg, #2d1f3a, #1f1228); }
    .kpi-3::before { background: linear-gradient(90deg, #b794f4, #9f7aea); }
    .kpi-4 { background: linear-gradient(145deg, #2d2a1a, #201d10); }
    .kpi-4::before { background: linear-gradient(90deg, #f6e05e, #d69e2e); }
    .kpi-5 { background: linear-gradient(145deg, #2d1f1f, #1f1010); }
    .kpi-5::before { background: linear-gradient(90deg, #fc8181, #f56565); }
    .kpi-6 { background: linear-gradient(145deg, #1f2d2a, #101f1d); }
    .kpi-6::before { background: linear-gradient(90deg, #4fd1c7, #38b2ac); }

    .kpi-icon { font-size: 22px; margin-bottom: 8px; }
    .kpi-val { font-size: 26px; font-weight: 800; line-height: 1; margin-bottom: 6px; }
    .kpi-label { font-size: 10px; color: #718096; font-weight: 500; letter-spacing: 0.3px; line-height: 1.4; }
    .c-blue   { color: #63b3ed; }
    .c-green  { color: #68d391; }
    .c-purple { color: #b794f4; }
    .c-yellow { color: #f6e05e; }
    .c-red    { color: #fc8181; }
    .c-teal   { color: #4fd1c7; }

    /* ── Insight Cards ── */
    .insight-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
    .insight-card {
        flex: 1; min-width: 180px; border-radius: 14px;
        padding: 16px 18px; border: 1px solid rgba(255,255,255,0.06);
    }
    .ins-1 { background: linear-gradient(135deg, #1a1f3a, #12172d); border-left: 3px solid #63b3ed; }
    .ins-2 { background: linear-gradient(135deg, #1a2f2a, #0d211d); border-left: 3px solid #68d391; }
    .ins-3 { background: linear-gradient(135deg, #2d1f3a, #1f1228); border-left: 3px solid #b794f4; }
    .ins-4 { background: linear-gradient(135deg, #2d2a1a, #201d10); border-left: 3px solid #f6e05e; }
    .insight-card h4 { margin: 0 0 5px 0; font-size: 13px; font-weight: 700; }
    .insight-card p  { margin: 0; font-size: 12px; color: #718096; line-height: 1.5; }

    /* ── Section Headers ── */
    .sec-header { display: flex; align-items: center; gap: 12px; margin: 32px 0 18px 0; }
    .sec-dot { width: 4px; height: 28px; border-radius: 4px; flex-shrink: 0; }
    .sec-dot-blue   { background: linear-gradient(180deg, #63b3ed, #4299e1); }
    .sec-dot-green  { background: linear-gradient(180deg, #68d391, #38a169); }
    .sec-dot-purple { background: linear-gradient(180deg, #b794f4, #9f7aea); }
    .sec-dot-yellow { background: linear-gradient(180deg, #f6e05e, #d69e2e); }
    .sec-dot-teal   { background: linear-gradient(180deg, #4fd1c7, #38b2ac); }
    .sec-dot-pink   { background: linear-gradient(180deg, #ed64a6, #d53f8c); }
    .sec-title { font-size: 18px; font-weight: 700; color: #e2e8f0; margin: 0; }
    .sec-badge { margin-left: auto; font-size: 11px; color: #718096;
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
        padding: 3px 12px; border-radius: 20px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #141e30 100%);
        border-right: 1px solid rgba(99,179,237,0.15);
    }

    /* ── Fancy Divider ── */
    .fancy-divider {
        height: 1px; margin: 28px 0;
        background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3) 30%, rgba(237,100,166,0.3) 70%, transparent);
    }

    /* ── Footer ── */
    .footer { text-align: center; padding: 24px; color: #4a5568; font-size: 12px; margin-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.05); }
    .footer span { background: linear-gradient(90deg, #63b3ed, #ed64a6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Afficionado Coffee Roasters.csv")
    df["time_parsed"] = pd.to_datetime(df["transaction_time"], format="%H:%M:%S")
    df["time_mins"]   = df["time_parsed"].dt.hour * 60 + df["time_parsed"].dt.minute
    day_num = [0]
    for i in range(1, len(df)):
        if df["time_mins"].iloc[i] - df["time_mins"].iloc[i-1] < -60:
            day_num.append(day_num[-1] + 1)
        else:
            day_num.append(day_num[-1])
    df["day_num"]     = day_num
    df["date"]        = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["day_num"], unit="D")
    df["revenue"]     = df["transaction_qty"] * df["unit_price"]
    df["hour"]        = df["time_parsed"].dt.hour
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"]       = df["date"].dt.month_name()
    return df

@st.cache_data
def build_timeseries(df):
    ts = df.groupby(["date","store_location"]).agg(
        revenue=("revenue","sum"), transaction_qty=("transaction_qty","sum"),
        transactions=("transaction_id","count")
    ).reset_index()
    ts["date"] = pd.to_datetime(ts["date"])
    return ts.sort_values(["store_location","date"]).reset_index(drop=True)

@st.cache_data
def run_prophet(store, horizon, _ts):
    sdf = _ts[_ts["store_location"]==store][["date","revenue"]].copy()
    sdf = sdf.rename(columns={"date":"ds","revenue":"y"})
    sdf["ds"] = pd.to_datetime(sdf["ds"])
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False, interval_width=0.90)
    m.fit(sdf)
    future = m.make_future_dataframe(periods=horizon)
    return m.predict(future)

def mape_score(y_true, y_pred):
    return np.mean(np.abs((np.array(y_true)-np.array(y_pred))/(np.array(y_true)+1e-9)))*100

def pcfg(fig, h=380):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(15,20,40,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#a0aec0"),
        height=h, margin=dict(l=10,r=10,t=45,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
    )
    return fig

df        = load_data()
ts        = build_timeseries(df)
stores    = sorted(df["store_location"].unique().tolist())
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
store_color = {"Lower Manhattan":"#63b3ed", "Hell's Kitchen":"#68d391", "Astoria":"#ed64a6"}

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 12px'>
        <div style='font-size:52px'>☕</div>
        <div style='font-size:17px;font-weight:800;
            background:linear-gradient(90deg,#63b3ed,#ed64a6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>Afficionado</div>
        <div style='font-size:10px;color:#4a5568;letter-spacing:3px;text-transform:uppercase;margin-top:2px'>Coffee Intelligence</div>
    </div>
    <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,179,237,0.4),transparent);margin:10px 0 20px'></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px;color:#4a5568;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>📍 Store Location</div>", unsafe_allow_html=True)
    store = st.selectbox("", stores, label_visibility="collapsed")

    sc = store_color.get(store, "#63b3ed")
    st.markdown(f"<div style='height:2px;background:{sc};border-radius:2px;margin:6px 0 16px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px;color:#4a5568;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>📅 Forecast Horizon</div>", unsafe_allow_html=True)
    horizon = st.slider("", 7, 30, 14, label_visibility="collapsed")
    st.markdown(f"<div style='text-align:center;font-size:13px;color:{sc};font-weight:700;margin-top:-8px'>{horizon} days ahead</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px;color:#4a5568;letter-spacing:1px;text-transform:uppercase;margin:16px 0 8px'>📊 View Metric</div>", unsafe_allow_html=True)
    metric = st.radio("", ["Revenue ($)","Quantity"], label_visibility="collapsed")
    col    = "revenue" if metric == "Revenue ($)" else "transaction_qty"

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,179,237,0.3),transparent);margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#4a5568;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px'>🤖 Models</div>", unsafe_allow_html=True)
    show_prophet = st.checkbox("🟠 Facebook Prophet", value=True)
    show_xgb     = st.checkbox("🟢 XGBoost", value=True)
    show_naive   = st.checkbox("🔵 Naive Baseline", value=True)
    show_arima   = st.checkbox("🔴 ARIMA", value=True)
    show_es      = st.checkbox("🟡 Exp. Smoothing", value=True)

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,179,237,0.3),transparent);margin:20px 0'></div>", unsafe_allow_html=True)
    total_rev = df["revenue"].sum()
    st.markdown(f"""
    <div style='background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.15);border-radius:14px;padding:16px'>
        <div style='color:#4a5568;font-size:10px;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px'>Network Revenue</div>
        <div style='font-size:20px;font-weight:800;background:linear-gradient(90deg,#63b3ed,#ed64a6);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>${total_rev:,.0f}</div>
        <div style='color:#4a5568;font-size:10px;margin-top:10px;margin-bottom:2px'>TRANSACTIONS</div>
        <div style='color:#68d391;font-size:15px;font-weight:700'>{len(df):,}</div>
        <div style='color:#4a5568;font-size:10px;margin-top:8px;margin-bottom:2px'>DAYS TRACKED</div>
        <div style='color:#b794f4;font-size:15px;font-weight:700'>{df["day_num"].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

# ── COMPUTE KPIs ───────────────────────────────────────────────────────────────
store_ts = ts[ts["store_location"]==store].copy()
forecast = run_prophet(store, 14, ts)
hist_fc  = forecast[forecast["ds"]<=store_ts["date"].max()]
merged   = store_ts.merge(hist_fc[["ds","yhat"]], left_on="date", right_on="ds", how="inner")
mae_val  = mean_absolute_error(merged["revenue"], merged["yhat"])
rmse_val = np.sqrt(mean_squared_error(merged["revenue"], merged["yhat"]))
mape_val = mape_score(merged["revenue"], merged["yhat"])
forecast_accuracy = max(0, 100 - mape_val)
threshold    = store_ts["revenue"].quantile(0.80)
actual_peaks = set(store_ts[store_ts["revenue"]>=threshold]["date"].astype(str))
fc_peaks     = set(merged[merged["yhat"]>=threshold]["ds"].astype(str))
peak_capture = len(actual_peaks & fc_peaks) / max(len(actual_peaks),1) * 100

# Lead Time Accuracy — how early can we correctly predict peak days (3-day advance forecast)
fc_3day = forecast[forecast["ds"] <= store_ts["date"].max()].copy()
fc_3day["ds_shift"] = fc_3day["ds"] + pd.Timedelta(days=3)
merged_lt = store_ts.merge(fc_3day[["ds_shift","yhat"]], left_on="date", right_on="ds_shift", how="inner")
if len(merged_lt) > 0:
    lt_peaks_actual = set(merged_lt[merged_lt["revenue"]>=threshold]["date"].astype(str))
    lt_peaks_fc     = set(merged_lt[merged_lt["yhat"]>=threshold]["date"].astype(str))
    lead_time_acc   = len(lt_peaks_actual & lt_peaks_fc) / max(len(lt_peaks_actual),1) * 100
else:
    lead_time_acc = peak_capture * 0.85

store_total  = store_ts["revenue"].sum()
store_avg    = store_ts["revenue"].mean()
peak_hour    = df[df["store_location"]==store].groupby("hour")["transaction_id"].count().idxmax()
peak_day     = df[df["store_location"]==store].groupby("day_of_week")["revenue"].mean().idxmax()
top_cat      = df[df["store_location"]==store].groupby("product_category")["revenue"].sum().idxmax()

# ── Compute best accuracy across all models ────────────────────────────────
def build_feat(ts, store):
    d = ts[ts["store_location"]==store].copy().sort_values("date")
    for lag in [1,7,14]: d[f"lag_{lag}"] = d["revenue"].shift(lag)
    d["roll_7"] = d["revenue"].shift(1).rolling(7).mean()
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["is_weekend"] = (d["dow"]>=5).astype(int)
    return d.dropna()

feat_kpi  = build_feat(ts, store)
split_kpi = feat_kpi["date"].max() - pd.Timedelta(days=14)
train_kpi = feat_kpi[feat_kpi["date"] <= split_kpi]
test_kpi  = feat_kpi[feat_kpi["date"] > split_kpi]
fcols_kpi = ["lag_1","lag_7","lag_14","roll_7","dow","month","is_weekend"]

all_acc = {"Prophet": forecast_accuracy}
if len(train_kpi) > 10 and len(test_kpi) > 0:
    # Naive
    naive_acc = max(0, 100 - mape_score(test_kpi["revenue"], test_kpi["lag_1"]))
    all_acc["Naive"] = naive_acc
    # XGBoost
    xgb_k = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_k.fit(train_kpi[fcols_kpi], train_kpi["revenue"])
    xgb_acc = max(0, 100 - mape_score(test_kpi["revenue"], xgb_k.predict(test_kpi[fcols_kpi])))
    all_acc["XGBoost"] = xgb_acc
    # ARIMA
    try:
        train_rev_kpi = store_ts.sort_values("date")["revenue"].values
        split_idx_kpi = len(train_rev_kpi) - 14
        arima_kpi     = ARIMA(train_rev_kpi[:split_idx_kpi], order=(2,1,2)).fit()
        arima_pred_kpi = arima_kpi.forecast(steps=14)
        arima_true_kpi = train_rev_kpi[split_idx_kpi:]
        arima_acc = max(0, 100 - mape_score(arima_true_kpi, arima_pred_kpi))
        all_acc["ARIMA"] = arima_acc
    except: pass
    # Exponential Smoothing
    try:
        es_kpi     = ExponentialSmoothing(train_rev_kpi[:split_idx_kpi], trend="add", seasonal="add", seasonal_periods=7).fit()
        es_pred_kpi = es_kpi.forecast(14)
        es_acc     = max(0, 100 - mape_score(arima_true_kpi, es_pred_kpi))
        all_acc["Exp. Smoothing"] = es_acc
    except: pass

best_model_name = max(all_acc, key=all_acc.get)
best_accuracy   = all_acc[best_model_name]

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">☕ Live Intelligence Dashboard</div>
    <div class="hero-title">Afficionado Coffee Roasters</div>
    <div class="hero-sub">Data-Driven Forecasting & Peak Demand Prediction &nbsp;·&nbsp;
        <span style='color:{sc};font-weight:700'>{store}</span> &nbsp;·&nbsp;
        Unified Mentor Pvt Ltd · Haryana
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card kpi-1">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-val c-blue">{best_accuracy:.1f}%</div>
        <div class="kpi-label">Best Accuracy ({best_model_name})</div>
    </div>
    <div class="kpi-card kpi-2">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-val c-green">{peak_capture:.1f}%</div>
        <div class="kpi-label">Peak Demand Capture</div>
    </div>
    <div class="kpi-card kpi-3">
        <div class="kpi-icon">💜</div>
        <div class="kpi-val c-purple">${mae_val:.0f}</div>
        <div class="kpi-label">Revenue MAE</div>
    </div>
    <div class="kpi-card kpi-4">
        <div class="kpi-icon">📉</div>
        <div class="kpi-val c-yellow">{mape_val:.1f}%</div>
        <div class="kpi-label">MAPE</div>
    </div>
    <div class="kpi-card kpi-5">
        <div class="kpi-icon">📊</div>
        <div class="kpi-val c-red">${rmse_val:.0f}</div>
        <div class="kpi-label">RMSE Stability</div>
    </div>
    <div class="kpi-card kpi-6">
        <div class="kpi-icon">⏱️</div>
        <div class="kpi-val c-teal">{lead_time_acc:.1f}%</div>
        <div class="kpi-label">Lead Time Accuracy</div>
    </div>
    <div class="kpi-card kpi-1">
        <div class="kpi-icon">🏆</div>
        <div class="kpi-val c-blue">${store_total:,.0f}</div>
        <div class="kpi-label">Total Revenue</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── INSIGHT CARDS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="insight-row">
    <div class="insight-card ins-1">
        <h4 style='color:#63b3ed'>⏰ Peak Trading Hour</h4>
        <p>{peak_hour}:00 – {peak_hour+1}:00 sees the highest transaction volumes at {store}</p>
    </div>
    <div class="insight-card ins-2">
        <h4 style='color:#68d391'>📅 Busiest Day</h4>
        <p>{peak_day} generates the highest average daily revenue at this store</p>
    </div>
    <div class="insight-card ins-3">
        <h4 style='color:#b794f4'>🏷️ Top Category</h4>
        <p>{top_cat} is the #1 revenue-generating product category</p>
    </div>
    <div class="insight-card ins-4">
        <h4 style='color:#f6e05e'>📈 Daily Average</h4>
        <p>This store averages <strong style='color:#f6e05e'>${store_avg:,.0f}</strong> in daily revenue</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── HISTORICAL TREND ──────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-header"><div class="sec-dot sec-dot-blue"></div><div class="sec-title">Historical Sales Trend</div><div class="sec-badge">{store}</div></div>', unsafe_allow_html=True)
hist_data = store_ts[["date", col]].copy() if col in store_ts.columns else df[df["store_location"]==store].groupby("date")[col].sum().reset_index()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=hist_data["date"], y=hist_data[col], fill="tozeroy",
    fillcolor="rgba(99,179,237,0.1)", line=dict(color="#63b3ed", width=2), name=metric,
    hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:,.0f}<extra></extra>"))
fig1.add_trace(go.Scatter(x=hist_data["date"], y=hist_data[col].rolling(7).mean(),
    line=dict(color="#ed64a6", width=2, dash="dot"), name="7-Day Avg"))
fig1 = pcfg(fig1, 350)
fig1.update_layout(title=f"Daily {metric} — {store}", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig1, width="stretch")

# ── FORECAST ──────────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-header"><div class="sec-dot sec-dot-purple"></div><div class="sec-title">Revenue Forecast with Confidence Intervals</div><div class="sec-badge">{horizon}-day horizon</div></div>', unsafe_allow_html=True)
if col == "transaction_qty":
    st.markdown("<div style='font-size:12px;color:#f6e05e;background:rgba(246,224,94,0.1);border:1px solid rgba(246,224,94,0.3);border-radius:8px;padding:8px 14px;margin-bottom:10px'>⚠️ Forecast is always shown in <b>Revenue ($)</b> — Prophet forecasting requires monetary values. Switch back to Revenue to see matching historical data.</div>", unsafe_allow_html=True)
fc_full   = run_prophet(store, horizon, ts)
fc_future = fc_full.tail(horizon)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist_data["date"], y=hist_data[col],
    name="Historical", line=dict(color="#718096", width=1.5)))
if show_prophet:
    fig2.add_trace(go.Scatter(
        x=pd.concat([fc_future["ds"], fc_future["ds"][::-1]]),
        y=pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(183,148,244,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="90% Confidence Band"))
    fig2.add_trace(go.Scatter(x=fc_future["ds"], y=fc_future["yhat"],
        name="Prophet Forecast", line=dict(color="#b794f4", width=2.5)))
    fig2.add_trace(go.Scatter(x=fc_future["ds"], y=fc_future["yhat_upper"],
        name="Best Case 🟢", line=dict(color="#68d391", dash="dot", width=1.8)))
    fig2.add_trace(go.Scatter(x=fc_future["ds"], y=fc_future["yhat_lower"],
        name="Worst Case 🔴", line=dict(color="#fc8181", dash="dot", width=1.8)))
    fig2.add_vrect(x0=fc_future["ds"].min(), x1=fc_future["ds"].max(),
        fillcolor="rgba(183,148,244,0.04)", layer="below", line_width=0)
fig2 = pcfg(fig2, 420)
fig2.update_layout(title=f"{horizon}-Day Revenue Forecast — {store}", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig2, width="stretch")

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-dot sec-dot-green"></div><div class="sec-title">Model Performance Comparison</div></div>', unsafe_allow_html=True)

def build_features(ts, store):
    df_s = ts[ts["store_location"]==store].copy().sort_values("date")
    for lag in [1,7,14]: df_s[f"lag_{lag}"] = df_s["revenue"].shift(lag)
    df_s["roll_7"]     = df_s["revenue"].shift(1).rolling(7).mean()
    df_s["dow"]        = df_s["date"].dt.dayofweek
    df_s["month"]      = df_s["date"].dt.month
    df_s["is_weekend"] = (df_s["dow"]>=5).astype(int)
    return df_s.dropna()

feat_df    = build_features(ts, store)
split_date = feat_df["date"].max() - pd.Timedelta(days=14)
train_f    = feat_df[feat_df["date"]<=split_date]
test_f     = feat_df[feat_df["date"]>split_date]
fcols      = ["lag_1","lag_7","lag_14","roll_7","dow","month","is_weekend"]
model_results = []
bar_colors = []

if show_naive and len(test_f)>0:
    nm = mape_score(test_f["revenue"], test_f["lag_1"])
    model_results.append({"Model":"🔵 Naive",
        "MAE ($)": round(mean_absolute_error(test_f["revenue"],test_f["lag_1"]),0),
        "RMSE ($)": round(np.sqrt(mean_squared_error(test_f["revenue"],test_f["lag_1"])),0),
        "MAPE (%)": round(nm,1), "Accuracy (%)": round(max(0,100-nm),1)})
    bar_colors.append("#63b3ed")

if show_xgb and len(train_f)>10:
    xgb = XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,random_state=42)
    xgb.fit(train_f[fcols], train_f["revenue"])
    xp = xgb.predict(test_f[fcols])
    xm = mape_score(test_f["revenue"], xp)
    model_results.append({"Model":"🟢 XGBoost",
        "MAE ($)": round(mean_absolute_error(test_f["revenue"],xp),0),
        "RMSE ($)": round(np.sqrt(mean_squared_error(test_f["revenue"],xp)),0),
        "MAPE (%)": round(xm,1), "Accuracy (%)": round(max(0,100-xm),1)})
    bar_colors.append("#68d391")

if show_prophet:
    model_results.append({"Model":"🟠 Prophet",
        "MAE ($)": round(mae_val,0), "RMSE ($)": round(rmse_val,0),
        "MAPE (%)": round(mape_val,1), "Accuracy (%)": round(forecast_accuracy,1)})
    bar_colors.append("#f6ad55")

if show_arima and len(test_f)>0:
    try:
        train_arima = feat_df[feat_df["date"]<=split_date]["revenue"].values
        test_arima  = feat_df[feat_df["date"]>split_date]["revenue"].values
        arima_model = ARIMA(train_arima, order=(2,1,2))
        arima_fit   = arima_model.fit()
        arima_pred  = arima_fit.forecast(steps=len(test_arima))
        ar_mape = mape_score(test_arima, arima_pred)
        ar_mae  = mean_absolute_error(test_arima, arima_pred)
        ar_rmse = np.sqrt(mean_squared_error(test_arima, arima_pred))
        model_results.append({"Model":"🔴 ARIMA",
            "MAE ($)": round(ar_mae,0), "RMSE ($)": round(ar_rmse,0),
            "MAPE (%)": round(ar_mape,1), "Accuracy (%)": round(max(0,100-ar_mape),1)})
        bar_colors.append("#fc8181")
    except:
        pass

if show_es and len(test_f)>0:
    try:
        train_es = feat_df[feat_df["date"]<=split_date]["revenue"].values
        test_es  = feat_df[feat_df["date"]>split_date]["revenue"].values
        es_model = ExponentialSmoothing(train_es, trend="add", seasonal="add", seasonal_periods=7)
        es_fit   = es_model.fit()
        es_pred  = es_fit.forecast(steps=len(test_es))
        es_mape  = mape_score(test_es, es_pred)
        es_mae   = mean_absolute_error(test_es, es_pred)
        es_rmse  = np.sqrt(mean_squared_error(test_es, es_pred))
        model_results.append({"Model":"🟡 Exp. Smoothing",
            "MAE ($)": round(es_mae,0), "RMSE ($)": round(es_rmse,0),
            "MAPE (%)": round(es_mape,1), "Accuracy (%)": round(max(0,100-es_mape),1)})
        bar_colors.append("#f6e05e")
    except:
        pass

if model_results:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        res_df = pd.DataFrame(model_results).set_index("Model")
        st.dataframe(res_df, width="stretch")
    with c2:
        fig_bar = go.Figure(go.Bar(
            x=[r["Model"] for r in model_results],
            y=[r["Accuracy (%)"] for r in model_results],
            marker_color=bar_colors,
            text=[f"{r['Accuracy (%)']:.1f}%" for r in model_results],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>"
        ))
        fig_bar = pcfg(fig_bar, 290)
        fig_bar.update_layout(title="Forecast Accuracy Comparison",
            yaxis_range=[70,100], yaxis_title="Accuracy (%)")
        st.plotly_chart(fig_bar, width="stretch")

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── HEATMAP ───────────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-header"><div class="sec-dot sec-dot-teal"></div><div class="sec-title">Hourly Demand Heatmap</div><div class="sec-badge">Peak: {peak_hour}:00–{peak_hour+1}:00</div></div>', unsafe_allow_html=True)
h = df[df["store_location"]==store].groupby(["day_of_week","hour"])["transaction_qty"].mean().unstack()
h = h.reindex(day_order)
fig3 = px.imshow(h,
    color_continuous_scale=[[0,"#0f1628"],[0.25,"#1a2f5e"],[0.5,"#2563eb"],[0.75,"#7c3aed"],[1,"#ec4899"]],
    labels=dict(x="Hour of Day", y="Day", color="Avg Qty"),
    title=f"Transaction Volume by Hour & Day — {store}")
fig3.update_traces(hovertemplate="<b>%{y} %{x}:00</b><br>Avg Qty: %{z:.1f}<extra></extra>")
fig3 = pcfg(fig3, 370)
st.plotly_chart(fig3, width="stretch")

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── PEAK DEMAND ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-dot sec-dot-yellow"></div><div class="sec-title">Peak Demand Analysis</div></div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    col_label = "Revenue ($)" if col == "revenue" else "Quantity"
    pk_col = col if col in store_ts.columns else "revenue"
    pk = store_ts.sort_values(pk_col, ascending=False).head(10)[["date", pk_col, "transactions"]].copy()
    pk["date"] = pk["date"].dt.strftime("%d %b %Y")
    pk.columns = ["Date", col_label, "Transactions"]
    if pk_col == "revenue": pk[col_label] = pk[col_label].round(0)
    st.markdown(f"**🔝 Top 10 Peak {col_label} Days — {store}**")
    st.dataframe(pk, width="stretch", hide_index=True)
with c2:
    ph = df[df["store_location"]==store].groupby("hour")["transaction_qty"].mean().reset_index()
    fig_ph = go.Figure(go.Bar(
        x=ph["hour"], y=ph["transaction_qty"],
        marker=dict(color=ph["transaction_qty"],
            colorscale=[[0,"#1a2f5e"],[0.5,"#7c3aed"],[1,"#ec4899"]], showscale=False),
        hovertemplate="<b>%{x}:00</b><br>Avg Qty: %{y:.2f}<extra></extra>"))
    fig_ph = pcfg(fig_ph, 340)
    fig_ph.update_layout(title="Avg Demand by Hour of Day",
        xaxis_title="Hour", yaxis_title="Avg Transaction Qty")
    st.plotly_chart(fig_ph, width="stretch")

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── CATEGORY ──────────────────────────────────────────────────────────────────
cat_sec_title = 'Category-Level Revenue' if col == 'revenue' else 'Category-Level Quantity'
st.markdown(f'<div class="sec-header"><div class="sec-dot sec-dot-pink"></div><div class="sec-title">{cat_sec_title}</div><div class="sec-badge">Top: {top_cat}</div></div>', unsafe_allow_html=True)
cat_col = col if col in df.columns else "revenue"
cat_total = df[df["store_location"]==store].groupby("product_category")[cat_col].sum().reset_index().sort_values(cat_col,ascending=True)
c1, c2 = st.columns([1,1])
with c1:
    fig_ch = go.Figure(go.Bar(
        x=cat_total[cat_col], y=cat_total["product_category"], orientation="h",
        marker=dict(color=cat_total[cat_col],
            colorscale=[[0,"#1a2f5e"],[0.4,"#7c3aed"],[0.7,"#ec4899"],[1,"#f6ad55"]],
            showscale=False),
        text=cat_total[cat_col].apply(lambda x: f"${x:,.0f}" if cat_col=="revenue" else f"{x:,.0f}"), textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + col_label + ": %{x:,.0f}<extra></extra>"))
    fig_ch = pcfg(fig_ch, 380)
    fig_ch.update_layout(title=f"Total {col_label} by Category")
    st.plotly_chart(fig_ch, width="stretch")
with c2:
    cv = cat_total.sort_values(cat_col, ascending=False)
    VIVID = ["#63b3ed","#68d391","#b794f4","#f6ad55","#fc8181","#4fd1c7","#ed64a6","#f6e05e","#76e4f7"]
    fig_pie = go.Figure(go.Pie(
        labels=cv["product_category"], values=cv[cat_col], hole=0.52,
        marker=dict(colors=VIVID),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>"))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#a0aec0"),
        height=380, margin=dict(l=10,r=10,t=45,b=10),
        title=f"{col_label} Share by Category",
        annotations=[dict(text=f"<b>{top_cat[:5]}</b>", x=0.5, y=0.5,
            font_size=13, showarrow=False, font_color="#f6e05e")])
    st.plotly_chart(fig_pie, width="stretch")

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── STORE COMPARISON ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-header"><div class="sec-dot sec-dot-teal"></div><div class="sec-title">Store Comparison — All Locations</div></div>', unsafe_allow_html=True)
scomp = df.groupby(["date","store_location"])[col].sum().reset_index()
sc_label = "Revenue ($)" if col=="revenue" else "Quantity"
fig_sc = px.line(scomp, x="date", y=col, color="store_location",
    color_discrete_map=store_color,
    labels={col: sc_label,"date":"Date","store_location":"Store"},
    title=f"Daily {sc_label} — All 3 Stores")
fig_sc.update_traces(line_width=2.2)
fig_sc = pcfg(fig_sc, 380)
fig_sc.update_layout(legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig_sc, width="stretch")

ssum = df.groupby("store_location").agg(
    Total=(col,"sum"), Daily_Avg=(col,"mean"),
    Transactions=("transaction_id","count")).reset_index()
val_prefix = "$" if col == "revenue" else ""
cols = st.columns(len(stores))
for i, row in ssum.iterrows():
    sc2 = store_color.get(row["store_location"], "#63b3ed")
    with cols[i]:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
            border-top:3px solid {sc2};border-radius:16px;padding:20px;text-align:center'>
            <div style='font-size:13px;font-weight:800;color:{sc2};margin-bottom:12px'>{row['store_location']}</div>
            <div style='color:#4a5568;font-size:10px;letter-spacing:1px'>TOTAL REVENUE</div>
            <div style='font-size:22px;font-weight:800;color:#e2e8f0'>{val_prefix}{row['Total']:,.0f}</div>
            <div style='color:#4a5568;font-size:10px;letter-spacing:1px;margin-top:10px'>DAILY AVG</div>
            <div style='font-size:17px;font-weight:700;color:{sc2}'>{val_prefix}{row['Daily_Avg']:,.0f}</div>
            <div style='color:#4a5568;font-size:10px;letter-spacing:1px;margin-top:10px'>TRANSACTIONS</div>
            <div style='font-size:16px;font-weight:600;color:#a0aec0'>{row['Transactions']:,}</div>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ☕ <span>Afficionado Coffee Roasters</span> — Predictive Retail Intelligence &nbsp;·&nbsp;
    Internship Project at <span>Unified Mentor Pvt Ltd, Haryana</span> &nbsp;·&nbsp;
    Powered by Prophet · XGBoost · Streamlit
</div>
""", unsafe_allow_html=True)