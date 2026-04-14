"""
Gold Price Predictor — Indian Market (MCX)
Run with: streamlit run app.py

Dependencies:
    pip install streamlit yfinance scikit-learn tensorflow pandas numpy matplotlib
    pip install smtplib  # stdlib, no install needed
    pip install twilio   # for SMS alerts (optional)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import warnings
import resend
import time
import os
from datetime import datetime
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

load_dotenv()
resend.api_key = os.getenv("RESEND_API_KEY")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Predictor — India",
    page_icon="🥇",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .main { background-color: #080808; color: #f0f0f0; }
    .block-container { padding-top: 2rem; }

    .metric-card {
        background: linear-gradient(135deg, #111 0%, #1a1a1a 100%);
        border: 1px solid #2a2a2a;
        border-top: 2px solid #FFD700;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        transition: border-color 0.3s;
    }
    .metric-card:hover { border-color: #FFD700; border-top-color: #FFD700; }
    .metric-label { font-size: 11px; color: #666; margin-bottom: 4px; letter-spacing: 1.5px; text-transform: uppercase; }
    .metric-value { font-size: 24px; font-weight: 800; font-family: 'Space Mono', monospace; }
    .up   { color: #00E676; }
    .down { color: #FF1744; }
    .gold { color: #FFD700; }
    .neutral { color: #aaa; }

    .alert-card {
        background: #0f0f0f;
        border: 1px solid #333;
        border-left: 3px solid #FFD700;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .alert-triggered {
        border-left-color: #00E676;
        background: #001a00;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FFD700, #e6a800);
        color: #080808;
        font-weight: 800;
        font-family: 'Syne', sans-serif;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        width: 100%;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #ffe033, #FFD700); }
    .refresh-badge {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px;
        color: #888;
        font-family: 'Space Mono', monospace;
    }
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00E676;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    .disclaimer {
        font-size: 11px;
        color: #444;
        text-align: center;
        margin-top: 1.5rem;
        letter-spacing: 0.5px;
    }
    .section-header {
        font-size: 13px;
        color: #FFD700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

TROY_OZ_TO_GRAMS = 31.1035
SEQ_LEN          = 60
FEATURES         = ['Open','High','Low','Close','Volume','USDINR',
                    'MA_10','MA_20','MA_50','RSI','BB_upper','BB_lower',
                    'USDINR_MA5','Daily_Return','Volatility']


# ── Session State Init ────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "model_ready": False,
        "alerts": [],          # list of dicts: {id, type, threshold, channel, email, phone, triggered}
        "last_refresh": None,
        "live_price": None,
        "live_fx": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(period: str):
    gold = yf.download('GC=F',  period=period, interval='1d', progress=False)
    fx   = yf.download('INR=X', period=period, interval='1d', progress=False)

    # ── FIX: Flatten MultiIndex columns produced by newer yfinance versions ──
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.get_level_values(0)
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)

    gold.dropna(inplace=True)
    fx.dropna(inplace=True)

    fx_close = fx[['Close']].rename(columns={'Close': 'USDINR'})
    df = gold[['Open','High','Low','Close','Volume']].join(fx_close, how='inner')
    df.dropna(inplace=True)

    for col in ['Open','High','Low','Close']:
        df[col] = (df[col] * df['USDINR'] / TROY_OZ_TO_GRAMS) * 10

    # Technical features
    df['MA_10']        = df['Close'].rolling(10).mean()
    df['MA_20']        = df['Close'].rolling(20).mean()
    df['MA_50']        = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']          = 100 - (100 / (1 + gain / loss))

    std20 = df['Close'].rolling(20).std()
    df['BB_upper']     = df['MA_20'] + 2 * std20
    df['BB_lower']     = df['MA_20'] - 2 * std20
    df['USDINR_MA5']   = df['USDINR'].rolling(5).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility']   = df['Daily_Return'].rolling(10).std()

    df.dropna(inplace=True)
    return df


def fetch_live_price():
    """Fetch just the latest tick — lightweight, not cached."""
    try:
        gold = yf.download('GC=F',  period='5d', interval='1d', progress=False)
        fx   = yf.download('INR=X', period='5d', interval='1d', progress=False)
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        price_usd = float(gold['Close'].iloc[-1])
        fx_rate   = float(fx['Close'].iloc[-1])
        price_inr = (price_usd * fx_rate / TROY_OZ_TO_GRAMS) * 10
        return price_inr, fx_rate
    except Exception:
        return None, None


# ── Model Training ────────────────────────────────────────────────────────────
def build_sequences(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    close_idx = FEATURES.index('Close')

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i, close_idx])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], scaler, scaled, close_idx


def train_model(X_train, y_train):
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES))),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True, verbose=0)
    history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )
    return model, history


def inverse_close(scaler, scaled_vals, close_idx):
    dummy = np.zeros((len(scaled_vals), len(FEATURES)))
    dummy[:, close_idx] = scaled_vals
    return scaler.inverse_transform(dummy)[:, close_idx]


# ── Alert System ──────────────────────────────────────────────────────────────
def send_email_alert(to_email: str, subject: str, body: str):
    try:
        print("🚀 Sending email to:", to_email)   # 👈 ADD THIS

        response = resend.Emails.send({
            "from": "onboarding@resend.dev",
            "to": [to_email],
            "subject": subject,
            "html": f"<p>{body}</p>"
        })

        print("✅ Response:", response)  # 👈 ADD THIS
        return True, "Email sent ✓"

    except Exception as e:
        print("❌ ERROR:", e)  # 👈 ADD THIS
        return False, str(e)

def send_sms_alert(to_phone: str, body: str,
                   account_sid: str, auth_token: str, from_phone: str):
    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        client.messages.create(body=body, from_=from_phone, to=to_phone)
        return True, "SMS sent ✓"
    except ImportError:
        return False, "twilio not installed — run: pip install twilio"
    except Exception as e:
        return False, str(e)


def check_and_fire_alerts(current_price: float, predicted_price: float):
    """Check all alerts; fire those not yet triggered."""
    fired = []
    for alert in st.session_state.alerts:
        if alert.get("triggered") and alert.get("notification_sent"):
            continue
        thr  = alert["threshold"]
        atype = alert["type"]
        hit  = False
        direction = ""

        if atype == "Price Above"    and current_price  > thr:
            hit, direction = True, f"Current price ₹{current_price:,.0f} crossed ABOVE ₹{thr:,.0f}"
        elif atype == "Price Below"  and current_price  < thr:
            hit, direction = True, f"Current price ₹{current_price:,.0f} dropped BELOW ₹{thr:,.0f}"
        elif atype == "Predicted Above" and predicted_price > thr:
            hit, direction = True, f"Predicted price ₹{predicted_price:,.0f} is ABOVE ₹{thr:,.0f}"
        elif atype == "Predicted Below" and predicted_price < thr:
            hit, direction = True, f"Predicted price ₹{predicted_price:,.0f} is BELOW ₹{thr:,.0f}"

        if hit:
            alert["triggered"] = True
            alert["fired_at"]  = datetime.now().strftime("%H:%M:%S %d-%b-%Y")
            msg = f"{direction} at {alert['fired_at']}."
            fired.append({"alert": alert, "msg": msg})

    return fired


# ── Dark Chart Helper ─────────────────────────────────────────────────────────
def dark_fig(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#080808')
    ax.set_facecolor('#0f0f0f')
    ax.tick_params(colors='#666', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1f1f1f')
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# UI Layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 🥇 Gold Price Predictor — Indian Market (MCX)")
st.markdown("Powered by LSTM · Prices in **₹ per 10 grams** · Data: Yahoo Finance")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    history_period = st.selectbox("Training history", ["2y","3y","5y","10y"], index=2)
    show_indicators = st.multiselect(
        "Chart overlays",
        ["MA 10", "MA 20", "MA 50", "Bollinger Bands"],
        default=["MA 20", "MA 50"]
    )
    run_btn = st.button("🔄 Fetch & Train Model")

    st.markdown("---")
    # ── Live Refresh Settings ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">⚡ Live Price Refresh</p>', unsafe_allow_html=True)
    auto_refresh = st.toggle("Enable auto-refresh", value=False)
    refresh_interval = st.selectbox(
        "Refresh every",
        [1, 2, 5, 10, 15, 30],
        index=2,
        format_func=lambda x: f"{x} minute{'s' if x > 1 else ''}"
    ) if auto_refresh else 5
    if st.button("🔃 Refresh Price Now"):
        with st.spinner("Fetching latest price..."):
            price, fx = fetch_live_price()
            if price:
                st.session_state.live_price = price
                st.session_state.live_fx    = fx
                st.session_state.last_refresh = datetime.now()
                st.success(f"₹{price:,.0f} / 10g  |  USD/INR: {fx:.2f}")
            else:
                st.error("Failed to fetch live price.")

    st.markdown("---")
    st.markdown("**About**")
    st.caption("Uses GC=F (gold futures) converted to INR via live USD/INR rate. "
               "Model: 2-layer LSTM with 60-day lookback.")
    st.markdown('<p class="disclaimer">⚠️ Not financial advice. Educational use only.</p>',
                unsafe_allow_html=True)


# ── Auto-refresh logic ────────────────────────────────────────────────────────
if auto_refresh and st.session_state.get("model_ready"):
    last = st.session_state.get("last_refresh")
    should_refresh = (
        last is None or
        (datetime.now() - last).total_seconds() >= refresh_interval * 60
    )
    if should_refresh:
        price, fx = fetch_live_price()
        if price:
            st.session_state.live_price    = price
            st.session_state.live_fx       = fx
            st.session_state.last_refresh  = datetime.now()
        # Schedule next rerun
        time.sleep(0.1)
        st.rerun()


# ── Main Content ──────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Fetching live gold & USD/INR data..."):
        df = load_data(history_period)
        load_data.clear()   # bust cache so next fetch is fresh
        df = load_data(history_period)

    with st.spinner("Training LSTM model… this may take 1–2 minutes"):
        X_train, X_test, y_train, y_test, scaler, scaled, close_idx = build_sequences(df)
        model, history_obj = train_model(X_train, y_train)

    # Initial live price fetch
    live_price, live_fx = fetch_live_price()
    st.session_state.live_price   = live_price or df['Close'].iloc[-1]
    st.session_state.live_fx      = live_fx    or df['USDINR'].iloc[-1]
    st.session_state.last_refresh = datetime.now()

    st.session_state.model_ready  = True
    st.session_state.df           = df
    st.session_state.model        = model
    st.session_state.history_obj  = history_obj
    st.session_state.scaler       = scaler
    st.session_state.scaled       = scaled
    st.session_state.close_idx    = close_idx
    st.session_state.X_test       = X_test
    st.session_state.y_test       = y_test
    st.session_state.split        = len(X_train)

if not st.session_state.model_ready:
    st.info("👈 Select a history period and click **Fetch & Train Model** to get started.")
    st.stop()

# ── Use cached state ──────────────────────────────────────────────────────────
df         = st.session_state.df
model      = st.session_state.model
history_obj= st.session_state.history_obj
scaler     = st.session_state.scaler
scaled     = st.session_state.scaled
close_idx  = st.session_state.close_idx
X_test     = st.session_state.X_test
y_test     = st.session_state.y_test

# ── Predictions ───────────────────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred  = inverse_close(scaler, y_pred_scaled, close_idx)
y_true  = inverse_close(scaler, y_test, close_idx)

last_seq    = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURES))
next_scaled = model.predict(last_seq, verbose=0).flatten()
next_price  = inverse_close(scaler, next_scaled, close_idx)[0]

# Prefer live price if available
last_price = st.session_state.live_price or df['Close'].iloc[-1]
last_fx    = st.session_state.live_fx    or df['USDINR'].iloc[-1]
change     = next_price - last_price
change_pct = (change / last_price) * 100

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ── Check alerts ──────────────────────────────────────────────────────────────
fired_now = check_and_fire_alerts(last_price, next_price)

# ── Live refresh status bar ───────────────────────────────────────────────────
refresh_col, _ = st.columns([3, 1])
with refresh_col:
    last_r = st.session_state.last_refresh
    if last_r:
        if auto_refresh:
            next_r_sec = max(0, refresh_interval * 60 - (datetime.now() - last_r).total_seconds())
            st.markdown(
                f'<span class="live-dot"></span>'
                f'<span class="refresh-badge">LIVE · Updated {last_r.strftime("%H:%M:%S")} · '
                f'Next refresh in {int(next_r_sec//60):02d}:{int(next_r_sec%60):02d}</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="refresh-badge">Last updated {last_r.strftime("%H:%M:%S %d-%b-%Y")}</span>',
                unsafe_allow_html=True
            )

st.markdown("<br>", unsafe_allow_html=True)

# ── Metric Cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Live Price</div>
      <div class="metric-value gold">₹{last_price:,.0f}</div>
      <div class="metric-label">per 10g</div>
    </div>""", unsafe_allow_html=True)

with col2:
    color = "up" if change >= 0 else "down"
    arrow = "▲" if change >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Predicted Next Day</div>
      <div class="metric-value {color}">₹{next_price:,.0f}</div>
      <div class="metric-label">{arrow} ₹{abs(change):,.0f} ({change_pct:+.2f}%)</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">USD / INR</div>
      <div class="metric-value gold">₹{last_fx:.2f}</div>
      <div class="metric-label">live rate</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">MAE</div>
      <div class="metric-value gold">₹{mae:,.0f}</div>
      <div class="metric-label">MAPE: {mape:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">RMSE</div>
      <div class="metric-value gold">₹{rmse:,.0f}</div>
      <div class="metric-label">per 10g error</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Actual vs Predicted",
    "📉 Training Loss",
    "🕯️ Price History",
    "🔔 Alerts"
])

with tab1:
    test_dates = df.index[SEQ_LEN + st.session_state.split:]
    fig, ax = dark_fig((13, 4))
    ax.plot(test_dates, y_true, color='#FFD700', lw=1.5, label='Actual (MCX INR)')
    ax.plot(test_dates, y_pred, color='#FF6B35', lw=1.5, ls='--', label='Predicted')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title('Gold Price — Actual vs Predicted (Test Set)', color='#FFD700', fontsize=13)
    ax.set_ylabel('₹ per 10g', color='#666', fontsize=11)
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=10)
    ax.grid(alpha=0.06, color='white')
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, ax = dark_fig((13, 4))
    ax.plot(history_obj.history['loss'],     color='#FFD700', label='Train Loss')
    ax.plot(history_obj.history['val_loss'], color='#FF6B35', label='Val Loss')
    ax.set_title('Training & Validation Loss', color='#FFD700', fontsize=13)
    ax.set_xlabel('Epoch', color='#666')
    ax.set_ylabel('MSE Loss', color='#666')
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=10)
    ax.grid(alpha=0.06, color='white')
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    fig, ax = dark_fig((13, 4))
    ax.plot(df.index, df['Close'], color='#FFD700', lw=1.2, label='Close (₹/10g)')

    if "MA 10" in show_indicators:
        ax.plot(df.index, df['MA_10'], color='#4FC3F7', lw=1, ls='--', alpha=0.8, label='MA 10')
    if "MA 20" in show_indicators:
        ax.plot(df.index, df['MA_20'], color='#81C784', lw=1, ls='--', alpha=0.8, label='MA 20')
    if "MA 50" in show_indicators:
        ax.plot(df.index, df['MA_50'], color='#FF8A65', lw=1, ls='--', alpha=0.8, label='MA 50')
    if "Bollinger Bands" in show_indicators:
        ax.fill_between(df.index, df['BB_lower'], df['BB_upper'],
                        alpha=0.06, color='#FFD700', label='Bollinger Band')

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title(f'Gold Price History — {history_period}', color='#FFD700', fontsize=13)
    ax.set_ylabel('₹ per 10g', color='#666', fontsize=11)
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=10, ncol=3)
    ax.grid(alpha=0.06, color='white')
    plt.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Alerts
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔔 Price Alerts")
    st.caption("Get notified via Email or SMS when gold price crosses your threshold.")

    # Show any alerts fired this cycle
    if fired_now:
        for f in fired_now:
            st.success(f"🔔 Alert triggered: {f['msg']}")

    # ── Add New Alert ─────────────────────────────────────────────────────────
    with st.expander("➕ Add New Alert", expanded=len(st.session_state.alerts) == 0):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            alert_type = st.selectbox(
                "Condition",
                ["Price Above", "Price Below", "Predicted Above", "Predicted Below"],
                key="new_alert_type"
            )
        with ac2:
            alert_threshold = st.number_input(
                "Threshold (₹ per 10g)",
                min_value=10000,
                max_value=500000,
                value=int(last_price * 1.02),
                step=100,
                key="new_alert_threshold"
            )
        with ac3:
            alert_channel = st.selectbox(
                "Notify via",
                ["Email", "SMS (Twilio)", "Both"],
                key="new_alert_channel"
            )

        # Email config
        if alert_channel in ["Email", "Both"]:
            st.markdown("**📧 Email Settings**")
            alert_email = st.text_input("Recipient email", key="alert_email")

        else:
            alert_email = smtp_host = smtp_user = smtp_pass = ""
            smtp_port = 465

        # SMS config
        if alert_channel in ["SMS (Twilio)", "Both"]:
            st.markdown("**📱 Twilio SMS Settings**")
            sc1, sc2 = st.columns(2)
            with sc1:
                twilio_sid    = st.text_input("Account SID", key="twilio_sid")
                twilio_token  = st.text_input("Auth Token", type="password", key="twilio_token")
            with sc2:
                twilio_from   = st.text_input("From phone (+1234567890)", key="twilio_from")
                alert_phone   = st.text_input("To phone (+919876543210)", key="alert_phone")
        else:
            twilio_sid = twilio_token = twilio_from = alert_phone = ""

        if st.button("Add Alert"):
            new_alert = {
                "id":          len(st.session_state.alerts),
                "type":        alert_type,
                "threshold":   alert_threshold,
                "channel":     alert_channel,
                "email":       alert_email,
                "phone":       alert_phone,
                "twilio_sid":  twilio_sid,
                "twilio_token":twilio_token,
                "twilio_from": twilio_from,
                "triggered":   False,
                "fired_at":    None,
                "notification_sent": False,
            }
            st.session_state.alerts.append(new_alert)
            st.success(f"✅ Alert added: {alert_type} ₹{alert_threshold:,.0f} via {alert_channel}")
            st.rerun()

    # ── Active Alerts List ────────────────────────────────────────────────────
    if st.session_state.alerts:
        st.markdown("#### Active Alerts")
        for i, alert in enumerate(st.session_state.alerts):
            card_class = "alert-card alert-triggered" if alert.get("triggered") else "alert-card"
            status_icon = "✅" if alert.get("triggered") else "⏳"
            fired_info  = f" · Fired at {alert['fired_at']}" if alert.get("fired_at") else ""
            st.markdown(f"""
            <div class="{card_class}">
              <strong>{status_icon} {alert['type']} ₹{alert['threshold']:,.0f}</strong>
              &nbsp;·&nbsp; via {alert['channel']}
              {fired_info}
            </div>""", unsafe_allow_html=True)

            col_test, col_del, _ = st.columns([1, 1, 5])
            with col_test:
                if st.button("Test 🧪", key=f"test_{i}"):
                    st.write("Button clicked") 
                    msg = (f"TEST ALERT: {alert['type']} ₹{alert['threshold']:,.0f}\n"
                           f"Current: ₹{last_price:,.0f} | Predicted: ₹{next_price:,.0f}")
                    if alert["channel"] in ["Email", "Both"] and alert.get("email"):
                        ok, info = send_email_alert(
                            alert["email"],
                            "🥇 Gold Alert Test",
                            msg
                        )
                        st.success(f"Email sent: {info}")
                    if alert["channel"] in ["SMS (Twilio)", "Both"] and alert.get("phone"):
                        ok, info = send_sms_alert(
                            alert["phone"], msg,
                            alert["twilio_sid"], alert["twilio_token"], alert["twilio_from"]
                        )
                        st.info(f"SMS: {info}")

            with col_del:
                if st.button("Remove 🗑", key=f"del_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()
    else:
        st.info("No alerts configured yet. Add one above.")

    # ── Auto-fire on each render ───────────────────────────────────────────────
    for alert in st.session_state.alerts:
        if alert.get("triggered") and not alert.get("notification_sent"):
            msg = (f"Gold Alert: {alert['type']} ₹{alert['threshold']:,.0f} triggered! "
                   f"Current: ₹{last_price:,.0f} | Predicted: ₹{next_price:,.0f} "
                   f"at {alert.get('fired_at', '')}")
            if alert["channel"] in ["Email", "Both"] and alert.get("email"):
                ok, info = send_email_alert(
                    alert["email"],
                    "🥇 Gold Price Alert Triggered",
                    msg
                )
                if ok:
                    alert["notification_sent"] = True

            if alert["channel"] in ["SMS (Twilio)", "Both"] and alert.get("phone"):
                ok, info = send_sms_alert(
                    alert["phone"], msg,
                    alert["twilio_sid"], alert["twilio_token"], alert["twilio_from"]
                )
                if ok:
                    alert["notification_sent"] = True


# ── Auto-rerun for live refresh ───────────────────────────────────────────────
if auto_refresh and st.session_state.model_ready:
    last = st.session_state.get("last_refresh")
    if last:
        elapsed = (datetime.now() - last).total_seconds()
        wait    = max(0, refresh_interval * 60 - elapsed)
        # Use st.empty as a countdown placeholder
        countdown_ph = st.empty()
        countdown_ph.markdown(
            f'<p class="disclaimer">Auto-refresh in {int(wait//60):02d}:{int(wait%60):02d}</p>',
            unsafe_allow_html=True
        )

st.markdown(
    '<p class="disclaimer">⚠️ This tool is for educational purposes only. '
    'Predictions are not financial advice. Gold prices are influenced by many macro factors '
    'not captured in this model.</p>',
    unsafe_allow_html=True
)