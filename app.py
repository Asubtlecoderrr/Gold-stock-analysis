"""
Gold Price Predictor — Indian Market (MCX)
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Predictor — India",
    page_icon="🥇",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d0d0d; color: #f0f0f0; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; }
    .up   { color: #4CAF50; }
    .down { color: #F44336; }
    .gold { color: #FFD700; }
    .stButton > button {
        background: #FFD700;
        color: #0d0d0d;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton > button:hover { background: #e6c200; }
    .disclaimer {
        font-size: 11px;
        color: #555;
        text-align: center;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

TROY_OZ_TO_GRAMS = 31.1035
SEQ_LEN          = 60
FEATURES         = ['Open','High','Low','Close','Volume','USDINR',
                    'MA_10','MA_20','MA_50','RSI','BB_upper','BB_lower',
                    'USDINR_MA5','Daily_Return','Volatility']


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(period: str):
    gold = yf.download('GC=F',  period=period, interval='1d', progress=False)
    fx   = yf.download('INR=X', period=period, interval='1d', progress=False)
    gold.dropna(inplace=True)
    fx.dropna(inplace=True)

    fx_close = fx[['Close']].rename(columns={'Close': 'USDINR'})
    df = gold[['Open','High','Low','Close','Volume']].join(fx_close, how='inner')
    df.dropna(inplace=True)

    for col in ['Open','High','Low','Close']:
        df[col] = (df[col] * df['USDINR'] / TROY_OZ_TO_GRAMS) * 10

    # Features
    df['MA_10']       = df['Close'].rolling(10).mean()
    df['MA_20']       = df['Close'].rolling(20).mean()
    df['MA_50']       = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    std20 = df['Close'].rolling(20).std()
    df['BB_upper']    = df['MA_20'] + 2 * std20
    df['BB_lower']    = df['MA_20'] - 2 * std20
    df['USDINR_MA5']  = df['USDINR'].rolling(5).mean()
    df['Daily_Return']= df['Close'].pct_change()
    df['Volatility']  = df['Daily_Return'].rolling(10).std()

    df.dropna(inplace=True)
    return df


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
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
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


# ── Dark Chart Helper ─────────────────────────────────────────────────────────
def dark_fig(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0d0d0d')
    ax.set_facecolor('#111')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_color('#2a2a2a')
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
    run_btn = st.button("🔄 Fetch & Predict")
    st.markdown("---")
    st.markdown("**About**")
    st.caption("Uses GC=F (gold futures) converted to INR via live USD/INR rate. "
               "Model: 2-layer LSTM with 60-day lookback.")
    st.markdown('<p class="disclaimer">⚠️ Not financial advice. For educational purposes only.</p>',
                unsafe_allow_html=True)


# ── Main Content ──────────────────────────────────────────────────────────────
if run_btn or "model_ready" not in st.session_state:

    with st.spinner("Fetching live gold & USD/INR data..."):
        df = load_data(history_period)

    with st.spinner("Training LSTM model... (may take 1-2 min)"):
        X_train, X_test, y_train, y_test, scaler, scaled, close_idx = build_sequences(df)
        model, history = train_model(X_train, y_train)

    # Store in session
    st.session_state.model_ready = True
    st.session_state.df          = df
    st.session_state.model       = model
    st.session_state.history     = history
    st.session_state.scaler      = scaler
    st.session_state.scaled      = scaled
    st.session_state.close_idx   = close_idx
    st.session_state.X_test      = X_test
    st.session_state.y_test      = y_test
    st.session_state.split       = int(len(X_train) + len(X_test)) - len(X_test)

# Use cached state
df        = st.session_state.df
model     = st.session_state.model
history   = st.session_state.history
scaler    = st.session_state.scaler
scaled    = st.session_state.scaled
close_idx = st.session_state.close_idx
X_test    = st.session_state.X_test
y_test    = st.session_state.y_test

# ── Predictions ───────────────────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred  = inverse_close(scaler, y_pred_scaled, close_idx)
y_true  = inverse_close(scaler, y_test,        close_idx)

last_seq    = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURES))
next_scaled = model.predict(last_seq, verbose=0).flatten()
next_price  = inverse_close(scaler, next_scaled, close_idx)[0]

last_price = df['Close'].iloc[-1]
last_fx    = df['USDINR'].iloc[-1]
change     = next_price - last_price
change_pct = (change / last_price) * 100

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ── Metric Cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Current Price</div>
      <div class="metric-value gold">₹{last_price:,.0f}</div>
      <div class="metric-label">per 10g</div>
    </div>""", unsafe_allow_html=True)

with col2:
    color = "up" if change >= 0 else "down"
    arrow = "▲" if change >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Predicted Next</div>
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
tab1, tab2, tab3 = st.tabs(["📈 Actual vs Predicted", "📉 Training Loss", "🕯️ Price History"])

with tab1:
    test_dates = df.index[SEQ_LEN + st.session_state.split:]
    fig, ax = dark_fig((13, 4))
    ax.plot(test_dates, y_true, color='#FFD700', lw=1.5, label='Actual (MCX INR)')
    ax.plot(test_dates, y_pred, color='#FF6B35', lw=1.5, ls='--', label='Predicted')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title('Gold Price — Actual vs Predicted (Test Set)', color='#FFD700', fontsize=13)
    ax.set_ylabel('₹ per 10g', color='#aaa', fontsize=11)
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=10)
    ax.grid(alpha=0.1, color='white')
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, ax = dark_fig((13, 4))
    ax.plot(history.history['loss'],     color='#FFD700', label='Train Loss')
    ax.plot(history.history['val_loss'], color='#FF6B35', label='Val Loss')
    ax.set_title('Training & Validation Loss', color='#FFD700', fontsize=13)
    ax.set_xlabel('Epoch', color='#aaa')
    ax.set_ylabel('MSE Loss', color='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=10)
    ax.grid(alpha=0.1, color='white')
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
                        alpha=0.08, color='#FFD700', label='Bollinger Band')

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title(f'Gold Price History — {history_period}', color='#FFD700', fontsize=13)
    ax.set_ylabel('₹ per 10g', color='#aaa', fontsize=11)
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=10, ncol=3)
    ax.grid(alpha=0.1, color='white')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown(
    '<p class="disclaimer">⚠️ This tool is for educational purposes only. '
    'Predictions are not financial advice. Gold prices are influenced by many macro factors '
    'not captured in this model.</p>',
    unsafe_allow_html=True
)
