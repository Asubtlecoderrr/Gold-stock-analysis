# Gold Price Predictor — Indian Market (MCX)
A full-stack Streamlit application that predicts gold prices (₹ per 10g) using an LSTM deep learning model and provides real-time alert notifications via email.

## Dashboard
<img width="1512" height="861" alt="Screenshot 2026-04-14 at 3 45 32 PM" src="https://github.com/user-attachments/assets/b3832a8c-0312-4048-b40d-a5d969266081" />

## Alert UI
<img width="1512" height="861" alt="Screenshot 2026-04-14 at 3 45 48 PM" src="https://github.com/user-attachments/assets/3da5085b-f836-4e97-a49c-152feef97c2d" />

## Received Mail
<img width="1228" height="346" alt="Screenshot 2026-04-14 at 3 46 05 PM" src="https://github.com/user-attachments/assets/fca307f1-d519-476d-8df9-9c9c4ea049c3" />

---

## Features

* 📈 **Gold Price Prediction**

  * LSTM-based deep learning model
  * Predicts next-day gold prices using historical data

* ⚡ **Live Price Tracking**

  * Fetches real-time gold price and USD/INR rate
  * Auto-refresh support

* 🔔 **Smart Alerts System**

  * Price Above / Below alerts
  * Predicted price alerts
  * Email notifications (via Resend API)

* 📊 **Interactive Dashboard**

  * Actual vs Predicted graph
  * Technical indicators (MA, RSI, Bollinger Bands)
  * Training loss visualization

---

## Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **ML Model**: TensorFlow (LSTM)
* **Data Source**: Yahoo Finance (yfinance)
* **Email Service**: Resend API
* **Database**: Session-based (Streamlit state)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Asubtlecoderrr/Gold-stock-analysis.git
cd Gold-stock-analysis
```

**2. Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

**4. Environment Setup**

Create a `.env` file in the root directory:

```env
RESEND_API_KEY=your_resend_api_key
```

---

**5. Run the App**

```bash
streamlit run app.py
```

---

## Email Alerts (Resend)

* Uses `onboarding@resend.dev` as default sender
* Only verified emails can receive alerts (in free plan)
* Ensure your email is same as your Resend account

---

## Model Details

* **Architecture**:

  * LSTM (128 units)
  * Dropout
  * LSTM (64 units)
  * Dense layers

* **Input Features**:

  * OHLC prices
  * Volume
  * USD/INR
  * Moving averages
  * RSI, Bollinger Bands
  * Volatility

---

## 📄 License

This project is for educational purposes only.
