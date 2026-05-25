# ATM Cash Withdrawal Forecasting App

This project is a Streamlit-based machine learning application that predicts ATM cash withdrawal demand for the next 7 days using a trained XGBoost model.

---

## 📁 Project Structure

📦 ATM_Forecasting
├── 📄 app.py                          # Streamlit UI application

├── 📄 prediction_version2.py          # Prediction pipeline logic

├── 📄 calenter.py                     # Feature engineering (latest calendar features)

├── 📦 artifact
  └── 📁 encoders                    # Label encoders for categorical features

├── 📦 merged_data
  └── 📄 last_40_days.csv            # Recent 40 days ATM data

├── 📄 XgbVersion_2_model.pkl          # Trained XGBoost model
└── 📁 Files_Directory (optional)      # Raw or historical data files

---

## 🚀 Features

- Predicts ATM cash demand for next 7 days
- Uses XGBoost regression model
- Time-series feature engineering (trend, lag, calendar features)
- ATM-wise prediction support
- Streamlit interactive dashboard

---

## 🧠 Model Details

- Algorithm: XGBoost Regressor
- Input: Last 40 days ATM transaction patterns + engineered features
- Output: Future cash withdrawal demand (7-day forecast)

---

## 📊 Feature Engineering

Handled in `calenter.py`:
- Holiday features
- Weekend effects
- Date-based features (day, month, weekday)
- Recent historical patterns (last 40 days)

---

## ▶️ How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
Run Streamlit app:
streamlit run app.py
Open browser:
http://localhost:8501
