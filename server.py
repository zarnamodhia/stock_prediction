import streamlit as st
import yfinance as yf
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("stock_model.pkl")

def compute_RSI(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def make_features(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    data['Return'] = data['Close'].pct_change()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['RSI'] = compute_RSI(data['Close'])
    return data.dropna()

st.title("📈 Stock Investment Predictor with ML")

ticker = st.selectbox("Choose a stock", ["AAPL","MSFT","GOOG","TSLA","AMZN"])

if st.button("Predict"):
    data = make_features(ticker)
    latest_rsi = data['RSI'].iloc[-1]  
    if latest_rsi > 70:
        st.write(f"⚠️ RSI is {latest_rsi:.2f} → Stock may be overbought")
    elif latest_rsi < 30:
        st.write(f"✅ RSI is {latest_rsi:.2f} → Stock may be oversold")
    else:
        st.write(f"RSI is {latest_rsi:.2f} → Neutral zone")

    features = data[['Return','SMA_5','SMA_10','RSI']].iloc[-1:]
    pred = model.predict(features)[0]

    st.subheader(f"Prediction for {ticker}:")
    st.success("✅ BUY" if pred==1 else "❌ DON'T BUY")

    st.subheader("Stock Price with Moving Averages")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['SMA_5'], label='SMA 5')
    ax.plot(data.index, data['SMA_10'], label='SMA 10')
    ax.legend()
    st.pyplot(fig)

    st.subheader("RSI (Relative Strength Index)")

    fig3, ax3 = plt.subplots()
    ax3.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax3.axhline(70, color='red', linestyle='--', label='Overbought')
    ax3.axhline(30, color='green', linestyle='--', label='Oversold')
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("Feature Importance (Model Explanation)")
    importance = model.feature_importances_
    feat_names = ['Return','SMA_5','SMA_10','RSI']
    fig3, ax3 = plt.subplots()
    ax3.bar(feat_names, importance, color="orange")
    ax3.set_ylabel("Importance")
    st.pyplot(fig3)
