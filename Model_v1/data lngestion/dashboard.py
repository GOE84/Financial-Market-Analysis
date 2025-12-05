import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import joblib
import numpy as np
from datetime import timedelta

# --- Configuration ---
DATA_DIR = "stock_data"
MODEL_DIR = "models"

st.set_page_config(page_title="Stock AI Trader", layout="wide", page_icon="📈")

# --- UI Header ---
st.title("📈 Professional Stock AI Trader")

# --- Helper Functions ---
def load_data(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}_history.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_model(symbol):
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# --- Sidebar ---
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()

# --- Main Logic ---
if st.sidebar.button("Analyze & Predict"):
    df = load_data(symbol)
    
    if df is not None:
        # Show Current Data
        last_close = df['Close'].iloc[-1]
        last_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
        
        # Load Model
        model = load_model(symbol)
        predicted_price = 0
        is_bullish = False
        
        if model:
            prediction_input = np.array([[last_close]]) 
            predicted_price = model.predict(prediction_input)[0]
            change_percent = ((predicted_price - last_close) / last_close) * 100
            is_bullish = change_percent > 0
            
            # Metric Cards
            col1, col2, col3 = st.columns(3)
            col1.metric("Date", last_date)
            col1.metric("Close Price", f"${last_close:.2f}")
            col3.metric("AI Prediction (Next Day)", f"${predicted_price:.2f}", f"{change_percent:.2f}%")

        # --- 📊 Creating The Chart ---
        st.subheader(f"{symbol} Daily Chart")
        
        fig = go.Figure()

        # 1. Candlestick
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Price'
        ))
        
        # 2. AI Prediction Line (เส้นทำนาย)
        if model:
            next_date = df['Date'].iloc[-1] + timedelta(days=1)
            # ถ้าเป็นขาขึ้นให้เส้นเขียว ขาลงให้เส้นแดง
            line_color = '#00ff00' if is_bullish else '#ff0000'
            
            fig.add_trace(go.Scatter(
                x=[df['Date'].iloc[-1], next_date], 
                y=[last_close, predicted_price],
                mode='lines+markers+text',
                name='AI Forecast',
                line=dict(color=line_color, width=2, dash='dash'),
                marker=dict(size=8),
                text=[f"", f"{predicted_price:.2f}"],
                textposition="top right",
                textfont=dict(color=line_color)
            ))

        # 3. Layout Settings (ตั้งค่าให้ใช้ง่าย)
        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="MAX")
                    ]),
                    font=dict(color="black") # สีตัวหนังสือปุ่ม
                ),
                rangeslider=dict(visible=False), # ซ่อนแถบด้านล่าง (เพราะมีปุ่มแล้วจะได้ไม่รก)
                type="date",
                # *** สำคัญ: บังคับให้เริ่มซูมที่ 90 วันล่าสุด ***
                range=[df['Date'].iloc[-90], next_date if model else df['Date'].iloc[-1]] 
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False # อนุญาตให้ลากแกน Y ขึ้นลงได้
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error(f"Data not found for {symbol}")
else:
    st.info("👈 Press 'Analyze & Predict' to start")