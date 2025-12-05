<img width="848" height="470" alt="Test1" src="https://github.com/user-attachments/assets/0a634a2b-6568-49cf-8914-f32d3edf4ff1" /># 📈 Stock Price Prediction using Machine Learning

A machine learning project to predict **Apple (AAPL)** stock prices using historical data and Technical Indicators. This project demonstrates the end-to-end data science pipeline, from data ingestion to model evaluation.

<img width="848" height="470" alt="Test1" src="https://github.com/user-attachments/assets/b92749c9-b709-4bef-83ef-8c2046dff069" />

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy)
- **Machine Learning** (Scikit-Learn: Linear Regression)
- **Data Visualization** (Matplotlib)
- **Data Source** (Yahoo Finance via `yfinance`)

## 📊 Methodology
1.  **Data Collection:** Fetched 5 years of historical data (Open, High, Low, Close).
2.  **Feature Engineering:**
    - `MA_10`, `MA_50` (Moving Averages for trend detection)
    - `Daily_Return` (Volatility)
    - `Close_Lag1` (Previous day's price)
3.  **Model Training:** Trained a **Linear Regression** model to predict the *next day's closing price*.
4.  **Evaluation:** Achieved an **RMSE of ~2.12 USD** on the test set (2023-2024).

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/stock-prediction-ml.git](https://github.com/YOUR_USERNAME/stock-prediction-ml.git)
