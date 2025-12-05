# 📈 Stock Price Prediction using Machine Learning

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


<img width="854" height="470" alt="download" src="https://github.com/user-attachments/assets/61b6de95-e2b5-4c3b-805a-9481c867ec3c" />
<img width="854" height="470" alt="download-1" src="https://github.com/user-attachments/assets/8843757a-05cf-4600-8bda-0d82455ebbad" />
<img width="868" height="470" alt="download-2" src="https://github.com/user-attachments/assets/b3a8dd96-02d2-4402-9b8e-f3c9d81121ab" />


## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/stock-prediction-ml.git](https://github.com/YOUR_USERNAME/stock-prediction-ml.git)
