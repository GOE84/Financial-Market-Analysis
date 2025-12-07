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
<img width="854" height="470" alt="download-2" src="https://github.com/user-attachments/assets/b3a8dd96-02d2-4402-9b8e-f3c9d81121ab" />

# 📈 Gold Price Prediction with Linear Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20|%20yfinance%20|%20Pandas-orange)

## 📖 Project Overview
This project implements a **Linear Regression** machine learning model to predict gold prices (`GC=F`) based on historical market data from **January 1, 2020, to January 1, 2024**.

The goal is to demonstrate the end-to-end process of a financial ML project, from data acquisition using Yahoo Finance to feature engineering and model evaluation.

## ⚙️ Methodology

1.  **Data Collection**: 
    - Retrieved historical data using the `yfinance` API.
2.  **Feature Engineering**:
    - Calculated **Moving Averages** (10-day and 50-day) to identify trends.
    - Computed **Daily Returns** and **Lagged Closing Prices** (`Close_Lag1`) to use past performance as a predictor.
3.  **Data Processing**: 
    - Cleaned `NaN` values resulting from rolling calculations.
    - Split data into **Training (80%)** and **Testing (20%)** sets chronologically to prevent data leakage.
4.  **Modeling**: 
    - Trained a standard Linear Regression model.
5.  **Evaluation**: 
    - Performance measured using **Root Mean Squared Error (RMSE)**.

## 📊 Results

The model was evaluated on unseen test data with the following results:

- **RMSE:** `$15.76 USD`
- **Key Observations:**
    - The model successfully captures the general long-term trend of gold prices.
    - As expected with Linear Regression, there is a slight lag during high-volatility periods, and the prediction line is smoother than the actual price action.

## 🚀 How to Run

### Prerequisites
Make sure you have Python installed.

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/stock-prediction-ml.git](https://github.com/YOUR_USERNAME/stock-prediction-ml.git)
   cd stock-prediction-ml