import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib  # เอาไว้ save โมเดลเป็นไฟล์
import os

# --- Configuration ---
DATA_DIR = "stock_data"
MODEL_DIR = "models"
STOCK_SYMBOL = "AAPL"

# สร้างโฟลเดอร์เก็บโมเดล
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_model(symbol):
    print(f"🧠 Training model for {symbol}...")
    
    # 1. Load Data
    file_path = os.path.join(DATA_DIR, f"{symbol}_history.csv")
    if not os.path.exists(file_path):
        print("❌ Error: Data file not found. Run data_collector.py first.")
        return

    df = pd.read_csv(file_path)
    df = df[['Date', 'Close']] # เราจะใช้แค่ราคาปิดก่อน
    
    # 2. Feature Engineering (เตรียมข้อมูลสอน)
    # โจทย์: ใช้ "ราคาปิดวันนี้" (X) เพื่อทำนาย "ราคาปิดพรุ่งนี้" (y)
    # สร้างคอลัมน์ 'Prediction' โดยการเลื่อนข้อมูลขึ้น 1 วัน
    df['Prediction'] = df[['Close']].shift(-1)
    
    # ตัดแถวสุดท้ายออก (เพราะไม่มีข้อมูลวันพรุ่งนี้ให้สอน)
    data = df.dropna()

    X = np.array(data[['Close']])
    y = np.array(data['Prediction'])

    # 3. Split Data (แบ่งข้อสอบ)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 4. Train Model (เริ่มสอน)
    model = LinearRegression()
    model.fit(x_train, y_train)

    # 5. Evaluate (ตรวจข้อสอบ)
    score = model.score(x_test, y_test)
    print(f"✅ Model Trained! Accuracy (R^2): {score:.4f}")

    # 6. Save Model (บันทึกสมองลงไฟล์)
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

if __name__ == "__main__":
    train_model(STOCK_SYMBOL)