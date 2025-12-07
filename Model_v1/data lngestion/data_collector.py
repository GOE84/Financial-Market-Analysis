import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# --- Configuration ---
STOCK_SYMBOL = "AAPL"  # เปลี่ยนเป็นหุ้นที่ชอบได้ เช่น "TSLA", "PTT.BK" (หุ้นไทยต้องมี .BK)
DATA_DIR = "stock_data" # โฟลเดอร์สำหรับเก็บไฟล์
LOG_FILE = "system_logs.txt"

# สร้างโฟลเดอร์ถ้ายังไม่มี
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def log_message(message):
    """ฟังก์ชันสำหรับบันทึก Log การทำงาน"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def fetch_and_save_data(symbol):
    """
    ดึงข้อมูลหุ้นและบันทึกเป็น CSV (จำลอง Data Lake)
    """
    log_message(f"🚀 Starting data collection for {symbol}...")
    
    try:
        # 1. ดึงข้อมูลจาก Yahoo Finance
        # period="max" คือดึงย้อนหลังให้ไกลที่สุดเท่าที่มี
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max")
        
        if df.empty:
            log_message(f"❌ Error: No data found for {symbol}")
            return

        # 2. ปรับแต่งข้อมูลเล็กน้อย
        df.reset_index(inplace=True)
        # แปลงวันที่ให้เป็นรูปแบบมาตรฐาน YYYY-MM-DD
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # 3. Save เป็น CSV (ในอนาคตเราจะเปลี่ยนตรงนี้เป็น save ลง Database)
        # ตั้งชื่อไฟล์แบบมีวันที่กำกับ หรือ save ทับไฟล์เดิมก็ได้ (ที่นี้ขอ save ทับเพื่อให้ง่ายต่อการเทรน)
        file_path = os.path.join(DATA_DIR, f"{symbol}_history.csv")
        df.to_csv(file_path, index=False)
        
        log_message(f"✅ Success: Data saved to {file_path} ({len(df)} records)")
        
    except Exception as e:
        log_message(f"❌ Critical Error: {str(e)}")

# --- Main Execution ---
if __name__ == "__main__":
    fetch_and_save_data(STOCK_SYMBOL)