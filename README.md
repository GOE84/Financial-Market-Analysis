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

## สรุปผลการทำนายราคาทองคำ:

### ภาพรวมโมเดล
โครงการนี้ได้พัฒนาโมเดล Linear Regression เพื่อทำนายราคาทองคำ (`GC=F`) โดยใช้ข้อมูลราคาย้อนหลังระหว่างปี 2020-01-01 ถึง 2024-01-01 

### ขั้นตอนการดำเนินงาน:
1.  **การรวบรวมข้อมูล**: ดึงข้อมูลราคาทองคำในอดีตจาก Yahoo Finance (`yfinance`).
2.  **การสร้างคุณลักษณะ (Feature Engineering)**: คำนวณค่าเฉลี่ยเคลื่อนที่ (Moving Averages) 10 วัน (`MA_10`) และ 50 วัน (`MA_50`), อัตราผลตอบแทนรายวัน (`Daily_Return`) และราคาปิดของวันก่อนหน้า (`Close_Lag1`) นอกจากนี้ยังสร้างตัวแปรเป้าหมาย (`Prediction`) คือราคาปิดของวันถัดไป.
3.  **การทำความสะอาดข้อมูล**: ลบแถวที่มีค่าว่าง (NaN) ซึ่งเกิดขึ้นจากการสร้างคุณลักษณะ.
4.  **การแบ่งข้อมูล**: แบ่งข้อมูลออกเป็นชุดฝึก (80%) และชุดทดสอบ (20%) ตามลำดับเวลา.
5.  **การฝึกโมเดล**: ฝึกโมเดล Linear Regression ด้วยชุดข้อมูลฝึก.
6.  **การประเมินผล**: ประเมินประสิทธิภาพของโมเดลบนชุดข้อมูลทดสอบโดยใช้ Root Mean Squared Error (RMSE).
7.  **การแสดงผล**: พล็อตเปรียบเทียบราคาทองคำจริงกับราคาที่โมเดลทำนายได้.

### ผลการทำนาย
โมเดล Linear Regression มีค่า RMSE อยู่ที่ **15.76 USD** บนชุดข้อมูลทดสอบ ซึ่งหมายถึงค่าเฉลี่ยความคลาดเคลื่อนในการทำนายอยู่ที่ประมาณ 15.76 ดอลลาร์สหรัฐฯ

### ข้อสังเกตจากการแสดงผล
*   โมเดลสามารถติดตามแนวโน้มโดยรวมของราคาทองคำได้ค่อนข้างดี.
*   มีการคาดการณ์ที่ล่าช้าเล็กน้อย โดยเฉพาะในช่วงที่ราคามีความผันผวนสูง.
*   เส้นกราฟที่โมเดลทำนายได้มักจะเรียบกว่าราคาจริง ซึ่งบ่งชี้ว่าโมเดล Linear Regression อาจมีข้อจำกัดในการจับความผันผวนที่รวดเร็วและรุนแรงของตลาด.

### ข้อเสนอแนะและขั้นตอนถัดไป
*   พิจารณาใช้โมเดลที่ซับซ้อนขึ้น (เช่น RandomForest, LSTM) หรือเพิ่มคุณลักษณะ (features) อื่นๆ เพื่อปรับปรุงความแม่นยำในการทำนาย โดยเฉพาะในช่วงที่มีความผันผวนสูง.
*   วิเคราะห์เพิ่มเติมเพื่อระบุสถานการณ์ตลาดที่โมเดลทำงานได้ไม่ดี เพื่อทำความเข้าใจข้อจำกัดของโมเดลอย่างลึกซึ้งยิ่งขึ้น.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/stock-prediction-ml.git](https://github.com/YOUR_USERNAME/stock-prediction-ml.git)
