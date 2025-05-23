import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# หัวเรื่อง
st.markdown("# Meta Platforms (META)")
st.write("NASDAQ Currency in USD")

# โหลดข้อมูลจาก Excel
df = pd.read_excel(r"C:\NDR Project Web\TNI-NDR-2213110527\Meta Platforms Stock Price History.xlsx")
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change%", "NASDAQ index"]

# แปลง Date เป็น datetime
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["Date"])  # ลบแถวที่ Date เป็น NaN
df["Date"] = df["Date"].dt.date  # ให้แสดงวันที่แบบสั้น
df = df.dropna()  # ลบ NaN อื่นๆ
df = df.sort_values("Date", ascending=False)  # เรียงใหม่จากวันล่าสุด

# Time frame options ที่คุณต้องการ
timeframes = {
    "1 วันย้อนหลัง": 1,
    "7 วันย้อนหลัง": 7,
    "30 วันย้อนหลัง": 30,
    "90 วันย้อนหลัง": 90,
    "ทั้งหมด": "max"
}

default_index = list(timeframes.keys()).index("7 วันย้อนหลัง")

# ให้ผู้ใช้เลือกช่วงเวลา
col1, col2 = st.columns([1, 3])  
with col1:
    choice = st.selectbox("เลือกช่วงเวลา", options=list(timeframes.keys()),
                          index=default_index
                          ,label_visibility="collapsed")

    # กรองข้อมูลตามช่วงที่เลือก
    if timeframes[choice] == "max":
        filtered_df = df.copy()
    else:
        n = timeframes[choice]
        # ใช้ 90 วันทำการล่าสุด (ข้อมูลจริง)
        filtered_df = df.iloc[:n]  # df ถูก sort จากใหม่ไปเก่าแล้ว

# แสดงผล
with col2:
   st.markdown(
        f"""
        <div style='display: flex; align-items: center; height: 100%; font-size: 1.3rem; font-weight: bold;'>
            ข้อมูลย้อนหลัง: {choice}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
filtered_df = filtered_df.reset_index(drop=True)
filtered_df.index += 1  # ให้เริ่มแสดง index ที่ 1
st.dataframe(filtered_df)


# -------------------------------
# กราฟแนวโน้มราคาหุ้น
# -------------------------------

st.subheader("Facebook (META) Stock Chart 6 Month Before")

# เรียงข้อมูล
df_sorted = df.sort_values("Date")

# เตรียมข้อมูลสำหรับ Linear Regression
X = df_sorted["Date"].apply(lambda x: x.toordinal()).values.reshape(-1, 1)
y = df_sorted["Price"].values

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

# วาดกราฟ
plt.figure(figsize=(12, 6))
plt.plot(df_sorted["Date"], y, label="Actual Closing Price")
plt.plot(df_sorted["Date"], trend, label="Trend (Linear Regression)", linestyle="--", color="red")
plt.title("META Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Closing Price (US Dollar)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ใช้ใน Streamlit
st.pyplot(plt)