import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# หัวเรื่อง 
col1, col2 = st.columns([1, 5])  # ปรับขนาดตามความเหมาะสม

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/05/Meta_Platforms_Inc._logo_%28cropped%29.svg", width=150)

with col2:
    st.title("Meta Platforms (META)")
    
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 8px; font-size: 1.1rem;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/a/a4/Flag_of_the_United_States.svg' width='24' height='24'>
            <span>NASDAQ Currency in USD</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# โหลดข้อมูลจาก Excel
df = pd.read_excel(r"C:\\NDR Project Web\\TNI-NDR-2213110527\\Meta Platforms Stock Price History.xlsx")
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change%", "NASDAQ index"]


# แปลง Date เป็น datetime
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["Date"])
df = df.dropna()
df = df.sort_values("Date", ascending=False)
df_sorted = df.sort_values("Date")

latest_price = df_sorted["Price"].iloc[-1]
max_price = df_sorted["Price"].max()
min_price = df_sorted["Price"].min()
mean_price = df_sorted["Price"].mean()

# สไลด์บาร์       
with st.sidebar:
        st.title("ℹ️Stock Infomations")
        st.markdown("""
        <div style="
            background-color: #f9f9f9;
            border-left: 6px solid #2e7bcf;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            font-size: 1.05rem;
            line-height: 1.6;
        ">
            <strong>Meta Platforms, Inc.</strong> บริษัทแม่ของ Facebook ดำเนินธุรกิจที่ครอบคลุมแพลตฟอร์มโซเชียลมีเดียชั้นนำอย่าง Facebook, Instagram, Messenger และ WhatsApp โดยสร้างรายได้หลักจากการโฆษณาดิจิทัล พร้อมมุ่งเน้นการพัฒนาเทคโนโลยี Metaverse ผ่าน Reality Labs ซึ่งรวมถึงฮาร์ดแวร์ AR/VR อย่าง Oculus และแพลตฟอร์มโลกเสมือนจริง เพื่อสนับสนุนนวัตกรรมด้านการเชื่อมต่อและการทำงานแห่งอนาคต.
            <br><br>
            <span style="font-size: 0.85rem; color: #555;">
                ที่มา: <a href="https://www.liberator.co.th/article/view/us-stock-meta" target="_blank" style="color: #2e7bcf; text-decoration: none;">https://www.liberator.co.th/article/view/us-stock-meta</a>
            </span>
        </div>
        """, unsafe_allow_html=True)
        

        st.header("")
        st.markdown(f"""
    ### 📊 สถิติราคาหุ้น META Platforms, Inc.
    - 📅 ราคาปัจจุบัน: **${latest_price:.2f}**
    - 🔺 ราคาสูงสุด: **${max_price:.2f}**
    - 🔻 ราคาต่ำสุด: **${min_price:.2f}**
    - 📈 ราคาเฉลี่ย: **${mean_price:.2f}**
    """)
    


# ตัวเลือกช่วงเวลา
timeframes = {
    "1 วันย้อนหลัง": 1,
    "7 วันย้อนหลัง": 7,
    "30 วันย้อนหลัง": 30,
    "90 วันย้อนหลัง": 90,
    "ทั้งหมด": "max"
}

default_index = list(timeframes.keys()).index("7 วันย้อนหลัง")

# เลือกช่วงเวลา
col1, col2 = st.columns([1, 3])
with col1:
    choice = st.selectbox("เลือกช่วงเวลา", options=list(timeframes.keys()), index=default_index, label_visibility="collapsed")
    if timeframes[choice] == "max":
        filtered_df = df.copy()
    else:
        n = timeframes[choice]
        filtered_df = df.iloc[:n]

with col2:
    st.markdown(f"<div style='display: flex; align-items: center; height: 100%; font-size: 1.3rem; font-weight: bold;'>◀️ข้อมูลย้อนหลัง: {choice}</div>", unsafe_allow_html=True)

filtered_df = filtered_df.reset_index(drop=True)
filtered_df.index += 1
filtered_df["Date"] = filtered_df["Date"].dt.date  # แสดงเฉพาะวันที่
st.dataframe(filtered_df)


# ฟังก์ชันคำนวณ MACD
def calculate_macd(df, price_col='Price'):
    df['EMA12'] = df[price_col].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df[price_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    return df

# ฟังก์ชันคำนวณ RSI
def calculate_rsi(df, price_col='Price', window=14):
    delta = df[price_col].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # ป้องกันหารด้วย 0
    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.clip(0, 100).fillna(method='bfill')  # fill ช่วงต้นกราฟ

    df['RSI'] = rsi
    return df


# ฟังก์ชันMACD Chart
def plot_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Date'], y=df['Histogram'], name='Histogram', marker_color='red', opacity=0.6))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal', line=dict(color='orange')))
    fig.update_layout(title='MACD (12, 26, 9)', xaxis_title='Date', yaxis_title='MACD', hovermode='x unified')
    return fig

# ฟังก์ชันRSI Chart
def plot_rsi(df):
    fig = go.Figure ()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
    fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought (70)')
    fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold (30)')
    fig.update_layout(title='RSI (14 วัน)', xaxis_title='Date', yaxis_title='RSI', yaxis_range=[0, 100], hovermode='x unified')
    return fig

st.title("📈แนวโน้มราคาหุ้น (Chart)")

# ตัวเลือกดูกราฟ
chart_option = st.selectbox("", [
    "Linear Regression Chart", 
    "Interactive Chart", 
    "MACD Chart",
    "RSI Chart"
])

# แสดงกราฟตามตัวเลือก
st.subheader("Facebook (META) Stock Chart 6 Month Before")
if chart_option == "Linear Regression Chart":
    X = df_sorted["Date"].apply(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = df_sorted["Price"].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted["Date"], y, label="Actual Closing Price")
    plt.plot(df_sorted["Date"], trend, label="Trend (Linear Regression)", linestyle="--", color="red")
    plt.title("META Closing Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (US Dollar)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

elif chart_option == "Interactive Chart":
    fig = px.line(df, x='Date', y='Price', title='ยอดขายรายวัน (60 วัน)')
    fig.update_traces(mode='lines+markers', hovertemplate='Date: %{x|%Y-%m-%d}<br>Closing Price: %{y} $USD')
    fig.update_layout(xaxis_title='Date', yaxis_title='Price', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

elif chart_option == "MACD Chart":
    df_macd = calculate_macd(df_sorted)
    st.plotly_chart(plot_macd(df_macd), use_container_width=True)

elif chart_option == "RSI Chart":
    df_rsi = calculate_rsi(df_sorted)
    st.plotly_chart(plot_rsi(df_rsi), use_container_width=True)


