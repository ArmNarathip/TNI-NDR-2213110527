import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------
# Header
# -----------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/05/Meta_Platforms_Inc._logo_%28cropped%29.svg", width=150)

with col2:
    st.title("Meta Platforms (META)")
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 8px;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/a/a4/Flag_of_the_United_States.svg' width='24' height='24'>
            <span style='font-size: 1.1rem;'>NASDAQ Currency in USD</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Load & Prepare Data
# -----------------------------
df = pd.read_excel("Meta Platforms Stock Price History.xlsx")
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change%", "NASDAQ index"]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df.dropna(inplace=True)
df.sort_values("Date", ascending=False, inplace=True)
df_sorted = df.sort_values("Date")

latest_price = df_sorted["Price"].iloc[-1]
max_price = df_sorted["Price"].max()
min_price = df_sorted["Price"].min()
mean_price = df_sorted["Price"].mean()


#call mata_stock_info.txt
with open("meta_stock_info.txt", "r", encoding="utf-8") as f:
    meta_text = f.read()


# -----------------------------
# Sidebar Info
# -----------------------------

with st.sidebar:
    st.markdown(f"""
    ### 📊 Facebook (META) Stock Statistics
    - 📅 Present: **${latest_price:.2f}**
    - 🔺 High: **${max_price:.2f}**
    - 🔻 Low: **${min_price:.2f}**
    - 📈 Mean: **${mean_price:.2f}**
    """)
    st.write("")  # Spacer

    st.title("ℹ️ Stock Information") 
    # แสดงข้อมูลแบบเต็ม
    st.markdown(
        f"""
        <div style="
            background-color: var(--secondary-background-color);
            border-left: 6px solid var(--primary-color);
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            color: var(--text-color);
        ">
            <strong>Meta Platforms, Inc.</strong> {meta_text}
            <br><br>
            <span style="font-size: 0.85rem; ">
                Source: <a href="https://www.nasdaq.com/market-activity/stocks/meta" target="_blank" style="color: var(--primary-color); text-decoration: none;">nasdaq.com</a>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Filter Timeframe
# -----------------------------
timeframes = {
    "1 Day": 1,
    "7 Day": 7,
    "30 Day": 30,
    "90 Day": 90,
    "All": "max"
}
choice = st.selectbox("Time Frame (6 Month)", list(timeframes.keys()), index=1)
filtered_df = df if timeframes[choice] == "max" else df.head(timeframes[choice])

st.markdown(f"#### 📆 Meta Platform Stock Price History: {choice}")
filtered_df = filtered_df.reset_index(drop=True)
filtered_df.index += 1
filtered_df["Date"] = filtered_df["Date"].dt.date
st.dataframe(filtered_df)

# -----------------------------
# Indicator Functions
# -----------------------------
def calculate_macd(df, col='Price'):
    ema12 = df[col].ewm(span=12).mean()
    ema26 = df[col].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal, macd - signal

def calculate_rsi(df, col='Price', window=14):
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100).fillna(method='bfill')  

# -----------------------------
# Chart Display
# -----------------------------
st.title("📈 Facebook (META) Stock Chart")
chart_type = st.selectbox("Select Indicators Chart", ["Linear Regression", "Interactive", "MACD", "RSI"])

st.subheader("Facebook (META) Stock Chart")
if chart_type == "Linear Regression":
    X = df_sorted["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = df_sorted["Price"].values
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)

    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted["Date"], y, label="Actual Closing Price")
    plt.plot(df_sorted["Date"], trend, label="Trend (Linear Regression)", linestyle="--", color="red")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    st.pyplot(plt)

elif chart_type == "Interactive":
    fig = px.line(df, x='Date', y='Price', title='META Stock Price')
    fig.update_layout(xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "MACD":
    macd, signal, hist = calculate_macd(df_sorted)
    fig = go.Figure([
        go.Bar(x=df_sorted['Date'], y=hist, name='Histogram', marker_color='red'),
        go.Scatter(x=df_sorted['Date'], y=macd, name='MACD', line=dict(color='blue')),
        go.Scatter(x=df_sorted['Date'], y=signal, name='Signal', line=dict(color='orange'))
    ])
    fig.update_layout(title='MACD Chart', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "RSI":
    rsi = calculate_rsi(df_sorted)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['Date'], y=rsi, name='RSI', line_color='purple'))
    fig.add_hline(y=70, line_dash='dash', line_color='red')
    fig.add_hline(y=30, line_dash='dash', line_color='green')
    fig.update_layout(title='RSI Chart', yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)