import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Blinkit Pro Dashboard", layout="wide")

st.title("🚀 Blinkit Advanced Analytics + Forecasting Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/blinkit_orders.csv")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

min_date = df["order_date"].min()
max_date = df["order_date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

status = st.sidebar.multiselect(
    "Delivery Status",
    df["delivery_status"].unique(),
    default=df["delivery_status"].unique()
)

filtered = df.copy()

if len(date_range) == 2:
    filtered = filtered[
        (filtered["order_date"] >= pd.to_datetime(date_range[0])) &
        (filtered["order_date"] <= pd.to_datetime(date_range[1]))
    ]

filtered = filtered[filtered["delivery_status"].isin(status)].copy()

# ---------------- KPIs ----------------
col1, col2, col3, col4 = st.columns(4)

total_revenue = filtered["order_total"].sum()
total_orders = filtered.shape[0]
avg_order = filtered["order_total"].mean()

on_time = filtered[filtered["delivery_status"] == "On Time"].shape[0]
on_time_pct = (on_time / total_orders * 100) if total_orders > 0 else 0

col1.metric("💰 Revenue", f"{total_revenue:,.0f}")
col2.metric("📦 Orders", total_orders)
col3.metric("🧾 Avg Order", f"{avg_order:.2f}")
col4.metric("⏱ On-Time %", f"{on_time_pct:.1f}")

st.markdown("---")

# ---------------- DAILY TREND ----------------
daily = filtered.groupby("order_date")["order_total"].sum().reset_index()

st.subheader("📈 Revenue Trend")
fig1 = px.line(daily, x="order_date", y="order_total", markers=True)
st.plotly_chart(fig1, use_container_width=True)

daily["MA7"] = daily["order_total"].rolling(7).mean()

st.subheader("📊 Moving Average (7 Days)")
fig2 = px.line(daily, x="order_date", y=["order_total", "MA7"])
st.plotly_chart(fig2, use_container_width=True)

# ---------------- DELIVERY STATUS ----------------
status_counts = filtered["delivery_status"].value_counts().reset_index()
status_counts.columns = ["status", "count"]

st.subheader("🚚 Delivery Status Distribution")
fig3 = px.bar(status_counts, x="status", y="count", text="count")
fig3.update_traces(textposition="outside")
st.plotly_chart(fig3, use_container_width=True)

# ---------------- DISTRIBUTION ----------------
st.subheader("📊 Order Value Distribution")
fig4 = px.histogram(filtered, x="order_total", nbins=40)
st.plotly_chart(fig4, use_container_width=True)

st.subheader("📦 Outlier Detection")
fig5 = px.box(filtered, y="order_total")
st.plotly_chart(fig5, use_container_width=True)

# ---------------- MONTHLY ----------------
filtered["month"] = filtered["order_date"].dt.month
monthly = filtered.groupby("month")["order_total"].sum().reset_index()

st.subheader("📅 Monthly Revenue")
fig6 = px.bar(monthly, x="month", y="order_total", text="order_total")
fig6.update_traces(textposition="outside")
st.plotly_chart(fig6, use_container_width=True)

# ---------------- WEEKDAY ----------------
filtered["day"] = filtered["order_date"].dt.day_name()
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

weekday = filtered["day"].value_counts().reindex(weekday_order).reset_index()
weekday.columns = ["day", "count"]

st.subheader("📆 Orders by Weekday")
fig7 = px.bar(weekday, x="day", y="count", text="count")
fig7.update_traces(textposition="outside")
st.plotly_chart(fig7, use_container_width=True)

# ---------------- PIE ----------------
st.subheader("🥧 Delivery Performance")
fig8 = px.pie(filtered, names="delivery_status", hole=0.4)
st.plotly_chart(fig8, use_container_width=True)

# ---------------- CUMULATIVE ----------------
daily["cumulative"] = daily["order_total"].cumsum()

st.subheader("📈 Cumulative Revenue")
fig9 = px.area(daily, x="order_date", y="cumulative")
st.plotly_chart(fig9, use_container_width=True)

# ---------------- SEGMENTATION ----------------
bins = [0, 500, 1000, 2000, 5000]
labels = ["Low", "Medium", "High", "Premium"]

filtered["segment"] = pd.cut(filtered["order_total"], bins=bins, labels=labels)

segment = filtered["segment"].value_counts().reset_index()
segment.columns = ["segment", "count"]

st.subheader("👥 Customer Segments")
fig10 = px.bar(segment, x="segment", y="count", text="count")
fig10.update_traces(textposition="outside")
st.plotly_chart(fig10, use_container_width=True)

# ---------------- CATEGORY ----------------
if "category" in filtered.columns:
    st.header("📦 Product & Category Insights")

    cat = filtered.groupby("category")["order_total"].sum().sort_values(ascending=False).reset_index()

    st.subheader("Top Categories by Revenue")
    fig_cat = px.bar(cat.head(10), x="category", y="order_total", text="order_total")
    fig_cat.update_traces(textposition="outside")
    st.plotly_chart(fig_cat, use_container_width=True)

# ---------------- HEATMAP ----------------
st.header("📊 Order Patterns")

heat = filtered.copy()
heat["month"] = heat["order_date"].dt.month
heat["weekday"] = heat["order_date"].dt.day_name()

heatmap = heat.groupby(["weekday","month"]).size().unstack().fillna(0)

fig_heat = px.imshow(heatmap, text_auto=True, aspect="auto")
st.subheader("Orders Heatmap (Weekday vs Month)")
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------- GROWTH ----------------
daily["prev"] = daily["order_total"].shift(1)
daily["growth"] = ((daily["order_total"] - daily["prev"]) / daily["prev"]) * 100

st.subheader("📈 Daily Growth %")
fig_growth = px.line(daily, x="order_date", y="growth")
st.plotly_chart(fig_growth, use_container_width=True)

# ---------------- CORRELATION ----------------
st.header("🔍 Correlation Analysis")

numeric_cols = filtered.select_dtypes(include=np.number)

if len(numeric_cols.columns) > 1:
    corr = numeric_cols.corr()
    fig_corr = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------- FORECAST ----------------
st.header("🤖 Revenue Forecasting (ML)")

forecast_data = daily.copy()
forecast_data["day_num"] = np.arange(len(forecast_data))

X = forecast_data[["day_num"]]
y = forecast_data["order_total"]

model = LinearRegression()
model.fit(X, y)

future_days = 30
future_X = np.arange(len(forecast_data), len(forecast_data)+future_days).reshape(-1,1)
predictions = model.predict(future_X)

future_dates = pd.date_range(
    start=forecast_data["order_date"].max(),
    periods=future_days+1
)[1:]

forecast_df = pd.DataFrame({
    "order_date":future_dates,
    "forecast":predictions
})

st.subheader("🔮 Forecast Next 30 Days")

fig11 = px.line(daily, x="order_date", y="order_total", title="Actual vs Forecast")
fig11.add_scatter(
    x=forecast_df["order_date"],
    y=forecast_df["forecast"],
    mode="lines",
    name="Forecast",
    line=dict(dash="dash")
)

st.plotly_chart(fig11, use_container_width=True)

# ---------------- TOP ORDERS ----------------
st.subheader("🏆 Top Orders")
top = filtered.sort_values("order_total", ascending=False).head(10)
st.dataframe(top)