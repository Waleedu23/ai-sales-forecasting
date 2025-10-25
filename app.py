# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Forecasting — AI System", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- Header ----------
st.markdown("""
# 📊 AI-Based Sales Forecasting System
*Upload your historical sales data in CSV format to generate a forecast.*

This app trains a Random Forest model using time-based and lag features, then predicts future sales with a confidence interval.
""")

# ---------- Sidebar: Controls ----------
with st.sidebar.form(key="controls"):
    st.header("Model Controls")
    uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])
    forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30)
    adaptive_lag = st.slider("Max lag (days)", min_value=3, max_value=30, value=14)
    test_size = st.slider("Test set proportion", min_value=0.05, max_value=0.4, value=0.2)
    n_estimators = st.number_input("Random Forest: n_estimators", min_value=10, max_value=1000, value=200, step=10)
    max_depth = st.number_input("Random Forest: max_depth (0 = None)", min_value=0, max_value=50, value=10, step=1)
    run_button = st.form_submit_button("Train & Forecast")

# ---------- Helper functions ----------
def auto_detect_columns(df):
    date_col = None
    sales_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_col = col
            break
    if date_col is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col], errors="raise")
                date_col = col
                break
            except Exception:
                continue

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if any(k in col.lower() for k in ("sale", "amount", "revenue", "total", "qty")):
            sales_col = col
            break
    if sales_col is None and len(numeric_cols) > 0:
        sales_col = numeric_cols[0]

    return date_col, sales_col

def create_time_features(df_index):
    return pd.DataFrame({
        "day_of_week": df_index.dayofweek,
        "day_of_month": df_index.day,
        "month": df_index.month,
        "year": df_index.year
    }, index=df_index)

def build_features(target_series, max_lag):
    df = pd.DataFrame({ "y": target_series }).sort_index()
    time_feats = create_time_features(df.index)
    df = df.join(time_feats)
    for i in range(1, max_lag + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df = df.dropna()
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y

def forecast_with_model(model, last_values_deque, start_date, features_template, horizon):
    preds = []
    dates = []
    lag_size = len(last_values_deque)
    cur_lags = deque(last_values_deque, maxlen=lag_size)
    cur_date = pd.to_datetime(start_date)
    for _ in range(horizon):
        cur_date += timedelta(days=1)
        feat = {
            "day_of_week": cur_date.dayofweek,
            "day_of_month": cur_date.day,
            "month": cur_date.month,
            "year": cur_date.year
        }
        for j in range(1, lag_size + 1):
            feat[f"lag_{j}"] = cur_lags[j-1]
        X_pred = pd.DataFrame([feat])
        pred = model.predict(X_pred)[0]
        preds.append(pred)
        dates.append(cur_date)
        cur_lags.appendleft(pred)
    return dates, preds

# ---------- Main flow ----------
if uploaded_file is None:
    st.warning("Please upload a CSV file from the sidebar to begin.")
    st.stop()

# Load data
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]
st.subheader("Preview uploaded data")
st.dataframe(df.head(10))

# Auto-detect columns
date_col, sales_col = auto_detect_columns(df)

col1, col2 = st.columns([1,1])
with col1:
    st.write("*Detected Date column:*", date_col if date_col else "None")
with col2:
    st.write("*Detected Sales column:*", sales_col if sales_col else "None")

if date_col is None or sales_col is None:
    st.info("Automatic detection failed. Please select the correct columns.")
    date_col = st.selectbox("Select Date column", options=[None] + list(df.columns), index=1 if date_col else 0)
    sales_col = st.selectbox("Select Sales (numeric) column", options=[None] + list(df.columns), index=1 if sales_col else 0)

if date_col is None or sales_col is None:
    st.error("Date or Sales column not selected. Please provide appropriate columns.")
    st.stop()

# Parse types
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
df = df.dropna(subset=[date_col, sales_col])
if df.empty:
    st.error("No valid rows after parsing date and sales columns.")
    st.stop()

df = df.groupby(date_col)[sales_col].sum().rename_axis("date").reset_index()
df = df.sort_values("date").set_index("date")
st.success(f"Using '{date_col}' as date and '{sales_col}' as sales column (aggregated daily).")

# Adaptive lag
data_size = len(df)
recommended_lag = min(adaptive_lag, max(3, data_size // 10))
recommended_lag = min(recommended_lag, max(1, data_size - 2))
st.info(f"Using lag = {recommended_lag} (adaptive). Data points: {data_size}")

# Build features
X, y = build_features(df[sales_col], recommended_lag)
if len(X) < 10:
    st.warning("Less than 10 rows after feature generation. Forecast quality may be poor.")

# Train/test split
split_index = int(len(X) * (1 - test_size))
if split_index < 2:
    st.error("Train set too small. Reduce test size or upload more data.")
    st.stop()
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

if not run_button:
    st.info("Click 'Train & Forecast' in the sidebar to run the model.")
    st.stop()

# Train model
with st.spinner("Training Random Forest model..."):
    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=(None if int(max_depth) == 0 else int(max_depth)),
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

# Evaluate
train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

st.subheader("Model Performance")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Train MAE", f"{train_mae:.2f}")
mcol2.metric("Test MAE", f"{test_mae:.2f}")
mcol3.metric("Train RMSE", f"{train_rmse:.2f}")
mcol4.metric("Test RMSE", f"{test_rmse:.2f}")

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
st.subheader("Feature Importance")
fig_imp = go.Figure(go.Bar(x=feat_imp.values, y=feat_imp.index, orientation='h'))
fig_imp.update_layout(height=400, margin=dict(l=150))
st.plotly_chart(fig_imp, use_container_width=True)

# Prepare lags
last_date = df.index.max()
most_recent_series = df[sales_col].sort_index()
last_lags = [most_recent_series.iloc[-i] if -i >= -len(most_recent_series) else most_recent_series.iloc[0]
             for i in range(1, recommended_lag + 1)]
last_lags_deque = deque(last_lags, maxlen=recommended_lag)

# Forecast
future_dates, future_preds = forecast_with_model(
    rf, last_lags_deque, start_date=last_date, features_template=None, horizon=forecast_days
)

# Confidence interval
conf = test_rmse * 1.96 if not np.isnan(test_rmse) else 0.0
lower = np.array(future_preds) - conf
upper = np.array(future_preds) + conf

# Plot results
st.subheader("Historical & Forecasted Sales")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[sales_col], mode="lines", name="Historical"))
fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers", name="Forecast",
                         line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=future_dates + future_dates[::-1],
                         y=np.concatenate([upper, lower[::-1]]),
                         fill='toself',
                         fillcolor='rgba(0,100,200,0.2)',
                         line=dict(color='rgba(255,255,255,0)'),
                         hoverinfo="skip",
                         showlegend=True,
                         name="95% CI"))
fig.update_layout(xaxis_title="Date", yaxis_title="Sales", height=550, margin=dict(t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# Forecast table
forecast_df = pd.DataFrame({
    "Date": pd.to_datetime(future_dates),
    "Forecast": np.round(future_preds, 2),
    "Lower (95%)": np.round(lower, 2),
    "Upper (95%)": np.round(upper, 2),
})
forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m-%d")
st.subheader("Forecasted Values")
st.dataframe(forecast_df)

csv = forecast_df.to_csv(index=False)
st.download_button("Download forecast CSV", data=csv, file_name="sales_forecast.csv", mime="text/csv")

st.success("✅ Forecast complete. Modify settings in the sidebar and re-run for different results.")
