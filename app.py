from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "daily_power.csv"
MODEL_PATH = APP_DIR / "power_model.pkl"
TARGET = "Global_active_power"
FEATURES = ["day_of_week", "month", "is_weekend", "lag_1", "lag_7", "rolling_7"]

st.set_page_config(
    page_title="Wattwise | Household power forecast",
    page_icon=":material/electric_bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_daily_data(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "date"
    frame = frame.sort_index()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=[TARGET])
    if not set(FEATURES).issubset(frame.columns):
        frame = add_features(frame)
    return frame


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified_time: float):
    del modified_time
    return joblib.load(path)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["day_of_week"] = result.index.dayofweek
    result["month"] = result.index.month
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["lag_1"] = result[TARGET].shift(1)
    result["lag_7"] = result[TARGET].shift(7)
    result["rolling_7"] = result[TARGET].shift(1).rolling(7).mean()
    return result.dropna(subset=FEATURES)


def evaluate_model(frame: pd.DataFrame, model) -> tuple[pd.DataFrame, dict[str, float]]:
    usable = frame.dropna(subset=FEATURES + [TARGET]).copy()
    split = int(len(usable) * 0.85)
    test = usable.iloc[split:].copy()
    test["Predicted"] = model.predict(test[FEATURES])
    metrics = {
        "mae": mean_absolute_error(test[TARGET], test["Predicted"]),
        "rmse": np.sqrt(mean_squared_error(test[TARGET], test["Predicted"])),
        "r2": r2_score(test[TARGET], test["Predicted"]),
    }
    return test, metrics


def fmt_kwh(value: float) -> str:
    return f"{value:,.1f} kWh"


if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("The model artifacts are missing. Run `python train.py` from the project folder first.")
    st.stop()

daily = load_daily_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
evaluation, metrics = evaluate_model(daily, model)
latest_value = float(daily[TARGET].iloc[-1])
recent_average = float(daily[TARGET].tail(7).mean())
peak_value = float(daily[TARGET].max())

with st.sidebar:
    st.markdown("## :material/electric_bolt: Wattwise")
    st.caption("Household energy forecasting workspace")
    view = st.selectbox(
        "Workspace view",
        ["Overview", "Make a forecast", "Model performance", "Data explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Data window: {daily.index.min():%d %b %Y} to {daily.index.max():%d %b %Y}")
    st.caption(f"{len(daily):,} daily observations")

st.title("Household power forecast", anchor=False)
st.caption("A practical next-day estimate built from historical household electricity demand.")

with st.container(horizontal=True):
    st.badge("Model online", icon=":material/check_circle:", color="green")
    st.badge("Random forest", icon=":material/forest:", color="blue")
    st.badge("Daily kWh", icon=":material/monitoring:", color="orange")

if view == "Overview":
    st.header("Demand at a glance", icon=":material/dashboard:")
    with st.container(horizontal=True):
        st.metric("Latest daily usage", fmt_kwh(latest_value), border=True)
        st.metric("7-day average", fmt_kwh(recent_average), border=True)
        st.metric("Historical peak", fmt_kwh(peak_value), border=True)
        st.metric("Validation MAE", fmt_kwh(metrics["mae"]), border=True)

    chart_col, insight_col = st.columns([1.7, 1], gap="large")
    with chart_col:
        with st.container(border=True):
            st.subheader("Recent demand", divider="gray")
            st.line_chart(daily[[TARGET]].tail(90), y=TARGET, height=340)
    with insight_col:
        with st.container(border=True):
            st.subheader("What the model sees", divider="gray")
            st.write("The forecast combines calendar signals with recent demand:")
            for label in ["Day of week and month", "Yesterday's usage", "Usage from 7 days ago", "Trailing 7-day average"]:
                st.markdown(f"- {label}")
            st.caption("The validation score is calculated on the final 15% of the time series, which is never used for training.")

elif view == "Make a forecast":
    st.header("Make a forecast", icon=":material/online_prediction:")
    st.write("Adjust the recent usage signals, then generate an estimate for a future date.")
    last_row = daily.iloc[-1]
    with st.form("forecast_form", border=True):
        input_col, cost_col = st.columns(2, gap="large")
        with input_col:
            target_date = st.date_input("Date to predict", value=daily.index[-1].date() + pd.Timedelta(days=1))
            lag_1 = st.number_input("Yesterday's usage (kWh)", min_value=0.0, value=float(last_row[TARGET]), step=0.1)
            lag_7 = st.number_input("Usage 7 days ago (kWh)", min_value=0.0, value=float(daily[TARGET].iloc[-7]), step=0.1)
            rolling_7 = st.number_input("7-day average (kWh)", min_value=0.0, value=recent_average, step=0.1)
        with cost_col:
            tariff = st.number_input("Electricity tariff (currency/kWh)", min_value=0.0, value=0.15, step=0.01, format="%.2f")
            st.caption("Tariff is optional and is used only to estimate the day's cost.")
            st.info("The forecast is most useful when the recent usage inputs reflect your actual meter readings.", icon=":material/lightbulb:")
        submitted = st.form_submit_button("Generate forecast", type="primary", icon=":material/bolt:")

    if submitted:
        features = pd.DataFrame([{
            "day_of_week": target_date.weekday(),
            "month": target_date.month,
            "is_weekend": int(target_date.weekday() in [5, 6]),
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_7": rolling_7,
        }])
        prediction = max(0.0, float(model.predict(features[FEATURES])[0]))
        with st.container(border=True):
            st.subheader(f"Forecast for {target_date:%d %b %Y}")
            result_col, cost_result_col = st.columns(2)
            result_col.metric("Expected consumption", fmt_kwh(prediction))
            cost_result_col.metric("Estimated cost", f"{prediction * tariff:,.2f}")
            st.success("Forecast generated successfully.", icon=":material/check_circle:")

elif view == "Model performance":
    st.header("Model performance", icon=":material/query_stats:")
    st.caption("Time-based holdout evaluation: the final 15% of observations are treated as unseen future data.")
    with st.container(horizontal=True):
        st.metric("Mean absolute error", fmt_kwh(metrics["mae"]), border=True)
        st.metric("Root mean squared error", fmt_kwh(metrics["rmse"]), border=True)
        st.metric("R² score", f"{metrics['r2']:.3f}", border=True)
    chart_col, table_col = st.columns([1.7, 1], gap="large")
    with chart_col:
        with st.container(border=True):
            st.subheader("Actual vs predicted", divider="gray")
            chart = evaluation[[TARGET, "Predicted"]].rename(columns={TARGET: "Actual"})
            st.line_chart(chart.tail(120), height=380)
    with table_col:
        with st.container(border=True):
            st.subheader("Latest validation results", divider="gray")
            display = evaluation[[TARGET, "Predicted"]].tail(12).rename(columns={TARGET: "Actual (kWh)", "Predicted": "Predicted (kWh)"})
            st.dataframe(display.round(2), width="stretch", height=380)

else:
    st.header("Data explorer", icon=":material/table_chart:")
    days = st.slider("Days to display", min_value=30, max_value=min(365, len(daily)), value=min(90, len(daily)), step=30)
    visible = daily.tail(days)[[TARGET] + FEATURES].copy()
    st.line_chart(visible[[TARGET]], y=TARGET, height=300)
    st.dataframe(visible.sort_index(ascending=False).round(3), width="stretch", height=440)
