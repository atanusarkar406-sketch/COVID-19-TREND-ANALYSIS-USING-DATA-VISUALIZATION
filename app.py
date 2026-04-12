import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

st.title("🦠 COVID-19 Trend Analysis Dashboard")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/owid-covid-data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.header("Filters")

countries = df['location'].unique()
selected_country = st.sidebar.selectbox("Select Country", countries)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['date'].min(), df['date'].max()]
)

filtered_df = df[
    (df['location'] == selected_country) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
]

# ---------------------------
# METRICS
# ---------------------------
st.subheader(f"📊 {selected_country} Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Cases", int(filtered_df['total_cases'].fillna(0).iloc[-1]))
col2.metric("Total Deaths", int(filtered_df['total_deaths'].fillna(0).iloc[-1]))
col3.metric("Total Vaccinations", int(filtered_df['total_vaccinations'].fillna(0).iloc[-1]))

# ---------------------------
# CASES TREND
# ---------------------------
fig_cases = px.line(
    filtered_df,
    x='date',
    y='new_cases_smoothed',
    title="Daily Cases (7-day avg)"
)
st.plotly_chart(fig_cases, use_container_width=True)

# ---------------------------
# DEATHS TREND
# ---------------------------
fig_deaths = px.line(
    filtered_df,
    x='date',
    y='new_deaths_smoothed',
    title="Daily Deaths (7-day avg)",
    color_discrete_sequence=['red']
)
st.plotly_chart(fig_deaths, use_container_width=True)

# ---------------------------
# VACCINATION TREND
# ---------------------------
fig_vacc = px.line(
    filtered_df,
    x='date',
    y='total_vaccinations',
    title="Vaccination Progress",
    color_discrete_sequence=['green']
)
st.plotly_chart(fig_vacc, use_container_width=True)

# ---------------------------
# GLOBAL MAP
# ---------------------------
st.subheader("🌍 Global COVID-19 Spread")

latest = df[df['date'] == df['date'].max()]

fig_map = px.choropleth(
    latest,
    locations="iso_code",
    color="total_cases",
    hover_name="location",
    title="Global COVID-19 Cases",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------
# MULTI-COUNTRY COMPARISON
# ---------------------------
st.subheader("📈 Compare Countries")

selected_multi = st.multiselect(
    "Select Multiple Countries",
    countries,
    default=[selected_country]
)

multi_df = df[df['location'].isin(selected_multi)]

fig_compare = px.line(
    multi_df,
    x='date',
    y='new_cases_smoothed',
    color='location',
    title="Cases Comparison"
)
st.plotly_chart(fig_compare, use_container_width=True)

# ---------------------------
# MACHINE LEARNING FORECAST
# ---------------------------
st.subheader("🔮 30-Day Prediction")

forecast_df = filtered_df[['date', 'new_cases_smoothed']].dropna()

forecast_df['days'] = (forecast_df['date'] - forecast_df['date'].min()).dt.days

X = forecast_df[['days']]
y = forecast_df['new_cases_smoothed']

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(X.max()[0], X.max()[0] + 30).reshape(-1, 1)
future_preds = model.predict(future_days)

future_dates = pd.date_range(filtered_df['date'].max(), periods=30)

fig_pred = px.line(title="Forecast vs Actual")

fig_pred.add_scatter(x=forecast_df['date'], y=y, name="Actual")
fig_pred.add_scatter(x=future_dates, y=future_preds, name="Prediction")

st.plotly_chart(fig_pred, use_container_width=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("Developed for COVID-19 Trend Analysis Project | CSE-AI")
