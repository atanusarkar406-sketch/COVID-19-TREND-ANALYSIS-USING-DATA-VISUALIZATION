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
# LOAD DATA (OPTIMIZED)
# ---------------------------
@st.cache_data
def load_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url, usecols=[
        'date','location','iso_code',
        'total_cases','new_cases_smoothed',
        'total_deaths','new_deaths_smoothed',
        'total_vaccinations'
    ])
    df['date'] = pd.to_datetime(df['date'])
    return df

with st.spinner("Loading COVID-19 Data..."):
    df = load_data()

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.header("Filters")

countries = sorted(df['location'].dropna().unique())
selected_country = st.sidebar.selectbox("Select Country", countries)

min_date = df['date'].min()
max_date = df['date'].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# BUG FIX: Ensure the user has selected both start and end dates
if len(date_range) != 2:
    st.warning("Please select an end date.")
    st.stop()

filtered_df = df[
    (df['location'] == selected_country) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
].dropna(subset=['total_cases'])

# ---------------------------
# SAFE CHECK
# ---------------------------
if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ---------------------------
# METRICS
# ---------------------------
st.subheader(f"📊 {selected_country} Overview")

col1, col2, col3 = st.columns(3)

# Improved formatting with commas for readability
total_cases = int(filtered_df['total_cases'].iloc[-1])
total_deaths = int(filtered_df['total_deaths'].fillna(0).iloc[-1])
total_vax = int(filtered_df['total_vaccinations'].fillna(0).iloc[-1])

col1.metric("Total Cases", f"{total_cases:,}")
col2.metric("Total Deaths", f"{total_deaths:,}")
col3.metric("Total Vaccinations", f"{total_vax:,}")

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

# BUG FIX: Get the maximum total cases reported for each country, avoiding missing recent dates
latest = df.groupby(['iso_code', 'location'], as_index=False)['total_cases'].max()

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

if len(forecast_df) > 10:
    forecast_df['days'] = (forecast_df['date'] - forecast_df['date'].min()).dt.days

    X = forecast_df[['days']]
    y = forecast_df['new_cases_smoothed']

    # Note: Consider swapping LinearRegression for a more complex time-series model in the future
    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(X.max()[0], X.max()[0] + 30).reshape(-1, 1)
    
    # Ensure no negative predictions are plotted
    future_preds = np.maximum(0, model.predict(future_days)) 

    future_dates = pd.date_range(filtered_df['date'].max(), periods=30)

    fig_pred = px.line(title="Forecast vs Actual (Linear Trend)")
    fig_pred.add_scatter(x=forecast_df['date'], y=y, name="Actual")
    fig_pred.add_scatter(x=future_dates, y=future_preds, name="Prediction", line=dict(dash='dash'))

    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info("Not enough data points for a reliable prediction.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Data Source: Our World in Data | Developed for CSE-AI Project")
