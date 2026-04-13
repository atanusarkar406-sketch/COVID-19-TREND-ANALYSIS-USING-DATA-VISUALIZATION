import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="COVID-19 Advanced Dashboard", layout="wide", page_icon="🦠")
st.title("🦠 COVID-19 Trend Analysis & Forecasting")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    df = pd.read_csv(
        url,
        storage_options=headers,
        usecols=[
            'continent','location','date','iso_code','population',
            'total_cases','new_cases_smoothed',
            'total_deaths','new_deaths_smoothed',
            'total_vaccinations'
        ]
    )
    
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['continent'].notna()]   # remove aggregates
    df = df.dropna(subset=['population'])  # IMPORTANT FIX
    
    return df

with st.spinner("Fetching Live Data..."):
    df = load_data()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Dashboard Controls")

countries = sorted(df['location'].unique())
default_ix = countries.index('United States') if 'United States' in countries else 0

selected_country = st.sidebar.selectbox("Select Country", countries, index=default_ix)

min_date = df['date'].min()
max_date = df['date'].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [pd.to_datetime('2020-03-01'), max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.stop()

filtered_df = df[
    (df['location'] == selected_country) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
].sort_values('date')

if filtered_df.empty:
    st.warning("No data available")
    st.stop()

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🌍 Global Overview", "📊 Deep Dive", "🔮 Forecast"])

# ==========================================
# TAB 1: GLOBAL OVERVIEW
# ==========================================
with tab1:
    st.subheader("Global Macro Trends")

    # Global aggregation
    global_df = df.groupby('date').sum(numeric_only=True).reset_index()

    # ✅ FIX: Vaccination %
    global_df['vaccination_pct'] = (
        global_df['total_vaccinations'] / global_df['population']
    ) * 100

    fig_global = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Global Cases (7-day avg)',
            'Global Deaths (7-day avg)',
            'Total Global Vaccinations',
            'Vaccination Coverage (%)'
        ),
        vertical_spacing=0.15
    )

    # Cases
    fig_global.add_trace(
        go.Scatter(x=global_df['date'], y=global_df['new_cases_smoothed'],
                   line=dict(color='red', width=2)),
        row=1, col=1
    )

    # Deaths
    fig_global.add_trace(
        go.Scatter(x=global_df['date'], y=global_df['new_deaths_smoothed'],
                   line=dict(color='darkred', width=2)),
        row=1, col=2
    )

    # Vaccinations total
    fig_global.add_trace(
        go.Scatter(x=global_df['date'], y=global_df['total_vaccinations'],
                   line=dict(color='green', width=2)),
        row=2, col=1
    )

    # ✅ FIXED Vaccination % chart
    fig_global.add_trace(
        go.Scatter(x=global_df['date'], y=global_df['vaccination_pct'],
                   line=dict(color='blue', width=2)),
        row=2, col=2
    )

    fig_global.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig_global, use_container_width=True)

# ==========================================
# TAB 2: DEEP DIVE
# ==========================================
with tab2:
    st.subheader(f"Analysis: {selected_country}")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Cases", f"{int(filtered_df['total_cases'].max()):,}")
    c2.metric("Total Deaths", f"{int(filtered_df['total_deaths'].max()):,}")
    c3.metric("Total Vaccinations", f"{int(filtered_df['total_vaccinations'].max()):,}")

    st.markdown("---")

    fig_cases = px.line(filtered_df, x='date', y='new_cases_smoothed',
                        title="Daily Cases (7-day avg)")
    st.plotly_chart(fig_cases, use_container_width=True)

    fig_deaths = px.line(filtered_df, x='date', y='new_deaths_smoothed',
                         title="Daily Deaths (7-day avg)", color_discrete_sequence=['red'])
    st.plotly_chart(fig_deaths, use_container_width=True)

# ==========================================
# TAB 3: FORECAST
# ==========================================
with tab3:
    st.subheader(f"🔮 90-Day Forecast: {selected_country}")

    ts_df = filtered_df[['date','new_cases_smoothed']].dropna()

    if len(ts_df) > 30:
        ts_df = ts_df.set_index('date').sort_index()
        ts_df = ts_df[~ts_df.index.duplicated()]

        ts_data = ts_df['new_cases_smoothed'].asfreq('D').ffill().fillna(0)

        model = ExponentialSmoothing(
            ts_data,
            trend='add',
            seasonal=None,
            initialization_method="estimated"
        )

        fit = model.fit(optimized=True)

        forecast = fit.forecast(90).clip(lower=0)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=ts_data.index, y=ts_data,
                                     name="Historical", line=dict(color='blue')))
        fig_pred.add_trace(go.Scatter(x=forecast.index, y=forecast,
                                     name="Forecast", line=dict(color='orange', dash='dash')))

        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("Not enough data for prediction")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Data Source: Our World in Data | Developed for Basic AI Tools & Applications Project")
