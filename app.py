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
# LOAD DATA (Live from OWID)
# ---------------------------
@st.cache_data
def load_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url, usecols=[
        'continent', 'location', 'date', 'iso_code', 'population',
        'total_cases', 'new_cases_smoothed',
        'total_deaths', 'new_deaths_smoothed',
        'total_vaccinations', 'people_fully_vaccinated_per_hundred'
    ])
    df['date'] = pd.to_datetime(df['date'])
    # Filter out aggregate regions (like 'World', 'Asia', 'High income')
    df = df[df['continent'].notna()]
    return df

with st.spinner("Fetching Live Global COVID-19 Data..."):
    df = load_data()

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.header("Dashboard Controls")

countries = sorted(df['location'].unique())
# Default to a major country to ensure good data on load
default_ix = countries.index('United States') if 'United States' in countries else 0
selected_country = st.sidebar.selectbox("Select Target Country", countries, index=default_ix)

min_date = df['date'].min()
max_date = df['date'].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [pd.to_datetime('2020-03-01'), max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.warning("Please select a complete date range.")
    st.stop()

# Filter data for the selected country
filtered_df = df[
    (df['location'] == selected_country) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
].sort_values('date')

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ---------------------------
# TAB LAYOUT
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🌍 Global Overview", "📊 Deep Dive & Heatmaps", "🔮 ML Forecast"])

# ==========================================
# TAB 1: GLOBAL OVERVIEW
# ==========================================
with tab1:
    st.subheader("Global Macro Trends")
    
    # Calculate global aggregates for the 2x2 subplot
    global_df = df.groupby('date').sum(numeric_only=True).reset_index()
    
    fig_global = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Global Cases (7-day avg)', 'Global Deaths (7-day avg)',
                        'Total Global Vaccinations', 'Vaccination Coverage (%)'),
        vertical_spacing=0.15
    )

    fig_global.add_trace(go.Scatter(x=global_df['date'], y=global_df['new_cases_smoothed'],
                                    line=dict(color='red', width=2), name='Cases'), row=1, col=1)
    fig_global.add_trace(go.Scatter(x=global_df['date'], y=global_df['new_deaths_smoothed'],
                                    line=dict(color='darkred', width=2), name='Deaths'), row=1, col=2)
    fig_global.add_trace(go.Scatter(x=global_df['date'], y=global_df['total_vaccinations'],
                                    line=dict(color='green', width=2), name='Vax'), row=2, col=1)
    
    # Scatter for Vax vs Cases
    latest_data = df[df['date'] == df['date'].max()].dropna(subset=['people_fully_vaccinated_per_hundred', 'total_cases'])
    fig_global.add_trace(go.Scatter(x=latest_data['people_fully_vaccinated_per_hundred'], 
                                    y=latest_data['total_cases'],
                                    mode='markers', marker=dict(size=8, color='blue', opacity=0.5),
                                    text=latest_data['location'], name='Vax vs Cases'), row=2, col=2)

    fig_global.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig_global, use_container_width=True)

# ==========================================
# TAB 2: DEEP DIVE & HEATMAPS
# ==========================================
with tab2:
    st.subheader(f"Analyzing: {selected_country}")
    
    # Top metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cases", f"{int(filtered_df['total_cases'].max()):,}")
    c2.metric("Total Deaths", f"{int(filtered_df['total_deaths'].max()):,}")
    c3.metric("Total Vaccinations", f"{int(filtered_df['total_vaccinations'].max()):,}")

    # Heatmap logic
    st.markdown("---")
    st.subheader("Regional Intensity Heatmap")
    
    # Get top 15 countries by total cases to make a clean heatmap
    top_countries = df.groupby('location')['total_cases'].max().nlargest(15).index
    heatmap_df = df[df['location'].isin(top_countries) & (df['date'] >= pd.to_datetime(date_range[0]))]
    
    # Pivot for heatmap
    pivot_cases = heatmap_df.pivot(index='location', columns='date', values='new_cases_smoothed').fillna(0)
    
    fig_heat = px.imshow(
        pivot_cases, 
        aspect="auto", 
        color_continuous_scale='Reds',
        title="Daily Cases Heatmap (Top 15 Most Affected Countries)"
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 3: MACHINE LEARNING FORECAST
# ==========================================
with tab3:
    st.subheader(f"🔮 90-Day Trend Forecast for {selected_country}")
    st.caption("Powered by statsmodels Exponential Smoothing (Holt-Winters)")

    # Prepare strictly continuous daily data for statsmodels to prevent ValueWarnings
    ts_df = filtered_df[['date', 'new_cases_smoothed']].dropna()
    
    if len(ts_df) > 30:
        # Set index, remove duplicates, and force Daily ('D') frequency
        ts_df = ts_df.set_index('date').sort_index()
        ts_df = ts_df[~ts_df.index.duplicated(keep='first')]
        
        # This .asfreq('D') is the magic fix for your statsmodels warnings
        ts_data = ts_df['new_cases_smoothed'].asfreq('D').ffill().fillna(0)

        # Build and fit the model
        model = ExponentialSmoothing(
            ts_data, 
            trend='add', 
            seasonal=None, # Dropped seasonal to prevent overfitting on noisy data
            initialization_method="estimated"
        )
        
        # optimized=True helps prevent the ConvergenceWarning
        fitted_model = model.fit(optimized=True)
        forecast_steps = 90
        future_preds = fitted_model.forecast(forecast_steps)
        future_preds = future_preds.clip(lower=0) # Prevent negative predictions

        # Plotly chart
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, 
                                      name='Historical Cases', line=dict(color='blue')))
        fig_pred.add_trace(go.Scatter(x=future_preds.index, y=future_preds.values, 
                                      name='AI Forecast (90 Days)', line=dict(color='orange', dash='dash')))
        
        fig_pred.update_layout(height=500, template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig_pred, use_container_width=True)
        
    else:
        st.warning("Not enough continuous historical data to generate a reliable forecast. Please select a wider date range.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Data Source: Our World in Data | Developed for CSE-AI Project")
