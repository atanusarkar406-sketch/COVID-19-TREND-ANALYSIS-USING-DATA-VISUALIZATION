import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sample COVID-19 data (in practice, you'd load from Our World in Data or similar)
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
n_days = len(dates)

# Generate realistic COVID-19 data for multiple countries
countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'France',
             'Germany', 'Italy', 'Spain', 'Japan', 'Australia']

data = []
for country in countries:
    # Simulate cases with waves
    base_cases = np.random.poisson(1000, n_days)
    wave1 = np.maximum(0, 50000 * np.exp(-((dates - pd.Timestamp('2020-04-01')).days.values/30)**2))
    wave2 = np.maximum(0, 40000 * np.exp(-((dates - pd.Timestamp('2021-01-01')).days.values/30)**2))
    wave3 = np.maximum(0, 30000 * np.exp(-((dates - pd.Timestamp('2021-08-01')).days.values/30)**2))

    cases = base_cases + wave1 + wave2 + wave3
    cases = np.maximum(0, cases.cumsum())

    # Deaths (2-5% fatality rate)
    # Fixed: Convert numpy array to Series to use .diff().fillna()
    daily_cases = pd.Series(cases).diff().fillna(0).astype(int)
    deaths = np.random.binomial(daily_cases, np.random.uniform(0.02, 0.05, n_days))
    deaths = np.maximum(0, deaths.cumsum())

    # Vaccinations
    vacc_start = pd.Timestamp('2020-12-01')
    vaccinations = np.zeros(n_days)
    vacc_idx = (dates >= vacc_start).nonzero()[0]
    vaccinations[vacc_idx] = np.cumsum(np.random.poisson(50000, len(vacc_idx)))

    data.extend([{
        'date': date,
        'country': country,
        'cases': cases[i],
        'deaths': deaths[i],
        'vaccinations': vaccinations[i],
        'cases_7d_avg': pd.Series(cases).diff().rolling(7).mean()[i],
        'deaths_7d_avg': pd.Series(deaths).diff().rolling(7).mean()[i]
    } for i, date in enumerate(dates)])

df = pd.DataFrame(data)

# 1. GLOBAL OVERVIEW DASHBOARD
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Daily New Cases (7-day avg)', 'Daily New Deaths (7-day avg)',
                   'Total Vaccinations', 'Cases per Million'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Global aggregates
global_cases = df.groupby('date')['cases'].sum().diff().fillna(0)
global_deaths = df.groupby('date')['deaths'].sum().diff().fillna(0)
global_vacc = df.groupby('date')['vaccinations'].sum()

fig.add_trace(
    go.Scatter(x=df['date'].unique(), y=global_cases.rolling(7).mean(),
               name='Global Cases', line=dict(color='red', width=3)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df['date'].unique(), y=global_deaths.rolling(7).mean(),
               name='Global Deaths', line=dict(color='darkred', width=3)),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=df['date'].unique(), y=global_vacc,
               name='Global Vaccinations', line=dict(color='green', width=3)),
    row=2, col=1
)

# Cases per million (simplified)
df['population'] = df['country'].map({
    'United States': 331e6, 'India': 1380e6, 'Brazil': 212e6, 'United Kingdom': 67e6,
    'France': 67e6, 'Germany': 83e6, 'Italy': 60e6, 'Spain': 47e6, 'Japan': 126e6, 'Australia': 25e6
})
df['cases_per_million'] = df['cases'] / df['population'] * 1e6
fig.add_trace(
    go.Scatter(x=df['date'].unique(), y=df.groupby('date')['cases_per_million'].mean(),
               name='Cases/Million', line=dict(color='orange', width=3)),
    row=2, col=2
)

fig.update_layout(height=800, title_text="🌍 COVID-19 Global Trends Dashboard",
                  showlegend=False, template='plotly_white')
fig.show()

# 2. INTERACTIVE COUNTRY COMPARISON
fig = px.line(df, x='date', y='cases_7d_avg', color='country',
              title='Daily New Cases by Country (7-day Moving Average)',
              labels={'cases_7d_avg': 'Daily New Cases (7-day avg)'})
fig.update_traces(line=dict(width=2.5))
fig.update_layout(height=600, template='plotly_white')
fig.show()

# 3. HEATMAP OF CASES BY COUNTRY AND TIME
pivot_cases = df.pivot(index='country', columns='date', values='cases_7d_avg')
fig = px.imshow(pivot_cases.T, aspect="auto", color_continuous_scale='Reds',
                title='COVID-19 Cases Heatmap (Countries x Time)')
fig.update_layout(height=700)
fig.show()

# 4. VACCINATION VS CASES CORRELATION
fig = px.scatter(df[df['date'] > '2021-01-01'], x='vaccinations', y='cases_7d_avg',
                size='population', color='country', hover_name='country',
                log_x=True, log_y=True,
                title='Vaccination Impact: Total Vaccinations vs Recent Cases',
                labels={'vaccinations': 'Total Vaccinations', 'cases_7d_avg': 'Recent Daily Cases'})
fig.show()

# 5. TOP COUNTRIES RANKING
latest_date = df['date'].max()
latest_data = df[df['date'] == latest_date]

fig = make_subplots(rows=1, cols=3,
                    subplot_titles=('Total Cases', 'Total Deaths', 'Vaccinations'),
                    specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]])

fig.add_trace(go.Bar(x=latest_data['country'], y=latest_data['cases'],
                    marker_color='red', name='Cases'), row=1, col=1)
fig.add_trace(go.Bar(x=latest_data['country'], y=latest_data['deaths'],
                    marker_color='darkred', name='Deaths'), row=1, col=2)
fig.add_trace(go.Bar(x=latest_data['country'], y=latest_data['vaccinations'],
                    marker_color='green', name='Vaccinations'), row=1, col=3)

fig.update_layout(height=500, showlegend=False, template='plotly_white',
                  title_text=f"🏆 Country Rankings - {latest_date.strftime('%Y-%m-%d')}")
fig.show()

# 6. TIME SERIES FORECAST (Simple Exponential Smoothing)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

us_data = df[df['country'] == 'United States'].set_index('date')['cases_7d_avg'].fillna(0)
model = ExponentialSmoothing(us_data, trend='add', seasonal='add', seasonal_periods=30)
fit = model.fit()
forecast = fit.forecast(90)

fig = go.Figure()
fig.add_trace(go.Scatter(x=us_data.index, y=us_data, name='Historical', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecast 90 days',
                        line=dict(color='orange', dash='dash')))
fig.update_layout(title='United States: Cases Forecast (90 days)',
                  template='plotly_white', height=500)
fig.show()

# 7. KEY INSIGHTS SUMMARY
print("\n" + "="*60)
print("📈 KEY COVID-19 TRENDS INSIGHTS")
print("="*60)

latest_global = df[df['date'] == df['date'].max()].sum(numeric_only=True)
print(f"🌍 Global Totals (Latest):")
print(f"   Total Cases: {latest_global['cases']:,}")
print(f"   Total Deaths: {latest_global['deaths']:,}")
print(f"   Total Vaccinations: {latest_global['vaccinations']:,}")

# Peak analysis
peak_info = df.loc[df['cases_7d_avg'].idxmax()]
print(f"\n⛰️  Global Peak (Simulated): {peak_info['cases_7d_avg']:,.0f} cases/day in {peak_info['country']} on {peak_info['date'].date()}")

# Vaccination coverage
print(f"\n💉 Vaccination Progress: {latest_global['vaccinations']/sum(df['population'].unique()):.1%} of global pop")