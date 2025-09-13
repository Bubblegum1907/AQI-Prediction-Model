import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX  # Ensure SARIMAX is imported
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AQI Dashboard", layout="wide")

# --- City Data Files ---
cities = {
    'Delhi': 'Delhi final data.xlsx',
    'Mumbai': 'Mumbai final data.xlsx',
    'Kolkata': 'Kolkata final data.xlsx',
    'Chennai': 'Chennai final data.xlsx',
    'Bangalore': 'Bangalore final data.xlsx'
}

st.title("🌍 AQI Dashboard")
st.markdown("Interactive analysis of AQI across Indian cities")

# --- Load Data ---
@st.cache_data
def load_data():
    dfs = []
    for city, file in cities.items():
        df_city = pd.read_excel(file)
        if 'Date' not in df_city.columns:
            df_city.rename(columns={df_city.columns[0]: 'Date'}, inplace=True)
        df_city['Date'] = pd.to_datetime(df_city['Date'], errors='coerce')
        for col in df_city.select_dtypes(include=np.number).columns:
            df_city[col].fillna(df_city[col].mean(), inplace=True)
        df_city['City'] = city
        dfs.append(df_city)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

df = load_data()

# --- Outlier Removal (IQR method) ---
df_filtered = pd.DataFrame()
for city in df['City'].unique():
    df_city = df[df['City'] == city].copy()
    Q1 = df_city['AQI'].quantile(0.25)
    Q3 = df_city['AQI'].quantile(0.75)
    IQR = Q3 - Q1
    df_city_filtered = df_city[(df_city['AQI'] >= Q1 - 1.5*IQR) & (df_city['AQI'] <= Q3 + 1.5*IQR)]
    df_filtered = pd.concat([df_filtered, df_city_filtered], ignore_index=True)

df = df_filtered

# --- Sidebar Filters ---
st.sidebar.header("Filters")
selected_cities = st.sidebar.multiselect("Select Cities", df["City"].unique(), default=df["City"].unique())
agg_freq = st.sidebar.radio("Aggregation", ["Daily", "Monthly", "Quarterly", "Half-Yearly"])

filtered_df = df[df["City"].isin(selected_cities)].copy()

# --- Aggregation ---
if agg_freq == "Monthly":
    filtered_df = filtered_df.groupby([pd.Grouper(key="Date", freq="M"), "City"])["AQI"].mean().reset_index()
elif agg_freq == "Quarterly":
    filtered_df = filtered_df.groupby([pd.Grouper(key="Date", freq="Q"), "City"])["AQI"].mean().reset_index()
elif agg_freq == "Half-Yearly":
    filtered_df = filtered_df.groupby([pd.Grouper(key="Date", freq="2Q"), "City"])["AQI"].mean().reset_index()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Overview", "Trends & Distribution", 
    "Peak Pollution Analysis", "Forecasting", "Policy Recommendations"
])

# --- Tab 1: Data Overview ---
with tab1:
    st.subheader("Filtered Data")
    st.write("Number of rows after filtering:", filtered_df.shape[0])
    st.dataframe(filtered_df.head(20))

# --- Tab 2: Trends & Distribution ---
with tab2:
    st.subheader("AQI Trends Over Time")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(x="Date", y="AQI", hue="City", data=filtered_df, marker="o", ax=ax)
    ax.set_title("AQI Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Air Quality Index (AQI)")
    st.pyplot(fig)

    st.subheader("AQI Distribution by City")
    fig2, ax2 = plt.subplots(figsize=(12,6))
    sns.boxplot(x="City", y="AQI", data=df_filtered[df_filtered["City"].isin(selected_cities)], ax=ax2)
    ax2.set_title("AQI Distribution Across Cities")
    ax2.set_xlabel("City")
    ax2.set_ylabel("AQI")
    st.pyplot(fig2)

# --- Tab 3: Peak Pollution Analysis ---
with tab3:
    st.subheader("Peak Pollution Weeks (Top 10%)")
    for city in selected_cities:
        df_city = filtered_df[filtered_df["City"] == city].copy()
        peak_threshold = df_city["AQI"].quantile(0.90)
        peak_weeks = df_city[df_city["AQI"] >= peak_threshold]
        st.write(f"**{city}** - Peak AQI Threshold: {peak_threshold:.1f}, Weeks Above Threshold: {len(peak_weeks)}")
        st.dataframe(peak_weeks)

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_city['Date'], df_city['AQI'], label="AQI", color='blue')
        ax.scatter(peak_weeks['Date'], peak_weeks['AQI'], color='red', label="Top 10% AQI Weeks")
        ax.set_title(f"Weekly AQI with Peak Pollution - {city}")
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.legend()
        st.pyplot(fig)

with tab4:
    st.subheader("Forecasting for Selected Cities")
    
    # Three tabs: ARIMA, Prophet, Comparison
    arima_tab, prophet_tab, comparison_tab = st.tabs(["ARIMA Forecast", "Prophet Forecast", "Model Comparison"])
    
    # --- Store metrics for comparison ---
    metrics_dict = {}

    # --- ARIMA Forecast ---
    with arima_tab:
        for city in selected_cities:
            st.markdown(f"### {city} - ARIMA Forecast")

            # --- Prepare weekly series with actual AQI values ---
            df_city = filtered_df[filtered_df["City"] == city].set_index('Date').asfreq('D')
            y = df_city['AQI'].astype(float).fillna(method='ffill').fillna(method='bfill')

            try:
                # Fit SARIMAX
                y_weekly = y.resample('W').mean()

                # Fit SARIMAX with weekly seasonality
                model = SARIMAX(
                    y_weekly,
                    order=(2,1,2),
                    seasonal_order=(1,1,1,52),  # add seasonal differencing for better pattern
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)

                # Forecast next 26 weeks
                forecast = results.get_forecast(steps=26)
                forecast_index = pd.date_range(y_weekly.index.max() + pd.Timedelta(weeks=1), periods=26, freq='W')
                forecast_df = pd.DataFrame({
                    'Date': forecast_index,
                    'Forecasted_AQI': forecast.predicted_mean,
                    'Lower_CI': forecast.conf_int().iloc[:,0],
                    'Upper_CI': forecast.conf_int().iloc[:,1]
                })

                # Plot last 1 year + forecast
                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(y_weekly[-52:].index, y_weekly[-52:].values, label="Recent AQI", color='blue')
                ax.plot(forecast_df['Date'], forecast_df['Forecasted_AQI'], label="ARIMA Forecast", color='red')
                ax.fill_between(forecast_df['Date'], forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='pink', alpha=0.3)
                ax.set_title(f"ARIMA Forecast - {city}")
                ax.set_xlabel("Date")
                ax.set_ylabel("AQI")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.write(f"ARIMA forecast could not be computed for {city}: {e}")

    # --- Prophet Forecast ---
    with prophet_tab:
        for city in selected_cities:
            st.markdown(f"### {city} - Prophet Forecast")
            df_prophet = df[df["City"] == city][['Date','AQI']].rename(columns={'Date':'ds','AQI':'y'})
            df_prophet['y'] = df_prophet['y'].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.03)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=26, freq='W')
            forecast = model.predict(future)

            # In-sample metrics
            forecast_in_sample = forecast[forecast['ds'] <= df_prophet['ds'].max()]
            mae = mean_absolute_error(df_prophet['y'], forecast_in_sample['yhat'])
            rmse = np.sqrt(mean_squared_error(df_prophet['y'], forecast_in_sample['yhat']))

            # Save metrics
            if city in metrics_dict:
                metrics_dict[city].update({'Prophet_MAE': mae, 'Prophet_RMSE': rmse})
            else:
                metrics_dict[city] = {'Prophet_MAE': mae, 'Prophet_RMSE': rmse}

            # Plot
            df_plot = df_prophet.set_index('ds').resample('W').mean().reset_index()
            df_recent = df_plot[-52:]
            forecast_plot = forecast[forecast['ds'] > df_prophet['ds'].max()]

            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df_recent['ds'], df_recent['y'], label="Past AQI (last 1 year)", color='blue', marker='o')
            ax.plot(forecast_plot['ds'], forecast_plot['yhat'], label="Prophet Forecast", color='green')
            ax.fill_between(forecast_plot['ds'], forecast_plot['yhat_lower'], forecast_plot['yhat_upper'], color='lightgreen', alpha=0.3)
            ax.set_title(f"Prophet Forecast - {city}")
            ax.set_xlabel("Date")
            ax.set_ylabel("AQI")
            ax.legend()
            st.pyplot(fig)

    # --- Model Comparison ---
    with comparison_tab:
        st.subheader("ARIMA vs Prophet In-Sample Metrics")
        comparison_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        st.dataframe(comparison_df.style.format("{:.2f}"))

# --- Tab 5: Policy Recommendations ---
with tab5:
    st.subheader("Policy-Level Interventions and Public Health Recommendations")
    st.markdown("""
    ### Policy-Level Interventions:
    - **Traffic Management:** Implement odd-even or car-free days during predicted peak pollution weeks.
    - **Industrial Regulations:** Temporarily limit emissions from factories and construction activities in high-AQI periods.
    - **Urban Planning:** Increase green cover in urban areas to reduce pollution accumulation.
    - **Monitoring & Alerts:** Strengthen real-time air quality monitoring and early warning systems for upcoming high-AQI periods.
    - **Public Transport Enhancement:** Promote public transport usage to reduce vehicular emissions.

    ### Public Health Recommendations:
    - **Stay Indoors:** Limit outdoor activities during peak AQI weeks or days.
    - **Air Purifiers & Masks:** Encourage use of air purifiers indoors and N95 masks outdoors during high-pollution periods.
    - **Sensitive Groups:** Advise children, elderly, and individuals with respiratory conditions to take extra precautions.
    - **Health Awareness:** Conduct campaigns to inform citizens about the health risks of air pollution and preventive measures.
    """)

