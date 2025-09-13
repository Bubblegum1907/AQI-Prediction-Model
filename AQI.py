import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

cities = {
    'Delhi': 'Delhi final data.xlsx',
    'Mumbai': 'Mumbai final data.xlsx',
    'Kolkata': 'Kolkata final data.xlsx',
    'Chennai': 'Chennai final data.xlsx',
    'Bangalore': 'Bangalore final data.xlsx'
}

dataframes = {city: pd.read_excel(file) for city, file in cities.items()}

for city, df_city in dataframes.items():
    if 'Unnamed: 0' in df_city.columns:
        df_city.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    if 'Date' in df_city.columns:
        df_city['Date'] = pd.to_datetime(df_city['Date'])

for city, df_city in dataframes.items():
    print(f"{city} - Min AQI: {np.min(df_city['AQI'].values)}, Max AQI: {np.max(df_city['AQI'].values)}")

for city, df_city in dataframes.items():
    for col in df_city.select_dtypes(include='number').columns:
        df_city[col].fillna(df_city[col].mean(), inplace=True)

for city, df_city in dataframes.items():
    print(f"\nMissing values in {city}:")
    print(df_city.isnull().sum())

df = pd.concat(
    [df_city.assign(City=city) for city, df_city in dataframes.items()],
    ignore_index=True
)

df_biyearly = df.groupby(['City', pd.Grouper(key='Date', freq='2Q')])['AQI'].mean().reset_index()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))
sns.boxplot(x='City', y='AQI', data=df)
plt.title('AQI Across Indian Cities')
plt.xlabel('City', fontsize=12)
plt.ylabel('Air Quality Index (AQI)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.lineplot(x='Date', y='AQI', hue='City', data=df_biyearly)
plt.title("AQI Trend Over Time")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Air Quality Index (AQI)", fontsize=12)
plt.legend(title="City")
plt.tight_layout()
plt.show()