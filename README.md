# AQI-Prediction-Model
An interactive data analytics and forecasting dashboard that predicts the Air Quality Index (AQI) for multiple Indian cities using machine learning time series models such as ARIMA (SARIMAX) and Prophet.
The project provides actionable insights through visualizations, trend analysis, and pollution forecasting.

## Project Overview
This project analyzes historical AQI data to identify pollution patterns and forecast future AQI levels.
It enables users to:
Explore AQI trends across different cities.
Visualize seasonal and monthly variations.
Detect peak pollution periods.
Compare forecasting models (ARIMA vs. Prophet).
Derive policy and public health recommendations.

## Tech Stack
Languages: Python
Frameworks: Streamlit
Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, prophet, scikit-learn
Visualization: Seaborn & Matplotlib

## Dataset
Based on publicly available AQI datasets for Indian cities on Kaggle
Cleaned and preprocessed by:
Handling missing values
Removing outliers using the IQR method
Aggregating data at daily, monthly, and quarterly levels
Dividing the dataset into five CSV files (one per region or city group)

## Features
Interactive Streamlit dashboard with multiple tabs:
Data Overview: Summary stats & basic cleaning steps
Trends & Distribution: AQI line plots and boxplots
Peak Pollution Analysis: Top 10% highest AQI periods
Forecasting: ARIMA and Prophet model predictions for upcoming weeks
Policy Insights: Actionable recommendations based on results

Real-time visualization of AQI forecasts
User-selectable city and aggregation frequency
Model performance metrics (MAE, RMSE)

## Model Workflow
Data Loading & Cleaning – Load CSVs, handle null values, remove outliers.
EDA (Exploratory Data Analysis) – Identify city-wise AQI patterns.
Feature Aggregation – Aggregate by day/month/quarter for trends.
Forecasting – Train ARIMA (SARIMAX) and Prophet models.
Evaluation – Compare model performance.
Visualization & Deployment – Display interactive results via Streamlit.

## Results
Forecasts AQI for next 26 weeks for selected cities.
Provides visual insights on pollution trends and high-risk periods.
Useful for policymakers, researchers, and the general public.

## Future Improvements
Add LSTM/Deep Learning models for time series prediction.
Integrate real-time AQI API data for live updates.
Add more cities and environmental parameters.
