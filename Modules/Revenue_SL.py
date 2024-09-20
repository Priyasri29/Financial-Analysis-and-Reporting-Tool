import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("Revenue Analysis Tool")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Transpose the data
    transposed_data = data.T
    transposed_data.columns = transposed_data.iloc[0]
    transposed_data = transposed_data.drop(transposed_data.index[0])
    transposed_data.index.name = 'Year'
    transposed_data.index = pd.to_numeric(transposed_data.index)
    st.write("Transposed Data:")
    st.dataframe(transposed_data)

    # Identify revenue-related columns
    revenue_columns = [
        'Total Share Capital',
        'Total Reserves and Surplus',
        'Total Shareholders\' Funds',
        'Total Current Assets',
        'FOB Value Of Goods',
        'Other Earnings'
    ]
    
    # Check if these columns exist in the dataset
    available_revenue_columns = [col for col in revenue_columns if col in transposed_data.columns]
    st.write("Available Revenue Columns:")
    st.write(available_revenue_columns)
    
    if available_revenue_columns:
        # Extract and aggregate revenue data
        revenue_data = transposed_data[available_revenue_columns].astype(float)
        
        # Optionally aggregate columns to a single 'Total Revenue' if needed
        revenue_data['Total Revenue'] = revenue_data.sum(axis=1)
        
        # Reset index to make 'Year' a column
        revenue_data = revenue_data.reset_index()
        revenue_data.rename(columns={'index': 'Year'}, inplace=True)
        
        st.write("Aggregated Revenue Data:")
        st.dataframe(revenue_data)

        # Historical Revenue Trends
        st.write("Historical Revenue Trends")
        
        # Select revenue metric for plotting
        revenue_metric = st.selectbox("Select Revenue Metric for Trends", revenue_data.columns.difference(['Year']))
        
        if revenue_metric:
            plt.figure(figsize=(10, 6))
            plt.plot(revenue_data['Year'], revenue_data[revenue_metric], marker='o', linestyle='-')
            plt.xlabel('Year')
            plt.ylabel(revenue_metric)
            plt.title(f'Historical Trends of {revenue_metric}')
            plt.grid(True)
            st.pyplot(plt)

        # Calculate year-on-year growth
        st.write("Year-on-Year Growth Analysis")
        
        if revenue_metric:
            revenue_data['Growth'] = revenue_data[revenue_metric].pct_change() * 100
            st.write(f"Year-on-Year Growth for {revenue_metric}:")
            st.dataframe(revenue_data[['Year', revenue_metric, 'Growth']])

        # Forecasting future revenue using ARIMA
        st.write("Revenue Forecasting")

        # Prepare data for ARIMA
        if revenue_metric:
            revenue_series = revenue_data.set_index('Year')[revenue_metric]
            
            # Fit ARIMA model
            model = ARIMA(revenue_series, order=(5, 1, 0))  # Adjust order if needed
            model_fit = model.fit()
            forecast_steps = 5  # Number of years to forecast
            forecast = model_fit.forecast(steps=forecast_steps)
            
            forecast_years = [revenue_series.index[-1] + i for i in range(1, forecast_steps + 1)]
            forecast_df = pd.DataFrame({'Year': forecast_years, f'Forecasted {revenue_metric}': forecast})
            
            st.write(f"Forecast for {revenue_metric}:")
            st.dataframe(forecast_df)
            
            # Plot historical data and forecast
            plt.figure(figsize=(12, 6))
            plt.plot(revenue_series.index, revenue_series, label='Historical Data', marker='o')
            plt.plot(forecast_df['Year'], forecast_df[f'Forecasted {revenue_metric}'], label='Forecasted Data', linestyle='--', marker='o', color='red')
            plt.xlabel('Year')
            plt.ylabel(revenue_metric)
            plt.title(f'Revenue Forecasting for {revenue_metric}')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
    else:
        st.write("No revenue-related columns found in the dataset.")
