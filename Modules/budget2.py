import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("Budget Forecasting Tool")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    
    # Transpose and clean the data
    transposed_data = data.T
    transposed_data.columns = transposed_data.iloc[0]
    transposed_data = transposed_data.drop(transposed_data.index[0])
    transposed_data.index.name = 'Year'
    transposed_data.index = pd.to_numeric(transposed_data.index)
    st.write("Transposed Data:")
    st.dataframe(transposed_data)

    # List of columns for forecasting
    forecastable_columns = [
        'Sales Turnover',
        'Net Sales',
        'Total Income',
        'Operating Profit',
        'Reported Net Profit',
        'Earning Per Share (Rs)'
    ]

    # User input for the metric to forecast
    metric = st.selectbox("Select the financial metric for forecasting", forecastable_columns)

    # Function to forecast using ARIMA
    def forecast_with_arima(df, column_name, forecast_years=5):
        data = df[column_name].dropna().astype(float)
        model = ARIMA(data, order=(5, 1, 0))  # Adjust order as necessary
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_years)
        forecast_years = [data.index[-1] + i for i in range(1, forecast_years + 1)]
        forecast_df = pd.DataFrame({'Year': forecast_years, f'Forecasted {column_name}': forecast})
        return forecast_df

    # Perform forecasting
    forecast_df = forecast_with_arima(transposed_data, metric)

    # Display results
    st.write(f"Forecasting Results for {metric}:")
    st.dataframe(forecast_df)

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(transposed_data.index, transposed_data[metric].astype(float), label='Historical Data', marker='o')
    plt.plot(forecast_df['Year'], forecast_df[f'Forecasted {metric}'], label='Forecasted Data', linestyle='--', marker='o')
    plt.xlabel('Year')
    plt.ylabel(metric)
    plt.title(f'Forecasting of {metric}')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
