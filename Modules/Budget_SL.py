import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Streamlit app title
st.title("Budgeting and Forecasting Tool")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Transpose the data
    transposed_data = data.T
    transposed_data.columns = transposed_data.iloc[0]  # Set first row as header
    transposed_data = transposed_data.drop(transposed_data.index[0])  # Drop the first row
    transposed_data.index.name = 'Year'  # Set index name to 'Year'
    transposed_data.index = pd.to_numeric(transposed_data.index)  # Ensure index is numeric

    # Display the transposed data
    st.write("Transposed Data:")
    st.dataframe(transposed_data)
    
    # User input to select a column for analysis
    column_name = st.selectbox("Select the financial metric for forecasting", transposed_data.columns)

    # Function to calculate budget
    def calculate_budget(data, column_name, forecast_years=5):
        data = data[column_name].dropna().astype(float)
        growth_rate = data.pct_change().mean()  # Calculate average growth rate
        last_value = data.iloc[-1]
        future_years = [data.index[-1] + i for i in range(1, forecast_years + 1)]
        future_values = [last_value * (1 + growth_rate) ** i for i in range(1, forecast_years + 1)]
        budget_df = pd.DataFrame({'Year': future_years, f'Projected {column_name}': future_values})
        return budget_df

    # Function to forecast using ARIMA
    def forecast_with_arima(data, column_name, forecast_years=5):
        data = data[column_name].dropna().astype(float)
        model = ARIMA(data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_years)
        forecast_years = [data.index[-1] + i for i in range(1, forecast_years + 1)]
        forecast_df = pd.DataFrame({'Year': forecast_years, f'Forecasted {column_name}': forecast})
        return forecast_df

    # Calculate budget and forecast
    budget_df = calculate_budget(transposed_data, column_name)
    forecast_df = forecast_with_arima(transposed_data, column_name)

    # Display results
    st.write("Budgeting Results:")
    st.dataframe(budget_df)
    
    st.write("Forecasting Results:")
    st.dataframe(forecast_df)

    # Plot the results
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(transposed_data.index, transposed_data[column_name].values, label='Historical Data', marker='o')
    ax.plot(budget_df['Year'], budget_df[f'Projected {column_name}'], label='Budgeted Data', linestyle='--', marker='o')
    ax.plot(forecast_df['Year'], forecast_df[f'Forecasted {column_name}'], label='Forecasted Data', linestyle=':', marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel(column_name + ' (Rs. Cr.)')
    ax.set_title(f'Budgeting and Forecasting of {column_name}')
    ax.legend()
    ax.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
