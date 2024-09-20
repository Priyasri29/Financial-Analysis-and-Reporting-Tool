import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Cost Analysis with Linear Regression")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 2: Load and display data
    data = pd.read_csv(uploaded_file)
    st.write("Original Data:")
    st.write(data)

    # Step 3: Transpose data
    data_transposed = data.set_index(data.columns[0]).T
    data_transposed.reset_index(inplace=True)
    data_transposed.rename(columns={'index': 'Year'}, inplace=True)
    st.write("Transposed Data:")
    st.write(data_transposed)

    # Step 4: Select cost-related column for analysis
    columns = list(data_transposed.columns)
    columns.remove('Year')
    cost_column = st.selectbox("Select the Cost Metric for Analysis", columns)

    # Step 5: Prepare data for linear regression
    X = data_transposed[['Year']].astype(int)  # Independent variable (Year)
    y = data_transposed[cost_column].astype(float)  # Dependent variable (Cost)

    # Step 6: Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Step 7: Predict future costs
    future_years = np.array([X['Year'].max() + i for i in range(1, 6)]).reshape(-1, 1)
    future_costs = model.predict(future_years)

    # Step 8: Display results
    st.write("Future Cost Predictions:")
    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        f'Predicted {cost_column}': future_costs
    })
    st.write(forecast_df)

    # Step 9: Plot the historical and predicted data
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label='Historical Costs', marker='o')
    plt.plot(future_years, future_costs, label='Predicted Costs', linestyle='--', marker='o', color='red')
    plt.xlabel('Year')
    plt.ylabel(cost_column)
    plt.title(f'Cost Analysis and Prediction for {cost_column}')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
