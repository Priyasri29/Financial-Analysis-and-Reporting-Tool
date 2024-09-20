import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns  
# Streamlit app title
st.title("Financial Analysis Tool")


# Sidebar menu for navigation
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Select an option", [
    "Budget Forecasting Tool", 
    "Revenue Analysis Tool", 
    "Cost Analysis Tool",
    "Scenario Analysis Tool"
])

if app_mode == "Budget Forecasting Tool":
    st.title("Budget Forecasting Tool")

    # File uploader in Streamlit with unique key
    uploaded_file = st.file_uploader("Upload your CSV file for Budget Forecasting", type=["csv"], key="budget_forecasting_uploader")

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
            'Miscellaneous Expenses',
            'Tax',
            'Other Income',
            'Raw Materials',
            'Power & Fuel Cost',
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

elif app_mode == "Revenue Analysis Tool":
    st.title("Revenue Analysis Tool")

    # File uploader in Streamlit with unique key
    uploaded_file = st.file_uploader("Upload your CSV file for Revenue Analysis", type=["csv"], key="revenue_analysis_uploader")

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

elif app_mode == "Cost Analysis Tool":
    st.title("Cost Analysis Tool")

    # File uploader in Streamlit with unique key
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Original Data:")
    st.dataframe(data)

    # Feature Selection
    selected_features = st.multiselect("Select features for clustering:", data.columns.tolist())
    if selected_features:
        selected_data = data[selected_features]

        # Handling Categorical and Numerical Data
        categorical_cols = selected_data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = selected_data.select_dtypes(include=['number']).columns.tolist()

        if categorical_cols:
            selected_data = pd.get_dummies(selected_data, columns=categorical_cols)

        if numerical_cols:
            scaler = StandardScaler()
            selected_data[numerical_cols] = scaler.fit_transform(selected_data[numerical_cols])

        st.write("Preprocessed Data:")
        st.dataframe(selected_data)

        # Number of Clusters
        num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)

        # Agglomerative Clustering
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
        cluster_labels = clustering_model.fit_predict(selected_data)

        # Add cluster labels to original data
        data['Cluster'] = cluster_labels

        # Display the clustered data in table form
        st.write("Clustered Data with Labels:")
        st.dataframe(data)

        # Silhouette Score
        silhouette_avg = silhouette_score(selected_data, cluster_labels)
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")

        # PCA for Dimensionality Reduction
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(selected_data)

        # Plotting the Clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=cluster_labels, palette="viridis")
        plt.title("Clusters Visualized with PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Cluster")
        st.pyplot(plt)

elif app_mode == "Scenario Analysis Tool":
    st.title("Scenario Analysis Tool")

    # File uploader in Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Transpose the data
    transposed_data = data.T
    transposed_data.columns = transposed_data.iloc[0]  # Set the first row as header
    transposed_data = transposed_data.drop(transposed_data.index[0])  # Drop the old header row
    transposed_data.reset_index(inplace=True)
    transposed_data.rename(columns={'index': 'Year'}, inplace=True)
    
    # Display the transposed data
    st.write("Transposed Data:")
    st.dataframe(transposed_data)
    
    # Define revenue, cost, and profit variables
    revenue_vars = ['Sales Turnover', 'Net Sales', 'Total Income', 'Other Income']
    cost_vars = ['Raw Materials', 'Power & Fuel Cost', 'Employee Cost', 'Selling and Admin Expenses', 'Miscellaneous Expenses']
    profit_vars = ['Operating Profit', 'PBDIT', 'Profit Before Tax', 'Reported Net Profit', 'Earnings Per Share (EPS)']
    
    # Create a dropdown to select scenario
    scenario = st.selectbox("Select Scenario", ["Best-Case", "Worst-Case", "Most-Likely"])
    
    # Define scenario assumptions
    if scenario == "Best-Case":
        revenue_increase = 0.1  # 10% increase
        cost_decrease = 0.05  # 5% decrease
    elif scenario == "Worst-Case":
        revenue_increase = -0.1  # 10% decrease
        cost_decrease = 0.1  # 10% increase
    else:  # Most-Likely
        revenue_increase = 0.03  # 3% increase
        cost_decrease = 0.02  # 2% decrease
    
    # Calculate new values based on the selected scenario
    scenario_data = transposed_data.copy()
    
    # Apply scenario changes to revenue
    for col in revenue_vars:
        if col in scenario_data.columns:
            scenario_data[col] = scenario_data[col].astype(float) * (1 + revenue_increase)
    
    # Apply scenario changes to costs
    for col in cost_vars:
        if col in scenario_data.columns:
            scenario_data[col] = scenario_data[col].astype(float) * (1 - cost_decrease)
    
    # Recalculate profit metrics
    if 'Operating Profit' in scenario_data.columns:
        scenario_data['Operating Profit'] = (
            scenario_data['Total Income'].astype(float) - scenario_data[['Raw Materials', 'Power & Fuel Cost', 'Employee Cost', 'Selling and Admin Expenses', 'Miscellaneous Expenses']].astype(float).sum(axis=1)
        )
    if 'Reported Net Profit' in scenario_data.columns:
        scenario_data['Reported Net Profit'] = (
            scenario_data['Operating Profit'].astype(float) - scenario_data[['Interest', 'Depreciation', 'Tax']].astype(float).sum(axis=1)
        )
    
    # Display the scenario data
    st.write(f"Data for {scenario} Scenario:")
    st.dataframe(scenario_data)
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in revenue_vars + profit_vars:
        if col in scenario_data.columns:
            ax.plot(scenario_data['Year'], scenario_data[col], label=col)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title(f'{scenario} Scenario Analysis')
    ax.legend()
    ax.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
