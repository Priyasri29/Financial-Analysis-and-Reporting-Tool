import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("KPI Monitoring Tool")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    
    # Transpose the data
    transposed_data = data.T
    transposed_data.columns = transposed_data.iloc[0]
    transposed_data = transposed_data.drop(transposed_data.index[0])
    transposed_data.index.name = 'Year'
    transposed_data.index = pd.to_numeric(transposed_data.index)
    st.write("Transposed Data:")
    st.dataframe(transposed_data)

    # Define KPI calculations
    def calculate_liquidity_ratios(df):
        current_ratio = df['Total Current Assets'] / df['Total Current Liabilities']
        quick_ratio = (df['Total Current Assets'] - df['Inventories']) / df['Total Current Liabilities']
        return current_ratio, quick_ratio

    def calculate_profitability_ratios(df):
        roa = (df['Net Income'] / df['Total Assets']) * 100
        roe = (df['Net Income'] / df['Total Shareholders\' Funds']) * 100
        return roa, roe

    def calculate_leverage_ratios(df):
        debt_to_equity = df['Long Term Borrowings'] / df['Total Shareholders\' Funds']
        return debt_to_equity

    def calculate_efficiency_ratios(df):
        asset_turnover = df['Total Revenue'] / df['Total Assets']
        return asset_turnover

    def calculate_revenue_metrics(df):
        total_revenue = df['Total Revenue']
        revenue_growth = total_revenue.pct_change() * 100
        return total_revenue, revenue_growth

    # Select KPI for analysis
    kpi_choice = st.selectbox("Select KPI for Monitoring", [
        'Current Ratio',
        'Quick Ratio',
        'ROA',
        'ROE',
        'Debt to Equity Ratio',
        'Asset Turnover Ratio',
        'Total Revenue',
        'Revenue Growth'
    ])

    if kpi_choice:
        if kpi_choice in ['Current Ratio', 'Quick Ratio']:
            current_ratio, quick_ratio = calculate_liquidity_ratios(transposed_data)
            kpi_data = pd.DataFrame({
                'Year': transposed_data.index,
                'Current Ratio': current_ratio,
                'Quick Ratio': quick_ratio
            }).set_index('Year')
            st.write(f"{kpi_choice} Over Time:")
            st.dataframe(kpi_data)

        elif kpi_choice in ['ROA', 'ROE']:
            roa, roe = calculate_profitability_ratios(transposed_data)
            kpi_data = pd.DataFrame({
                'Year': transposed_data.index,
                'ROA': roa,
                'ROE': roe
            }).set_index('Year')
            st.write(f"{kpi_choice} Over Time:")
            st.dataframe(kpi_data)

        elif kpi_choice == 'Debt to Equity Ratio':
            debt_to_equity = calculate_leverage_ratios(transposed_data)
            kpi_data = pd.DataFrame({
                'Year': transposed_data.index,
                'Debt to Equity Ratio': debt_to_equity
            }).set_index('Year')
            st.write(f"Debt to Equity Ratio Over Time:")
            st.dataframe(kpi_data)

        elif kpi_choice == 'Asset Turnover Ratio':
            asset_turnover = calculate_efficiency_ratios(transposed_data)
            kpi_data = pd.DataFrame({
                'Year': transposed_data.index,
                'Asset Turnover Ratio': asset_turnover
            }).set_index('Year')
            st.write(f"Asset Turnover Ratio Over Time:")
            st.dataframe(kpi_data)

        elif kpi_choice == 'Total Revenue':
            total_revenue, _ = calculate_revenue_metrics(transposed_data)
            kpi_data = pd.DataFrame({
                'Year': transposed_data.index,
                'Total Revenue': total_revenue
            }).set_index('Year')
            st.write(f"Total Revenue Over Time:")
            st.dataframe(kpi_data)

        elif kpi_choice == 'Revenue Growth':
            _, revenue_growth = calculate_revenue_metrics(transposed_data)
            kpi_data = pd.DataFrame({
                'Year': transposed_data.index,
                'Revenue Growth (%)': revenue_growth
            }).set_index('Year')
            st.write(f"Revenue Growth Over Time:")
            st.dataframe(kpi_data)

        # Plot the KPI
        plt.figure(figsize=(12, 6))
        plt.plot(kpi_data.index, kpi_data.iloc[:, 0], marker='o', linestyle='-')
        plt.xlabel('Year')
        plt.ylabel(kpi_choice)
        plt.title(f'{kpi_choice} Over Time')
        plt.grid(True)
        st.pyplot(plt)

    # Benchmark Comparison
    st.write("Benchmark Comparison (Example)")

    benchmark = st.number_input("Enter Benchmark Value:", value=0.0)

    if kpi_choice and kpi_choice in kpi_data.columns:
        kpi_data['Benchmark'] = benchmark
        st.write("KPI vs Benchmark:")
        st.dataframe(kpi_data)

        plt.figure(figsize=(12, 6))
        plt.plot(kpi_data.index, kpi_data[kpi_choice], label='KPI', marker='o')
        plt.axhline(y=benchmark, color='r', linestyle='--', label='Benchmark')
        plt.xlabel('Year')
        plt.ylabel(kpi_choice)
        plt.title(f'{kpi_choice} vs Benchmark')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
