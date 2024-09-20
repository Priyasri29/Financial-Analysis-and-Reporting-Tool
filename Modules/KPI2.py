import pandas as pd
import streamlit as st

# Function to perform KPI analysis for a specific year
def calculate_revenue_kpis(data):
    kpis = {
        "Total Sales Revenue": data['sales'].sum(),
        "Gross Sales": data['gross sales'].sum(),
        "Average Sale Price": data['sale price'].mean(),
        "Revenue Growth Rate (%)": data['sales'].pct_change().mean() * 100,
    }
    return kpis

def calculate_profitability_kpis(data):
    kpis = {
        "Total Profit": data['profit'].sum(),
        "Profit Margin (%)": (data['profit'].sum() / data['sales'].sum()) * 100,
        "Gross Profit Margin (%)": ((data['gross sales'].sum() - data['cogs'].sum()) / data['gross sales'].sum()) * 100,
    }
    return kpis

def calculate_cost_kpis(data):
    kpis = {
        "Total COGS": data['cogs'].sum(),
        "Average Manufacturing Price": data['manufacturing price'].mean(),
        "COGS as a Percentage of Sales (%)": (data['cogs'].sum() / data['sales'].sum()) * 100,
    }
    return kpis

def calculate_sales_kpis(data):
    kpis = {
        "Total Units Sold": data['units sold'].sum(),
        "Average Sales Per Unit": data['sale price'].mean(),
        "Average Discount": data['discounts'].mean(),
        "Discount Impact (%)": (data['discounts'].sum() / data['gross sales'].sum()) * 100,
    }
    return kpis

# Streamlit app
def main():
    st.title("KPI Analysis: 2024 vs 2023")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Strip whitespace and ensure lowercase column names
        data.columns = data.columns.str.strip().str.lower()

        st.write("Data Preview:")
        st.write(data.head())
        
        # Ensure correct data types for numerical operations
        try:
            data['year'] = pd.to_numeric(data['year'], errors='coerce')
            data['units sold'] = pd.to_numeric(data['units sold'], errors='coerce')
            data['manufacturing price'] = pd.to_numeric(data['manufacturing price'], errors='coerce')
            data['sale price'] = pd.to_numeric(data['sale price'], errors='coerce')
            data['gross sales'] = pd.to_numeric(data['gross sales'], errors='coerce')
            data['discounts'] = pd.to_numeric(data['discounts'], errors='coerce')
            data['sales'] = pd.to_numeric(data['sales'], errors='coerce')
            data['cogs'] = pd.to_numeric(data['cogs'], errors='coerce')
            data['profit'] = pd.to_numeric(data['profit'], errors='coerce')
        except KeyError as e:
            st.error(f"Column not found: {e}")
            return

        # Filter data by year
        data_2023 = data[data['year'] == 2023]
        data_2024 = data[data['year'] == 2024]

        if data_2023.empty or data_2024.empty:
            st.error("Data for 2023 or 2024 is missing.")
            return
        
        # Calculate KPIs for each year
        revenue_kpis_2023 = calculate_revenue_kpis(data_2023)
        revenue_kpis_2024 = calculate_revenue_kpis(data_2024)
        profitability_kpis_2023 = calculate_profitability_kpis(data_2023)
        profitability_kpis_2024 = calculate_profitability_kpis(data_2024)
        cost_kpis_2023 = calculate_cost_kpis(data_2023)
        cost_kpis_2024 = calculate_cost_kpis(data_2024)
        sales_kpis_2023 = calculate_sales_kpis(data_2023)
        sales_kpis_2024 = calculate_sales_kpis(data_2024)

        # Combine KPIs into DataFrames for comparison
        revenue_comparison = pd.DataFrame({
            'KPI': list(revenue_kpis_2023.keys()),
            '2023': list(revenue_kpis_2023.values()),
            '2024': list(revenue_kpis_2024.values())
        })

        profitability_comparison = pd.DataFrame({
            'KPI': list(profitability_kpis_2023.keys()),
            '2023': list(profitability_kpis_2023.values()),
            '2024': list(profitability_kpis_2024.values())
        })

        cost_comparison = pd.DataFrame({
            'KPI': list(cost_kpis_2023.keys()),
            '2023': list(cost_kpis_2023.values()),
            '2024': list(cost_kpis_2024.values())
        })

        sales_comparison = pd.DataFrame({
            'KPI': list(sales_kpis_2023.keys()),
            '2023': list(sales_kpis_2023.values()),
            '2024': list(sales_kpis_2024.values())
        })

        # Display the KPI comparisons
        st.write("## Revenue KPI Comparison: 2024 vs 2023")
        st.table(revenue_comparison)
        
        st.write("## Profitability KPI Comparison: 2024 vs 2023")
        st.table(profitability_comparison)
        
        st.write("## Cost KPI Comparison: 2024 vs 2023")
        st.table(cost_comparison)
        
        st.write("## Sales KPI Comparison: 2024 vs 2023")
        st.table(sales_comparison)

if __name__ == "__main__":
    main()
