import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit app title
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
