import streamlit as st
import pandas as pd
import altair as alt

# Set up the Streamlit page configuration
st.set_page_config(page_title="Financial Dashboard", layout="wide")

# Title of the dashboard
st.title("Company Financial Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Attempt to read the CSV file
    try:
        # Read the CSV file, assuming the first column as the index (attributes)
        df = pd.read_csv(uploaded_file, index_col=0)
        
        # Transpose the DataFrame so that attributes become columns and years become rows
        df = df.T

        # Clean up column names by removing whitespace and converting to lowercase
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Convert all data to numeric, forcing non-numeric values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Display the cleaned data
        st.write("### Cleaned Data")
        st.dataframe(df)

        # Generate additional metrics if necessary
        df['tax_rate'] = df['tax_expense'] / df['profit_before_tax'] * 100
        df['net_profit_margin'] = df['profit_after_tax'] / df['total_revenue'] * 100

        # Create and display different types of charts

        # Cluster Chart for Total Revenue and Profit Before Tax
        cluster_chart = alt.Chart(df.reset_index()).mark_point(size=100).encode(
            x=alt.X('total_revenue:Q', title="Total Revenue"),
            y=alt.Y('profit_before_tax:Q', title="Profit Before Tax"),
            color=alt.Color('index:N', title='Financial Year'),
            tooltip=['index', 'total_revenue', 'profit_before_tax']
        ).properties(
            title="Cluster Chart: Total Revenue vs. Profit Before Tax"
        )

        # Line Chart for Profit Before Tax
        profit_chart = alt.Chart(df.reset_index()).mark_line(point=True).encode(
            x=alt.X('index:O', title="Financial Year"),
            y=alt.Y('profit_before_tax:Q', title="Profit Before Tax", axis=alt.Axis(format=",.0f")),
            color=alt.Color('index:N', title='Financial Year'),
            tooltip=['index', 'profit_before_tax']
        ).properties(
            title="Profit Before Tax Over Years"
        )


        # Line Chart for Tax Rate
        tax_rate_chart = alt.Chart(df.reset_index()).mark_line(point=True).encode(
            x=alt.X('index:O', title="Financial Year"),
            y=alt.Y('tax_rate:Q', title="Tax Rate (%)", axis=alt.Axis(format=",.2f")),
            color=alt.Color('index:N', title='Financial Year'),
            tooltip=['index', 'tax_rate']
        ).properties(
            title="Tax Rate Over Years"
        )

        # Bar Chart for Net Profit Margin
        net_profit_margin_chart = alt.Chart(df.reset_index()).mark_bar().encode(
            x=alt.X('index:O', title="Financial Year"),
            y=alt.Y('net_profit_margin:Q', title="Net Profit Margin (%)", axis=alt.Axis(format=",.2f")),
            color=alt.Color('index:N', title='Financial Year'),
            tooltip=['index', 'net_profit_margin']
        ).properties(
            title="Net Profit Margin Over Years"
        )

    

        # Bubble Chart for Tax Expense vs. Profit After Tax
        bubble_chart = alt.Chart(df.reset_index()).mark_circle(size=100).encode(
            x=alt.X('tax_expense:Q', title="Tax Expense"),
            y=alt.Y('profit_after_tax:Q', title="Profit After Tax"),
            color=alt.Color('index:N', title='Financial Year'),
            tooltip=['index', 'tax_expense', 'profit_after_tax']
        ).properties(
            title="Bubble Chart: Tax Expense vs. Profit After Tax"
        )

        # Display the charts in a two-column layout
        col1, col2 = st.columns(2)
        col1.altair_chart(cluster_chart, use_container_width=True)
        col2.altair_chart(profit_chart, use_container_width=True)

        

        col3, col4 = st.columns(2)
        col3.altair_chart(tax_rate_chart, use_container_width=True)
        col4.altair_chart(net_profit_margin_chart, use_container_width=True)

        
        st.altair_chart(bubble_chart, use_container_width=True)

        # Display Key Metrics
        st.write("### Key Metrics")
        st.metric(label="Latest Revenue", value=f"{df['total_revenue'].iloc[-1]:,}")
        st.metric(label="Latest Profit Before Tax", value=f"{df['profit_before_tax'].iloc[-1]:,}")
        st.metric(label="Latest Profit After Tax", value=f"{df['profit_after_tax'].iloc[-1]:,}")
        st.metric(label="Retained Earnings", value=f"{df['balance_at_the_end_of_the_year'].iloc[-1]:,}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.warning("Please upload a CSV file to see the dashboard.")