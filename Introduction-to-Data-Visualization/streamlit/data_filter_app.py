# Streamlit App for Dynamic Data Filtering with Sliders

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run_dynamic_filtering_app():
    st.title('Dynamic Data Filtering with Sliders')

    # Generate random data for demonstration
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(1000).cumsum(),
        'B': np.random.randn(1000).cumsum(),
        'C': np.random.randn(1000).cumsum(),
        'D': np.random.randn(1000).cumsum()
    })

    st.write(df)
    
    # Allow users to select a column
    column_to_plot = st.selectbox("1. Select a column to visualize", df.columns)

    # Display sliders to define the range for data filtering based on the selected column
    min_val, max_val = df[column_to_plot].min(), df[column_to_plot].max()
    low_bound, high_bound = st.slider(f"2. Filter {column_to_plot} within a range:", float(min_val), float(max_val), (float(min_val), float(max_val)))

    # Filter the dataframe based on the selected range
    filtered_df = df[(df[column_to_plot] >= low_bound) & (df[column_to_plot] <= high_bound)]

    # Display a histogram of the filtered data
    fig = px.histogram(filtered_df, x=column_to_plot, nbins=50, title=f"Histogram of {column_to_plot}")
    st.plotly_chart(fig)

    # Display some statistics
    st.write(f"Mean of {column_to_plot}: {filtered_df[column_to_plot].mean():.2f}")
    st.write(f"Standard Deviation of {column_to_plot}: {filtered_df[column_to_plot].std():.2f}")
    st.write(f"Minimum {column_to_plot}: {filtered_df[column_to_plot].min():.2f}")
    st.write(f"Maximum {column_to_plot}: {filtered_df[column_to_plot].max():.2f}")

# To run this app, save the code above to a file, e.g., "dynamic_filtering_app.py"
# Then, in the terminal, run: streamlit run dynamic_filtering_app.py



run_dynamic_filtering_app()