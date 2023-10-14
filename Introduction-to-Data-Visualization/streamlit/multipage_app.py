# Enhanced Streamlit App for Multi-page Structure

import streamlit as st
import plotly.express as px
import pandas as pd

# Sample data
df = px.data.iris()

def data_view():
    st.header("Data View")
    st.write("### Iris Dataset")
    st.write("This dataset comprises 3 species of iris plants with 4 features: sepal length, sepal width, petal length, and petal width.")
    st.dataframe(df.head())
    
def visualization():
    st.header("Visualization")
    st.write("Explore different visualizations of the Iris dataset.")
    

    
    # Plot customization options
    plot_type = st.selectbox("Choose a plot type", ["scatter", "histogram", "box"])
    x_axis = st.selectbox("Choose x-axis variable", df.columns[:-1])  # Exclude 'species' from x-axis
    y_axis = st.selectbox("Choose y-axis variable", df.columns[:-1])  # Exclude 'species' from y-axis
    
    if plot_type == "scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, color="species", title=f"Scatter Plot of {y_axis} vs. {x_axis}")
    elif plot_type == "histogram":
        fig = px.histogram(df, x=x_axis, color="species", title=f"{x_axis} Distribution by Species")
    else:
        fig = px.box(df, x="species", y=x_axis, title=f"{x_axis} Box Plot by Species")
    
    st.plotly_chart(fig)

def analysis():
    st.header("Analysis")
    st.write("Basic statistical analysis of the Iris dataset.")
    
    # Select the variable for statistical analysis within this function
    feature = st.selectbox("Choose a feature for analysis", df.columns[:-1])

    # Display statistics for the selected feature
    st.write(f"Statistics for {feature}:")
    st.write(f"Mean: {df[feature].mean():.2f}")
    st.write(f"Median: {df[feature].median():.2f}")
    st.write(f"Standard Deviation: {df[feature].std():.2f}")
    st.write(f"Variance: {df[feature].var():.2f}")

def run_enhanced_multipage_app():
    st.title('Enhanced Multi-page Application')
    
    # Using radio buttons as a form of menu to select the page view
    page = st.radio("Choose a page", ["Home", "Data View", "Visualization", "Analysis"])

    if page == "Home":
        st.write("Welcome to the enhanced multi-page application!")
        st.write("Navigate through the pages to explore the Iris dataset.")
    elif page == "Data View":
        data_view()
    elif page == "Visualization":
        visualization()
    else:
        analysis()

# Save the enhanced code to "enhanced_multipage_app.py"
# Then, in the terminal, run: streamlit run enhanced_multipage_app.py




run_enhanced_multipage_app()