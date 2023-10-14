# Improved Streamlit App for Interactive Plot Customization

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def run_improved_plot_customization_app():
    st.title('Enhanced Interactive Plot Customization')
    
    # Sample data
    df = px.data.iris()
    gapminder_df = px.data.gapminder()
    
    # Data description
    st.write("### Iris Dataset")
    st.write("This dataset comprises 3 species of iris plants with 4 features: sepal length, sepal width, petal length, and petal width.")
    st.dataframe(df.head())

    # Plot customization options
    plot_type = st.selectbox("Choose a plot type", ["scatter", "line", "bar", "histogram", "box", "3D scatter", "subplot", "choropleth map"])
    
    # For choropleth map, we don't need x and y axis selection
    if plot_type != "choropleth map":
        x_axis = st.selectbox("Choose x-axis variable", df.columns[:-1])  # Exclude 'species' from x-axis
        y_axis = st.selectbox("Choose y-axis variable", df.columns[:-1])  # Exclude 'species' from y-axis

    # Create the plot
    if plot_type == "scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, color="species", title=f"Scatter Plot of {y_axis} vs. {x_axis}")
    elif plot_type == "line":
        fig = px.line(df.sort_values(by=[x_axis]), x=x_axis, y=y_axis, color="species", title=f"Line Plot of {y_axis} vs. {x_axis}")
    elif plot_type == "bar":
        fig = px.bar(df, x=x_axis, y=y_axis, color="species", title=f"Bar Plot of {y_axis} by {x_axis}")
    elif plot_type == "histogram":
        fig = px.histogram(df, x=x_axis, color="species", title=f"Histogram of {x_axis}")
    elif plot_type == "box":
        fig = px.box(df, x="species", y=y_axis, title=f"Box Plot of {y_axis} by Species")
    elif plot_type == "3D scatter":
        fig = px.scatter_3d(df, x="sepal_width", y="sepal_length", z="petal_length", color="species", title="3D Scatter Plot of Iris Dataset")
    elif plot_type == "subplot":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis], mode='markers', name='scatter'))
        fig.add_trace(go.Histogram(x=df[x_axis], name='histogram'))
        fig.update_layout(title=f"Subplot of {y_axis} vs. {x_axis}")
    else:  # choropleth map
        fig = px.choropleth(gapminder_df, locations="iso_alpha", color="lifeExp", hover_name="country", title="Life Expectancy Choropleth Map", animation_frame="year")
        
    st.plotly_chart(fig)




# Save the enhanced code to "improved_plot_customization_app.py"
# Then, in the terminal, run: streamlit run improved_plot_customization_app.py



run_improved_plot_customization_app()