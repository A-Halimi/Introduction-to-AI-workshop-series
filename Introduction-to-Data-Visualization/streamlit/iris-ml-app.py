import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px

st.write("""
# Iris Flower Prediction App
Predict the class of your flower based on the measurements!
""")

st.sidebar.header('User Input Features')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input features in a nicer table format
st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target

# Model Building
st.sidebar.header('Model Details:')

n_estimators = st.sidebar.slider('Number of trees in the forest', 10, 100, 50)
max_depth = st.sidebar.slider('Maximum depth of the tree', 1, 20, 10)
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1234)
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
class_labels_df = pd.DataFrame({
    'Label': iris.target_names
})
st.write(class_labels_df)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')


# Convert probabilities to percentages
probs_percentage = prediction_proba[0] * 100

# Create a DataFrame for plotting
df_probs = pd.DataFrame({
    'Class': iris.target_names,
    'Probability': probs_percentage
})



# Format the text on the bars to show percentages
# Create the bar chart using Plotly
fig = px.bar(df_probs, x='Class', y='Probability', text='Probability',
             title='Prediction Probability for Each Iris Class',
             labels={'Probability': 'Probability (%)'},
             color='Probability', color_continuous_scale='Viridis')

# Format the text on the bars to show percentages
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)

fig.update_layout(
    uniformtext_minsize=10, 
    uniformtext_mode='hide',
    title_font=dict(size=24, family='Courier', color='green'),
    font=dict(size=20),
    height=600,  # Adjust the height of the figure
    width=800,   # Adjust the width of the figure
    xaxis=dict(
        tickfont=dict(size=18),  # Adjust x-axis tick font size
        title_font=dict(size=20)  # Adjust x-axis title font size
    ),
    yaxis=dict(
        tickfont=dict(size=18),  # Adjust y-axis tick font size
        title_font=dict(size=20)  # Adjust y-axis title font size
    )
)

st.plotly_chart(fig)
