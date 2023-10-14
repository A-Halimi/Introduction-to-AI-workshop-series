# Required Libraries
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.graph_objects as go

def run_ml_model_app():

    st.title('Building a Machine Learning Model with Streamlit')
    
    # Introduction
    st.markdown("""
    ### Introduction

    In this section, we will explore:
    - Using `scikit-learn` for machine learning within a Streamlit app.
    - How to build, train, and evaluate a simple machine learning model.
    - How to display the model's results and allow interactive predictions.
    """)
    
    # Load the Iris dataset
    st.markdown("### Iris Dataset")
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    st.write("First five rows of the Iris dataset:")
    st.dataframe(df.head())
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['species'], test_size=0.2, random_state=42)
    
    # Building and training a Decision Tree classifier
    st.markdown("### Training a Decision Tree Classifier")
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    st.write("Model trained!")
    
    # Displaying model results
    st.markdown("### Model Evaluation")

    # Predictions
    y_pred = classifier.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the model: {accuracy:.2f}")
    
    # Confusion Matrix
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=iris.target_names,
                        y=iris.target_names,
                        colorscale='Viridis',
                        hoverongaps = False))
    fig.update_layout(title="Confusion Matrix")

    st.plotly_chart(fig)
    
    # User predictions using input widgets
    st.markdown("### Make Your Own Predictions!")
    
    # Input widgets for each feature
    sl = st.slider("Sepal Length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()))
    sw = st.slider("Sepal Width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()))
    pl = st.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()))
    pw = st.slider("Petal Width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()))
    
    # Button to make predictions
    if st.button("Predict"):
        user_pred = classifier.predict([[sl, sw, pl, pw]])
        st.write(f"Predicted Species: {iris.target_names[user_pred[0]]}")


run_ml_model_app()