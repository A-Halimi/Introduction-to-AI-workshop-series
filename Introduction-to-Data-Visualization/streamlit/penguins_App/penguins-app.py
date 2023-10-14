import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import sklearn
import plotly.express as px

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('streamlit/penguins_App/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
#load_clf = pickle.load(open('streamlit/penguins_App/penguins_clf.pkl', 'rb'))
load_clf = load('streamlit/penguins_App/penguins_clf.joblib')

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])

# Section to display class labels and their corresponding index number
st.subheader('Class labels and their corresponding index number')
class_labels_df = pd.DataFrame({
    'Label': penguins_species
})
st.write(class_labels_df)

st.subheader('Prediction')
st.markdown(f"Predicted Penguin Species: <h2 style='text-align: center; color: yellow;'>{penguins_species[prediction][0]}</h2>", unsafe_allow_html=True)



# Display prediction probabilities using Plotly
st.subheader('Prediction Probability')
# Prepare data for Plotly visualization
df_probs = pd.DataFrame({
    'Class': penguins_species,
    'Probability': prediction_proba[0] * 100  # Convert probability to percentage
})

# Create the bar chart using Plotly
fig = px.bar(df_probs, x='Class', y='Probability', text='Probability',
             title='Prediction Probability for Each Penguin Species',
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


