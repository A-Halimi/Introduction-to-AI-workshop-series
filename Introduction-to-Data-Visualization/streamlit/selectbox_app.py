# Sample Streamlit App showcasing selectbox

import streamlit as st

def run_selectbox_app():
    st.title('Selectbox Widget Example')
    
    choice = st.selectbox("Pick your favorite color:", ["Red", "Green", "Blue"])
    st.write(f"You chose: {choice}")

        
run_selectbox_app()
