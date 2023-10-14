# Sample Streamlit App showcasing checkbox

import streamlit as st

def run_checkbox_app():
    st.title('Checkbox Widget Example')
    
    checked = st.checkbox("Click me!")
    if checked:
        st.write("Checkbox is checked!")
        
run_checkbox_app()