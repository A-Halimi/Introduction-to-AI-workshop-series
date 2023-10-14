# Sample Streamlit App showcasing toggle button

import streamlit as st

def run_button_app():
    st.title('Button Example')
    
    if st.button("Click me!"):
        st.write("Button was clicked!")
        
run_button_app()
