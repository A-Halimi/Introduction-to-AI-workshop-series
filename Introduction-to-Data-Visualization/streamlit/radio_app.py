# Sample Streamlit App showcasing radio buttons

import streamlit as st

def run_radio_app():
    st.title('Radio Buttons Widget Example')
    
    option = st.radio("Choose a fruit:", ["Apple", "Banana", "Cherry"])
    st.write(f"You selected: {option}")
        
run_radio_app()
