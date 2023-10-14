# Sample Streamlit App showcasing toggle button


import streamlit as st

def run_number_input_app():
    st.title('Number Input Example')
    
    number = st.number_input("Enter a number", min_value=0, max_value=100, step=1)
    st.write(f"You entered: {number}")
        
run_number_input_app()
