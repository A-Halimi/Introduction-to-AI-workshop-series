import streamlit as st

def run_date_input_app():
    st.title('Date Input Widget Example')
    
    selected_date = st.date_input("Pick a date")
    st.write(f"Selected Date: {selected_date}")

run_date_input_app()