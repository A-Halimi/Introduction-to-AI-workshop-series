import streamlit as st

def run_time_input_app():
    st.title('Time Input Widget Example')
    
    selected_time = st.time_input("Set an alarm for")
    st.write(f"Alarm set for: {selected_time}")

run_time_input_app()