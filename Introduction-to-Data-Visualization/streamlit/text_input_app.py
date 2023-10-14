import streamlit as st

def run_text_input_app():
    st.title('Text Input Widget Example')
    
    user_input = st.text_input("Enter your name:", value="John Doe")
    st.write(f"Hello, {user_input}!")

run_text_input_app()