import streamlit as st

def run_text_area_app():
    st.title('Text Area Widget Example')
    
    user_text = st.text_area("Share your thoughts:", "Type here...")
    st.write(f"You wrote: {user_text}")

run_text_area_app()