# Sample Streamlit App showcasing file uploader

import streamlit as st

def run_file_uploader_app():
    st.title('File Uploader Widget Example')
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        st.write(f"Uploaded file: {uploaded_file.name}")
        
run_file_uploader_app()