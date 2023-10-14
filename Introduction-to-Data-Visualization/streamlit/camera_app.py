import streamlit as st

def run_camera_app():
    st.title('Camera Input Example')
    picture = st.camera_input("Take a picture")
    
    if picture:
        st.image(picture)
        
run_camera_app()
