# Sample Streamlit App showcasing sidebar

import streamlit as st

def run_sidebar_app():
    st.title('Sidebar Example')
    
    # Adding widgets to the sidebar
    with st.sidebar:
        st.header("Sidebar Controls")
        number = st.number_input("Enter a number in the sidebar", min_value=0, max_value=100, step=1)
        color = st.color_picker("Pick a color in the sidebar", "#00f900")
    
    st.write(f"Number from sidebar: {number}")
    st.write(f"Color from sidebar: {color}")
        
run_sidebar_app()
