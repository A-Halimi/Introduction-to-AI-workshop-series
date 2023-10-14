import streamlit as st

def run_color_picker_app():
    st.title('Color Picker Example')
    
    color = st.color_picker("Pick a color", "#00f900")
    st.write(f"You selected: {color}")
        
run_color_picker_app()
