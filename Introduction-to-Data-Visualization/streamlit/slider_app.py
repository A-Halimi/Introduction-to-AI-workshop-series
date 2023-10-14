import streamlit as st

def run_slider_app():
    st.title('Streamlit Slider Widget Example')
    
    # Create a slider widget
    number = st.slider("Choose a number", 0, 100)
    
    # Display the selected value
    st.write(f"You selected the number: {number}")


        
run_slider_app()
