# Sample Streamlit App showcasing multiselect

import streamlit as st

def run_multiselect_app():
    st.title('Multiselect Widget Example')
    
    selections = st.multiselect("What are your favorite hobbies?", ["Reading", "Traveling", "Cooking", "Gaming"])
    st.write(f"You selected: {', '.join(selections)}")
        
run_multiselect_app()
