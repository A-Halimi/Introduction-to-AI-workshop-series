# Streamlit Widget Showcase Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import datetime

def run_widget_showcase_dashboard():
    st.title('Streamlit Widget Showcase Dashboard')
    st.write("This dashboard showcases the variety of widgets available in Streamlit.")

    # Text Input
    user_input = st.text_input("1. Text Input - Enter your name:", value="John Doe")
    st.write(f"Hello, {user_input}!")

    # Text Area
    user_text = st.text_area("2. Text Area - Share your thoughts:", "Type here...")
    st.write(f"You wrote: {user_text}")

    # Number Input
    number = st.number_input("3. Number Input - Enter a number", min_value=0, max_value=100, step=1)
    st.write(f"You entered: {number}")

    # Slider
    range_values = st.slider("4. Slider - Select a range of values", 0, 100, (25, 75))
    st.write(f"Selected range: {range_values}")

    # Selectbox
    option = st.selectbox("5. Selectbox - Choose an option:", ["Option A", "Option B", "Option C"])
    st.write(f"You selected: {option}")

    # Multiselect
    selections = st.multiselect("6. Multiselect - Choose multiple options:", ["Option 1", "Option 2", "Option 3", "Option 4"])
    st.write(f"You selected: {', '.join(selections)}")

    # Radio Buttons
    choice = st.radio("7. Radio Buttons - Choose a fruit:", ["Apple", "Banana", "Cherry"])
    st.write(f"You chose: {choice}")

    # Checkbox (Toggle)
    toggle = st.checkbox("8. Checkbox (Toggle) - Toggle me!")
    if toggle:
        st.write("Toggle is ON!")
    else:
        st.write("Toggle is OFF!")

    # Date Input
    date = st.date_input("9. Date Input - Select a date:")
    st.write(f"Selected Date: {date}")

    # Time Input
    t = st.time_input("10. Time Input - Set a time:", datetime.time(8, 45))
    st.write(f"Selected Time: {t}")

    # File Uploader
    uploaded_file = st.file_uploader("11. File Uploader - Choose a CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

    # Color Picker
    color = st.color_picker("12. Color Picker - Choose a color", "#00f900")
    st.write(f"Selected Color: {color}")

    # Buttons
    if st.button("13. Button - Click me!"):
        st.write("Button was clicked!")

    # Sidebar
    with st.sidebar:
        st.header("Sidebar Widgets")
        sidebar_text = st.text_input("Text Input in Sidebar", "Type here...")
        sidebar_number = st.number_input("Number Input in Sidebar", 0, 100, 50)
        sidebar_selectbox = st.selectbox("Selectbox in Sidebar", ["A", "B", "C"])

# To run the dashboard, save this code to a file, e.g., "widget_dashboard.py"
# Then, in the terminal, execute: streamlit run widget_dashboard.py

        
run_widget_showcase_dashboard()
