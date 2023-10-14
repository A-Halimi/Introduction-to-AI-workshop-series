import streamlit as st
import time
import random
import pandas as pd
import plotly.express as px

# Initialize a data frame to hold our simulated website visits data
initial_data = {
    "Timestamp": [pd.Timestamp.now()],
    "Visits": [random.randint(1, 10)]
}
visits_df = pd.DataFrame(initial_data)

def run_real_time_update_app():
    st.title('Real-time Website Visits Tracker')
    
    global visits_df

    st.write("This dashboard simulates the tracking of website visits in real-time. Click the 'Update Data' button to fetch the latest visit counts.")
    
    # Button to trigger data update
    if st.button("Update Data"):
        with st.spinner("Fetching new visit data..."):
            time.sleep(2)  # Simulate a delay for data fetching
            
            # Simulate the new data
            new_data = {
                "Timestamp": [pd.Timestamp.now()],
                "Visits": [random.randint(1, 10)]
            }
            # Convert dictionary to DataFrame and concatenate
            visits_df = pd.concat([visits_df, pd.DataFrame(new_data)], ignore_index=True)
        st.success("Data updated!")

    # Always display the updated plot
    st.plotly_chart(px.line(visits_df, x='Timestamp', y='Visits', title='Website Visits Over Time'))

    # Display the last few rows of the DataFrame to verify updates
    st.write("Last few records of the data:")
    st.dataframe(visits_df.tail())



        
run_real_time_update_app()
