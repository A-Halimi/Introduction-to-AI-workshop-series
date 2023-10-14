import streamlit as st

# Sample movie data
movies = {
    "Inception": {
        "Release Year": 2010,
        "Genre": "Sci-Fi",
        "Director": "Christopher Nolan",
        "Rating": 8.8
    },
    "The Shawshank Redemption": {
        "Release Year": 1994,
        "Genre": "Drama",
        "Director": "Frank Darabont",
        "Rating": 9.3
    },
    "The Dark Knight": {
        "Release Year": 2008,
        "Genre": "Action",
        "Director": "Christopher Nolan",
        "Rating": 9.0
    }
}

def run_enhanced_layout_app():
    
    st.title('Movie Database Dashboard')
    
    # Create two columns: Search and Details
    col_search, col_details = st.columns(2)
    
    with col_search:
        st.header("🔍 Search for a Movie")
        selected_movie = st.selectbox("Choose a movie", list(movies.keys()))
        
        # Button to fetch details
        if st.button("Fetch Details"):
            with col_details:
                st.header(f"🎬 {selected_movie} Details")
                for key, value in movies[selected_movie].items():
                    st.write(f"**{key}:** {value}")


    # Styling and additional information
    st.markdown("""
        ---
        ### 📜 **About this Dashboard**
        
        - **Data Source**: This dashboard uses a simulated dataset of popular movies.
        - **Purpose**: Demonstrating Streamlit's capability to create interactive web apps with ease.
        - **Tip**: Use the left column to select a movie and click 'Fetch Details' to view its attributes in the right column.
        
        For more information about Streamlit, visit the [official website](https://www.streamlit.io/).
        
        _Made with ❤️ by [Your Name]_
    """)
    st.markdown("---")

    # New Tips & Tricks section with an expander
    with st.expander("Tips & Tricks with Streamlit 🚀"):
        st.markdown("""
            1. **Widgets**: Streamlit offers a variety of widgets like buttons, sliders, and text input that make your apps interactive.
            2. **Caching**: Use `@st.cache` to cache your data and functions, speeding up your apps significantly.
            3. **Layouts**: Organize your app's layout using columns and expanders for a better user experience.
            4. **Share**: Deploy and share your Streamlit apps with the world using Streamlit sharing.
            5. **Community**: Engage with the Streamlit community for discussions, questions, and sharing resources.
            
            Explore the [Streamlit documentation](https://docs.streamlit.io/) for more details and best practices.
        """)

    # Theming Section
    with st.expander("Theming with Streamlit 🎨"):
        st.markdown("""
            Streamlit provides robust theming capabilities to customize the look and feel of your app. Here's how you can use theming:
            
            1. **Configuration**: Adjust the `config.toml` file in your Streamlit configuration directory.
            2. **Colors**: Customize the primary color, background color, and text color of your app.
            3. **Fonts**: Choose different fonts for the body and headings of your app.
            4. **Dark Mode**: Easily switch between light and dark themes.
            
            Check the [official Streamlit documentation](https://docs.streamlit.io/library/api-reference/widgets/st.theme) for more details on theming and customization options.
        """)

# To run this enhanced layout app, save the code to a separate file, e.g., "enhanced_layout_app.py"
# Execute the app using the command: streamlit run enhanced_layout_app.py


        
run_enhanced_layout_app()
