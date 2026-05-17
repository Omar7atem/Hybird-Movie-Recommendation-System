import sys
import os
import streamlit as st
import pandas as pd

# Add src to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import load_and_clean_data
from hybrid_model import hybrid_recommendations

# Set page config
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #E50914;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #B20710;
        color: white;
    }
    .movie-card {
        background-color: #1F2937;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #E50914;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    movies_path = os.path.join(os.path.dirname(__file__), "data", "raw", "movies.csv")
    ratings_path = os.path.join(os.path.dirname(__file__), "data", "raw", "ratings.csv")
    ratings_df, movies_df, _ = load_and_clean_data(movies_path, ratings_path)
    return ratings_df, movies_df

def main():
    st.title("🍿 Hybrid Movie Recommendation System")
    st.markdown("### Discover your next favorite movie!")
    
    with st.spinner("Loading movie database..."):
        ratings_df, movies_df = load_data()
    
    # Input section
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=600, value=1, step=1)
    with col2:
        # Create a dropdown with movie titles for easier selection
        movie_titles = movies_df['title'].tolist()
        movie_title = st.selectbox("Select a movie you liked:", movie_titles, index=movie_titles.index("Toy Story (1995)") if "Toy Story (1995)" in movie_titles else 0)
    
    st.markdown("---")
    
    if st.button("Generate Recommendations 🚀"):
        with st.spinner("Analyzing your preferences..."):
            try:
                recs = hybrid_recommendations(
                    user_id=user_id,
                    movie_title=movie_title,
                    movies_df=movies_df,
                    ratings_df=ratings_df
                )
                
                if isinstance(recs, str):
                    st.error(recs)
                else:
                    st.success("Here are our top picks for you:")
                    
                    for _, row in recs.head(10).iterrows():
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3 style='margin-bottom: 5px; margin-top: 0;'>{row['title']}</h3>
                            <p style='color: #9CA3AF; margin-bottom: 10px;'><i>{row['genres']}</i></p>
                            <div style='display: flex; justify-content: space-between;'>
                                <span><strong style='color: #E50914;'>Match Score:</strong> {row['final_score']:.2f} / 5.0</span>
                                <span><strong>Predicted Rating:</strong> ⭐ {row['predicted_rating']:.2f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
