import pandas as pd

def load_and_clean_data(movies_path, ratings_path):

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path) 
    
    # No missing values or duplications .
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    # Merge both tables 
    combined_data = pd.merge(ratings, movies, on='movieId' , how='inner') 
    
    return ratings, movies, combined_data

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    movies_path = os.path.join(base_dir, "data", "raw", "movies.csv")
    ratings_path = os.path.join(base_dir, "data", "raw", "ratings.csv")
    
    ratings_df, movies_df, full_df = load_and_clean_data(movies_path, ratings_path)
    
    print("\n---Merged Data---")
    print(full_df[['userId', 'title', 'rating', 'genres']].head())    
    
    