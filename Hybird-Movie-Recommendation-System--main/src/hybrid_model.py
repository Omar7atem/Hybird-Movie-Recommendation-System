

import pandas as pd
from content_based import build_content_engine, get_content_recommendations
from collaborative import train_collaborative_model, predict_rating

def hybrid_recommendations(user_id, movie_title, movies_df, ratings_df, cb_weight=0.6, cf_weight=0.4):
    
    # Content-Based
    cosine_sim = build_content_engine(movies_df)
    cb_recs = get_content_recommendations(movie_title, movies_df, cosine_sim)
    
    if isinstance(cb_recs, str):
        return cb_recs
    
    # Collaborative
    model = train_collaborative_model(ratings_df)
    
    # Hybrid
    results = []
    for _, row in cb_recs.iterrows():
        movie_id = movies_df[movies_df['title'] == row['title']]['movieId'].values[0]
        
        
        cf_score = predict_rating(user_id, movie_id, model)
        
    
        idx = movies_df[movies_df['title'] == movie_title].index[0]
        movie_idx = movies_df[movies_df['title'] == row['title']].index[0]
        cb_score = cosine_sim[idx][movie_idx]
        
    
        final_score = (cb_weight * cb_score) + (cf_weight * cf_score / 5)
        
        results.append({
            'title': row['title'],
            'genres': row['genres'],
            'similarity': round(cb_score, 3),
            'predicted_rating': round(cf_score, 3),
            'final_score': round(final_score, 3)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('final_score', ascending=False)
    
    return results_df


if __name__ == "__main__":
    from data_processing import load_and_clean_data
    import os
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    movies_path = os.path.join(base_dir, "data", "raw", "movies.csv")
    ratings_path = os.path.join(base_dir, "data", "raw", "ratings.csv")
    
    ratings_df, movies_df, _ = load_and_clean_data(movies_path, ratings_path)
    
    recs = hybrid_recommendations(
        user_id=1,
        movie_title="Toy Story (1995)",
        movies_df=movies_df,
        ratings_df=ratings_df
    )
    
    print("\n--- Hybrid Recommendations ---")
    print(recs)