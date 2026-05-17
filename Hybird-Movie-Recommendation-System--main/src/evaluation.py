
import pandas as pd
from collections import defaultdict
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split


def load_data(movies_path, rating_path):
    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(rating_path)
    movies["genres"] = movies["genres"].str.replace('|', " ", regex=False)
    return movies, ratings


def train_split(rating_df):
    reader = Reader(rating_scale=(rating_df["rating"].min(), rating_df["rating"].max()))
    data   = Dataset.load_from_df(rating_df[['userId', 'movieId', 'rating']], reader)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(random_state=42)
    model.fit(train)
    predictions = model.test(test)
    return model, predictions


def evaluate_rmse_mae(predictions):
    
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)
    return round(rmse, 4), round(mae, 4)




def evaluate_precision_recall_f1(predictions, k=10, threshold=3.5):
    user_preds = defaultdict(list)
    for pred in predictions:
        user_preds[pred.uid].append((pred.est, pred.r_ui))

    precisions, recalls = [], []

    for uid, user_ratings in user_preds.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        hits_in_topk   = sum(1 for est, true in top_k        if true >= threshold)
        total_relevant = sum(1 for est, true in user_ratings  if true >= threshold)

        precisions.append(hits_in_topk / k)
        recalls.append(hits_in_topk / total_relevant if total_relevant > 0 else 0)

    avg_p  = sum(precisions) / len(precisions)
    avg_r  = sum(recalls)    / len(recalls)
    avg_f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) > 0 else 0

    return round(avg_p, 4), round(avg_r, 4), round(avg_f1, 4)




def print_report(rmse, mae, precision, recall, f1, k=10, threshold=3.5):
    print("\n" + "="*47)
    print("   COLLABORATIVE FILTERING  —  EVALUATION REPORT")
    print("="*47)

    print("\n Rating Prediction Accuracy")
    print(f"   RMSE : {rmse}   (lower = better)")
    print(f"   MAE  : {mae}   (lower = better)")

    print(f"\n Recommendation Quality  (K={k}, threshold≥{threshold})")
    print(f"   Precision@{k} : {precision}   (quality  of top-{k} list)")
    print(f"   Recall@{k}    : {recall}   (coverage of liked movies)")
    print(f"   F1@{k}        : {f1}   (balance of both)")
    print("="*47 + "\n")




if __name__ == "__main__":

    MOVIES_PATH  = "data/raw/movies.csv"
    RATINGS_PATH = "data/raw/ratings.csv"
    K            = 10       
    THRESHOLD    = 3.5      

    print("Loading data...")
    movies_df, ratings_df = load_data(MOVIES_PATH, RATINGS_PATH)

    print("Training SVD model on 80% split...")
    
    model, predictions = train_split(ratings_df)

    print("Computing metrics...")
    rmse, mae             = evaluate_rmse_mae(predictions)
    precision, recall, f1 = evaluate_precision_recall_f1(
                                predictions, k=K, threshold=THRESHOLD)

    print_report(rmse, mae, precision, recall, f1, k=K, threshold=THRESHOLD)