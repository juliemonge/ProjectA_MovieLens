import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

# Load dataset (MovieLens 100K)
df = pd.read_csv("data/ml-100k/u.data", delimiter="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])

# Convert ratings to binary implicit feedback (Like = 1, Dislike = 0)
df["interaction"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

# Remove timestamp and rating columns
df = df[["user_id", "item_id", "interaction"]]

# Initialize LightFM Dataset
dataset = Dataset()
dataset.fit(df["user_id"].unique(), df["item_id"].unique())

# Build interaction matrix
(interactions, weights) = dataset.build_interactions(zip(df["user_id"], df["item_id"]))

# Train a BPR Model
model = LightFM(loss="bpr")  # Bayesian Personalized Ranking
model.fit(interactions, epochs=30, num_threads=4)

# Get recommendations for a user
def recommend(model, dataset, user_id, num_items=5):
    """ Recommend top movies for a given user using BPR model. """
    item_ids = np.arange(len(dataset.mapping()[2]))  # Get item indices
    scores = model.predict(user_id, item_ids)  # Predict scores
    top_items = np.argsort(-scores)[:num_items]  # Sort by highest score
    
   # Convert item indices back to actual movie IDs
    reverse_item_mapping = dataset.mapping()[2]
    recommended_movies = [reverse_item_mapping[i] for i in top_items]
    
    return recommended_movies

# Example: Recommend movies for user 1
user_id = 1
recommendations = recommend(model, dataset, user_id, num_items=5)
print(f"Recommended movies for User {user_id}: {recommendations}")
