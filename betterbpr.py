import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
import random
import pickle
from scipy import sparse

# Load user pairwise preferences data
df = pd.read_csv("generated_data/user_pairwise_preferences.csv")


interaction_matrix = sparse.load_npz("interaction_matrix.npz")

# Load dataset mappings
with open("lightfm_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Train a BPR Model
model = LightFM(loss="bpr")  # Bayesian Personalized Ranking
model.fit(interaction_matrix, epochs=5, num_threads=4)

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
#user_id = 1
#recommendations = recommend(model, dataset, user_id, num_items=5)
#print(f"Recommended movies for User {user_id}: {recommendations}")


def recommend_group(model, dataset, user_ids, num_items=5):
    """
    Recommend top items for a group of users by averaging their predicted scores.
    """
    # Get item indices
    item_ids = np.arange(len(dataset.mapping()[2]))

    # Predict scores for each user
    group_scores = np.zeros(len(item_ids))

    for user_id in user_ids:
        user_scores = model.predict(np.repeat(user_id, len(item_ids)), item_ids)
        group_scores += user_scores  # Sum scores across users

    # Average the scores
    group_scores /= len(user_ids)

    # Get top items
    top_items = np.argsort(-group_scores)[:num_items]

    # Reverse map to original item IDs
    reverse_item_mapping = {v: k for k, v in dataset.mapping()[2].items()}
    recommended_movies = [reverse_item_mapping[i] for i in top_items]

    return recommended_movies

# Pick 5 random users from your dataset
unique_users = df["User_ID"].unique()
group_user_ids = random.sample(list(unique_users), 5)

# Recommend for the group
group_recommendations = recommend_group(model, dataset, group_user_ids, num_items=5)

print(f"Group User IDs: {group_user_ids}")
print(f"Recommended movies for the group: {group_recommendations}")

# def recommend_group_lms(model, dataset, user_ids, num_items = 5):
#     """Recommend top movies for the group based on least misery strategy"""

#      # Get item indices
#     item_ids = np.arange(len(dataset.mapping()[2]))

#     # Predict scores for each user
#     group_scores = np.zeros(len(item_ids))

#     for user_id in user_ids:
#         user_scores = model.predict(np.repeat(user_id, len(item_ids)), item_ids)
#         group_scores += user_scores  # Sum scores across users