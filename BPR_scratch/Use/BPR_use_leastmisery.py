import numpy as np
import pickle
import random
import pandas as pd
from collections import defaultdict

# Load MovieLens item metadata
movies = pd.read_csv("data/ml-100k/u.item", sep='|', header=None, encoding='latin-1', usecols=[0, 1])
movies.columns = ["item_id", "title"]
item_id_to_title = dict(zip(movies["item_id"], movies["title"]))

# Load original ratings data
ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])

# Load trained BPR model
with open("best_model5.pkl", "rb") as f:
    model = pickle.load(f)

user_factors = model["user_factors"]
item_factors = model["item_factors"]
user2id = model["user2id"]
item2id = model["item2id"]
id2item = {v: k for k, v in item2id.items()}
id2user = {v: k for k, v in user2id.items()}

# ---- Define Helper Functions ----
def generate_user_groups(user_ids, group_size=4, num_groups=1):
    user_ids = list(user_ids)
    random.shuffle(user_ids)
    
    groups = []
    for i in range(num_groups):
        group = user_ids[i*group_size:(i+1)*group_size]
        if len(group) == group_size:
            groups.append(group)
    return groups

def predict_scores_for_users(user_factors, item_factors, group):
    return {user: user_factors[user] @ item_factors.T for user in group}

def least_misery_aggregation(predictions):
    all_user_scores = np.stack(list(predictions.values()))
    return np.min(all_user_scores, axis=0)

def recommend_items_for_group(aggregated_scores, id2item, top_k=10):
    top_indices = np.argsort(-aggregated_scores)[:top_k]
    return [id2item[i] for i in top_indices]

def get_relevant_items(user_raw_id, ratings_df, threshold=4):
    return ratings_df[(ratings_df["user_id"] == user_raw_id) & (ratings_df["rating"] >= threshold)]["item_id"].tolist()

def compute_group_precision_at_k(group, recommendations, ratings_df, id2user, k=10):
    hits = 0
    total = len(group) * k

    for internal_uid in group:
        raw_uid = id2user[internal_uid]
        relevant_items = set(get_relevant_items(raw_uid, ratings_df))

        overlap = set(recommendations).intersection(relevant_items)
        hits += len(overlap)

    precision = hits / total if total > 0 else 0
    return precision * 100

# ---- MAIN EXECUTION ----
all_user_ids = list(user2id.values())
groups = generate_user_groups(all_user_ids, group_size=4, num_groups=1)

for i, group in enumerate(groups, 1):
    print(f"\n🎬 Group {i} internal user IDs:", group)

    predictions = predict_scores_for_users(user_factors, item_factors, group)
    aggregated = least_misery_aggregation(predictions)
    recommendations = recommend_items_for_group(aggregated, id2item, top_k=10)

    print("Recommended item IDs:", recommendations)

    # Evaluate precision@k
    precision = compute_group_precision_at_k(group, recommendations, ratings, id2user, k=10)
    print(f"✅ Precision@10 for Group {i}: {precision:.2f}%")

    # Show titles
    titles = [item_id_to_title.get(item_id, f"Movie {item_id}") for item_id in recommendations]
    print("🎥 Titles:", titles)

