import numpy as np
import pickle
import random
import pandas as pd

# Load MovieLens item metadata
movies = pd.read_csv("data/ml-100k/u.item", sep='|', header=None, encoding='latin-1', usecols=[0, 1])
movies.columns = ["item_id", "title"]
item_id_to_title = dict(zip(movies["item_id"], movies["title"]))

# Load ground truth ratings
ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])
user_relevant_items = (
    ratings.sort_values(by=["user_id", "rating"], ascending=[True, False])
    .groupby("user_id")["item_id"]
    .apply(lambda x: x.head(5).tolist())
    .to_dict()
)

# Load the trained model
with open("best_model5.pkl", "rb") as f:
    model = pickle.load(f)

user_factors = model["user_factors"]
item_factors = model["item_factors"]
user2id = model["user2id"]
item2id = model["item2id"]
id2item = {v: k for k, v in item2id.items()}

# --- Utility functions ---
def generate_user_groups(user_ids, group_size=5, num_groups=1, seed=42):
    random.seed(seed)
    user_ids = list(user_ids)
    random.shuffle(user_ids)
    return [user_ids[i*group_size:(i+1)*group_size] for i in range(num_groups) if len(user_ids[i*group_size:(i+1)*group_size]) == group_size]

def predict_scores_for_users(user_factors, item_factors, group):
    return {user: user_factors[user] @ item_factors.T for user in group}

def average_aggregation(predictions):
    return sum(predictions.values()) / len(predictions)

def recommend_items_for_group(aggregated_scores, id2item, top_k=10):
    top_indices = np.argsort(-aggregated_scores)[:top_k]
    return [id2item[i] for i in top_indices]

def precision_at_k_group(group_users, recommended_items, ground_truth, k=10):
    precisions = []
    for internal_user_id in group_users:
        raw_user_id = list(user2id.keys())[list(user2id.values()).index(internal_user_id)]
        relevant = set(ground_truth.get(raw_user_id, []))
        if not relevant:
            continue
        hits = sum(1 for item in recommended_items[:k] if item in relevant)
        precisions.append(hits / k)
    return np.mean(precisions) if precisions else 0

# --- Run Recommendation + Evaluation ---
all_user_ids = list(user2id.values())
groups = generate_user_groups(all_user_ids, group_size=5, num_groups=1)

for i, group in enumerate(groups, 1):
    predictions = predict_scores_for_users(user_factors, item_factors, group)
    aggregated = average_aggregation(predictions)
    recommendations = recommend_items_for_group(aggregated, id2item, top_k=10)

    print(f"\n🎬 Group {i} internal user IDs: {group}")
    print("Recommended item IDs:", recommendations)

    # Map internal → raw user IDs
    raw_group_users = [list(user2id.keys())[list(user2id.values()).index(uid)] for uid in group]
    print("Raw user IDs:", raw_group_users)

    for user in raw_group_users:
        print(f"Relevant items for user {user}:", user_relevant_items.get(user, []))

    precision = precision_at_k_group(group, recommendations, user_relevant_items, k=10)
    print(f"✅ Precision@10 for Group {i}: {precision:.2%}")

print("\n🎯 Sanity check: Are recommended IDs known to the model?")
for item in recommendations:
    print(f"Item {item}: in model? {'Yes' if item in item2id else 'No'}")



"""
for i, group in enumerate(groups, 1):
    predictions = predict_scores_for_users(user_factors, item_factors, group)
    aggregated = average_aggregation(predictions)
    recommendations = recommend_items_for_group(aggregated, id2item, top_k=10)

    print(f"\n🎬 Recommended items for Group {i} (users {group}):")
    titles = [item_id_to_title.get(item_id, f"Movie {item_id}") for item_id in recommendations]
    print(titles)

    precision = precision_at_k_group(group, recommendations, user_relevant_items, k=10)
    print(f"📈 Precision@10 for group: {precision:.2%}")
"""