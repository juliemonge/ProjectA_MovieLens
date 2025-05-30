import pickle
import numpy as np
import pandas as pd
import random
from collections import defaultdict

# === Load model ===
with open("Models/best_model5.pkl", "rb") as f:
    model_data = pickle.load(f)

user_factors = model_data["user_factors"]
item_factors = model_data["item_factors"]
user2id = model_data["user2id"]
item2id = model_data["item2id"]
id2user = {v: k for k, v in user2id.items()}
id2item = {v: k for k, v in item2id.items()}

# === Load dataset ===
df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

# === Build user history (training) and true items (preferred, for eval) ===
user_history = defaultdict(set)
true_items_by_user = defaultdict(set)

for _, row in df.iterrows():
    uid_raw, iid_raw, rating = row["user_id"], row["item_id"], row["rating"]
    uid = user2id.get(uid_raw)
    iid = item2id.get(iid_raw)
    if uid is not None and iid is not None:
        user_history[uid].add(iid)
        if rating >= 4:
            true_items_by_user[uid].add(iid)

# === Pick a group of 5 random users ===
all_user_ids = list(user2id.values())
group_user_ids = random.sample(all_user_ids, 5)

# === Predict scores for all items per user ===
group_scores = np.array([np.dot(item_factors, user_factors[uid]) for uid in group_user_ids])

# === Least misery aggregation ===
aggregated_scores = np.min(group_scores, axis=0)

# === Mask known items ===
known_items_group = set().union(*[user_history[uid] for uid in group_user_ids])
for iid in known_items_group:
    aggregated_scores[iid] = -np.inf

# === Top 5 recommendations ===
top_items = np.argsort(-aggregated_scores)[:5]
recommended_item_ids = [id2item[i] for i in top_items]

# === Load movie names ===
item_id_to_name = {}
with open("data/ml-100k/u.item", encoding="ISO-8859-1") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            movie_id = int(parts[0])
            movie_title = parts[1]
            item_id_to_name[movie_id] = movie_title

recommended_movie_names = [item_id_to_name.get(mid, f"Unknown ID {mid}") for mid in recommended_item_ids]

# === Evaluation ===
group_true_items = set().union(*[true_items_by_user[uid] for uid in group_user_ids])
recommended_set = set(top_items)

hits = recommended_set & group_true_items
precision = len(hits) / 5
recall = len(hits) / len(group_true_items) if group_true_items else 0
hitrate = 1 if len(hits) > 0 else 0

# NDCG
dcg = 0
for rank, item in enumerate(top_items, start=1):
    if item in group_true_items:
        dcg += 1 / np.log2(rank + 1)
idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(group_true_items), 5) + 1))
ndcg = dcg / idcg if idcg > 0 else 0

# === Output ===
print("Group members (internal IDs):", group_user_ids)
print("Recommended Movies for Group:")
for name in recommended_movie_names:
    print(f"- {name}")

print("\nGroup Evaluation:")
print(f"Precision@5: {precision:.4f}")
print(f"Recall@5: {recall:.4f}")
print(f"HitRate@5: {hitrate:.4f}")
print(f"NDCG@5: {ndcg:.4f}")
