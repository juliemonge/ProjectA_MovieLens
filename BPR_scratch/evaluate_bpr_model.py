import pickle
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

# Load pairwise preference data
df = pd.read_csv("generated_data/user_pairwise_preferences.csv")

# Create test set: one positive per user
def create_test_set(df):
    test_data = []
    train_data = []
    grouped = df.groupby("User_ID")

    for user_id, group in grouped:
        if len(group) < 2:
            continue
        test_row = group.sample(n=1)
        test_data.append((user_id, test_row["Preferred"].values[0]))
        train_data.append(group.drop(test_row.index))

    train_df = pd.concat(train_data)
    return train_df, test_data

train_df, test_triplets = create_test_set(df)

# Load trained BPR model
with open("BPR_scratch/bpr_model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract learned embeddings and mappings
user_factors = model["user_factors"]
item_factors = model["item_factors"]
user_to_index = model["user2id"]
item_to_index = model["item2id"]
index_to_item = {idx: item for item, idx in item_to_index.items()}

# Get all item indices
all_item_indices = list(index_to_item.keys())

# Recommend function
def recommend(user_id, k=5, exclude_items=None):
    if user_id not in user_to_index:
        return []

    u_idx = user_to_index[user_id]
    user_vector = user_factors[u_idx]
    scores = item_factors @ user_vector

    if exclude_items:
        exclude_indices = [item_to_index[i] for i in exclude_items if i in item_to_index]
        scores[exclude_indices] = -np.inf

    top_k_indices = np.argsort(-scores)[:k]
    return [index_to_item[i] for i in top_k_indices]

# Precision@K
def precision_at_k(test_triplets, k=5):
    hits = 0
    total = 0

    for user_id, true_item in tqdm(test_triplets, desc="Evaluating"):
        # Get items user has interacted with in train set
        user_train_items = set(train_df[train_df["User_ID"] == user_id]["Preferred"].values)

        recs = recommend(user_id, k=k, exclude_items=user_train_items)
        if true_item in recs:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0

# Run evaluation
K = 5
score = precision_at_k(test_triplets, k=K)
print(f"\n📈 Precision@{K}: {score:.4f}")
