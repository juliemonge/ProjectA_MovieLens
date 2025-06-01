import numpy as np
import pickle
import random
import pandas as pd

# Load MovieLens item metadata
movies = pd.read_csv("data/ml-100k/u.item", sep='|', header=None, encoding='latin-1', usecols=[0, 1])
movies.columns = ["item_id", "title"]
item_id_to_title = dict(zip(movies["item_id"], movies["title"]))

# Load the trained model
with open("best_model5.pkl", "rb") as f:
    model = pickle.load(f)

user_factors = model["user_factors"]
item_factors = model["item_factors"]
user2id = model["user2id"]
item2id = model["item2id"]
id2item = {v: k for k, v in item2id.items()}

# Group generation
def generate_user_groups(user_ids, group_size=4, num_groups=1, seed=None):
    rng = random.Random(seed)
    user_ids = list(user_ids)
    rng.shuffle(user_ids)
    groups = []
    for i in range(num_groups):
        group = user_ids[i*group_size:(i+1)*group_size]
        if len(group) == group_size:
            groups.append(group)
    return groups

# Predict scores for each user in a group
def predict_scores_for_users(user_factors, item_factors, group):
    return {user: user_factors[user] @ item_factors.T for user in group}

# Fairness-aware aggregation: combine average and least misery
def fairness_aware_aggregation(predictions, alpha=0.5):
    user_scores = list(predictions.values())
    avg_scores = np.mean(user_scores, axis=0)
    least_misery_scores = np.min(user_scores, axis=0)
    return alpha * avg_scores + (1 - alpha) * least_misery_scores

# Recommend top-k items based on aggregated scores
def recommend_items_for_group(aggregated_scores, id2item, top_k=10):
    top_indices = np.argsort(-aggregated_scores)[:top_k]
    return [id2item[i] for i in top_indices]

# Get top relevant items from training data for precision@k
def get_user_relevant_items(pairwise_df, user_id, top_k=10):
    return pairwise_df[pairwise_df["User_ID"] == user_id]["Preferred"].value_counts().index[:top_k].tolist()

# Evaluate group recommendation using precision@k
def evaluate_precision_at_k(recommended_items, group, pairwise_df, user2id, k=10):
    relevant_items = set()
    for internal_id in group:
        raw_user_id = [k for k, v in user2id.items() if v == internal_id][0]
        relevant_items.update(get_user_relevant_items(pairwise_df, raw_user_id, top_k=k))
    hits = sum(1 for item in recommended_items if item in relevant_items)
    return hits / k

# Load pairwise preference data
pairwise_df = pd.read_csv("generated_data/user_pairwise_preferences.csv")

# Run the fairness-aware strategy
all_user_ids = list(user2id.values())
groups = generate_user_groups(all_user_ids, group_size=4, num_groups=1, seed=7)

for i, group in enumerate(groups, 1):
    predictions = predict_scores_for_users(user_factors, item_factors, group)
    aggregated = fairness_aware_aggregation(predictions, alpha=0.5)
    recommendations = recommend_items_for_group(aggregated, id2item, top_k=10)
    
    print(f"\n🎬 Group {i} internal user IDs: {group}")
    print(f"Recommended item IDs: {recommendations}")
    
    precision = evaluate_precision_at_k(recommendations, group, pairwise_df, user2id, k=10)
    print(f"✅ Precision@10 for Group {i}: {precision * 100:.2f}%")
    
    titles = [item_id_to_title.get(item_id, f"Movie {item_id}") for item_id in recommendations]
    print(f"🎥 Titles: {titles}")
