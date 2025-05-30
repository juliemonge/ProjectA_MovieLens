import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

# === Step 1: Load and clean dataset ===
df = pd.read_csv("generated_data/cleaned_ratings.csv")

# Ensure consistent int types for safety
df['user_id'] = df['user_id'].astype(int)
df['item_id'] = df['item_id'].astype(int)

# === Step 2: Split data ===
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
test_data['user_id'] = test_data['user_id'].astype(int)
test_data['item_id'] = test_data['item_id'].astype(int)

# === Step 3: Create training matrix (NaN for missing) ===
train_user_item_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating')

# === Step 4: Fill missing values with user mean ===
train_user_item_matrix_filled = train_user_item_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

# === Step 5: SVD model ===
svd = TruncatedSVD(n_components=50, random_state=42)
svd_matrix = svd.fit_transform(train_user_item_matrix_filled)
reconstructed_matrix = np.dot(svd_matrix, svd.components_)

# === Step 6: Predicted ratings ===
predicted_ratings = pd.DataFrame(reconstructed_matrix,
                                 index=train_user_item_matrix.index,
                                 columns=train_user_item_matrix.columns)

# === Step 7: RMSE Evaluation ===
test_data_filtered = test_data[
    (test_data['user_id'].isin(predicted_ratings.index)) &
    (test_data['item_id'].isin(predicted_ratings.columns))
]

test_preds = test_data_filtered.apply(
    lambda row: predicted_ratings.loc[row['user_id'], row['item_id']],
    axis=1
)
rmse = np.sqrt(mean_squared_error(test_data_filtered['rating'], test_preds))
print(f"RMSE: {rmse:.4f}")

# === Step 8: Group Recommendation ===
group_of_users = [196, 186, 22, 244, 166]  # Example group
all_items = set(predicted_ratings.columns)

# Seen items per user
rated_items_by_user = {
    user: set(df[df['user_id'] == user]['item_id']) for user in group_of_users
}
seen_items_group = set.union(*rated_items_by_user.values())
unseen_items = all_items - seen_items_group

# Predicted ratings for group members (unseen only)
group_predicted_ratings = predicted_ratings.loc[group_of_users, list(unseen_items)]

# Aggregate scores (average)
average_predicted_ratings = group_predicted_ratings.mean(axis=0)
top_5_group_recommendations = average_predicted_ratings.sort_values(ascending=False).head(5)

# === Step 9: Print recommendations ===
print("\nTop 5 group recommendations (based on average predicted rating):")
for item_id, score in top_5_group_recommendations.items():
    print(f"  Movie {item_id} with predicted score {score:.2f}")

# === Step 10: Evaluation ===
test_relevant_by_user = {
    user: set(test_data[(test_data['user_id'] == user) & (test_data['rating'] >= 4)]['item_id'])
    for user in group_of_users
}
group_relevant_items = set.union(*test_relevant_by_user.values())

# Ensure we only compare items that are actually predictable
group_relevant_items = group_relevant_items & set(predicted_ratings.columns)

recommended_items = set(top_5_group_recommendations.index)
hits = recommended_items & group_relevant_items

precision = len(hits) / 5
recall = len(hits) / len(group_relevant_items) if group_relevant_items else 0
hitrate = 1 if len(hits) > 0 else 0

# NDCG
dcg = 0
for rank, item in enumerate(top_5_group_recommendations.index, start=1):
    if item in group_relevant_items:
        dcg += 1 / np.log2(rank + 1)
idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(group_relevant_items), 5) + 1))
ndcg = dcg / idcg if idcg > 0 else 0

# === Final Metrics ===
print("\nGroup Recommendation Evaluation:")
print(f"Precision@5: {precision:.4f}")
print(f"Recall@5: {recall:.4f}")
print(f"HitRate@5: {hitrate:.4f}")
print(f"NDCG@5: {ndcg:.4f}")

# === Debug: Check overlap ===
print("\n--- Evaluation Debug ---")
print("Recommended item IDs:", list(top_5_group_recommendations.index))
print("Relevant group item IDs:", list(group_relevant_items))
print("Common (hits):", list(hits))
print("Do recommended items exist in test set?")
for item in recommended_items:
    print(f"Item {item}: {'✅' if item in test_data['item_id'].values else '❌'}")
