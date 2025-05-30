import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("generated_data/cleaned_ratings.csv")

# Step 2: Binarize ratings (e.g., 1 if rating >= 4, else 0)
df['binary_rating'] = (df['rating'] >= 3).astype(int)

# Step 3: Train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Step 4: Create user-item matrix for training (sparse)
user_ids = train_data['user_id'].astype('category')
item_ids = train_data['item_id'].astype('category')

user_mapping = dict(enumerate(user_ids.cat.categories))
item_mapping = dict(enumerate(item_ids.cat.categories))

user_inverse_mapping = {v: k for k, v in user_mapping.items()}
item_inverse_mapping = {v: k for k, v in item_mapping.items()}

train_user_index = user_ids.cat.codes.values
train_item_index = item_ids.cat.codes.values

train_matrix = csr_matrix(
    (train_data['binary_rating'], (train_user_index, train_item_index)),
    shape=(len(user_mapping), len(item_mapping))
)

# Step 5: Train ALS model
als_model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
als_model.fit(train_matrix)

# Step 6: Group Recommendation

# Example group
group_of_users = [196, 186, 22, 244, 166]

# Filter out users/items not in training set
valid_user_ids = [u for u in group_of_users if u in user_inverse_mapping]
group_internal_ids = [user_inverse_mapping[u] for u in valid_user_ids]


# Get items rated by group members
rated_by_user = {
    u: set(df[(df['user_id'] == u) & (df['binary_rating'] == 1)]['item_id']) for u in group_of_users
}
all_rated_by_group = set.union(*rated_by_user.values())

# Get item internal IDs
rated_internal_ids = set(item_inverse_mapping[i] for i in all_rated_by_group if i in item_inverse_mapping)

# Step 7: Predict scores for unseen items for each group member
all_item_internal_ids = set(range(len(item_mapping)))
unseen_item_ids = list(all_item_internal_ids - rated_internal_ids)

group_scores = []

for user_id in group_of_users:
    if user_id not in user_inverse_mapping:
        print(f"Skipping user {user_id} – not in training data")
        continue

    uid = user_inverse_mapping[user_id]

    # Double-check this user exists in train_matrix
    if uid >= train_matrix.shape[0]:
        print(f"User ID {uid} out of bounds in train_matrix")
        continue

    try:
        scores = als_model.recommend(
            userid=uid,
            user_items=train_matrix[uid],
            N=len(unseen_item_ids),
            filter_items=list(rated_internal_ids),
            recalculate_user=True
        )
        scores_dict = dict(scores)
        group_scores.append(scores_dict)
    except Exception as e:
        print(f"Error for user {user_id}: {e}")


# Step 8: Average scores across group
item_scores = {}
for scores in group_scores:
    for item_id, score in scores.items():
        item_scores[item_id] = item_scores.get(item_id, []) + [score]

average_scores = {
    item_id: np.mean(scores) for item_id, scores in item_scores.items()
}

# Step 9: Top 5 recommendations
top_items = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)[:5]
top_5_group_recommendations = [(item_mapping[iid], score) for iid, score in top_items]

# Step 10: Print results
print("Top 5 group recommendations (based on average ALS scores):")
for item_id, score in top_5_group_recommendations:
    print(f"  Movie {item_id} with average predicted score {score:.2f}")