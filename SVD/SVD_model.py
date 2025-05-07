import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

# Step 1: Load your dataset
df = pd.read_csv("generated_data/cleaned_ratings.csv")

# Step 2: Create the user-item matrix (pivot table) for the full dataset
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Step 3: Split data into training and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Step 4: Create the training matrix (user-item matrix) from training data
train_user_item_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Step 5: Ensure that both train and test matrices have the same columns (item_id)
common_items = train_user_item_matrix.columns.intersection(test_data['item_id'].unique())
train_user_item_matrix = train_user_item_matrix[common_items]
test_user_item_matrix = test_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)[common_items]

# Step 6: Apply Truncated SVD for dimensionality reduction (latent factors)
svd = TruncatedSVD(n_components=20, random_state=42)  # You can adjust n_components for more/less complexity
svd_matrix = svd.fit_transform(train_user_item_matrix)

# Step 7: Reconstruct the matrix (predicted ratings)
reconstructed_matrix = np.dot(svd_matrix, svd.components_)

# Step 8: Create a DataFrame for the predicted ratings
predicted_ratings = pd.DataFrame(reconstructed_matrix, columns=train_user_item_matrix.columns, index=train_user_item_matrix.index)

# Step 9: Example: Predict the rating for a specific user-item pair
user_id = 196  # Example user
item_id = 242  # Example item

predicted_rating = predicted_ratings.loc[user_id, item_id]
print(f"Predicted rating for user {user_id} on item {item_id}: {predicted_rating}")

# Step 10: Evaluate the model using RMSE on the test set

# Create the test matrix (user-item matrix)
# Ensure that only common items are used for the test matrix
test_user_item_matrix = test_user_item_matrix[common_items]

# Predict ratings for the test set
test_predicted_ratings = predicted_ratings.loc[test_user_item_matrix.index, test_user_item_matrix.columns]

# Flatten both matrices (true values and predicted values)
test_true_values = test_user_item_matrix.values.flatten()
test_predicted_values = test_predicted_ratings.values.flatten()

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(test_true_values, test_predicted_values))
print(f"RMSE: {rmse}")



# Assuming the previous code for SVD model training is already executed

# Step 1: Define the group
group_of_users = [196, 186, 22, 244, 166]  # Example group

# Step 2: Get the set of all items (movies)
all_items = set(predicted_ratings.columns)

# Step 3: Find movies rated by each user
rated_items_by_user = {
    user: set(df[df['user_id'] == user]['item_id']) for user in group_of_users
}

# Step 4: Find movies that none of the group members have rated
# First, union of all rated items
all_rated_by_group = set.union(*rated_items_by_user.values())
unseen_items = all_items - all_rated_by_group

# Step 5: Collect predicted ratings for the unseen items
# We'll build a DataFrame with users as rows and unseen items as columns
group_predicted_ratings = predicted_ratings.loc[group_of_users, list(unseen_items)]


# Step 6: Compute average predicted rating per item across the group
average_predicted_ratings = group_predicted_ratings.mean(axis=0)

# Step 7: Recommend top 5 movies with highest average predicted ratings
top_5_group_recommendations = average_predicted_ratings.sort_values(ascending=False).head(5)

# Step 8: Print results
print("Top 5 group recommendations (based on average predicted rating):")
for item_id, avg_rating in top_5_group_recommendations.items():
    print(f"  Movie {item_id} with average predicted rating {avg_rating:.2f}")

