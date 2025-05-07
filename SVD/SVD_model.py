import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load the dataset
df = pd.read_csv("generated_data/cleaned_ratings.csv")

# Create a user-item matrix (pivot table)
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Split into training and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Create training matrix for SVD (user-item matrix)
train_user_item_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Apply SVD
svd = TruncatedSVD(n_components=20, random_state=42)  # You can adjust n_components
svd_matrix = svd.fit_transform(train_user_item_matrix)

# Reconstruct the matrix (predicted ratings)
reconstructed_matrix = np.dot(svd_matrix, svd.components_)

# Make predictions
predicted_ratings = pd.DataFrame(reconstructed_matrix, columns=train_user_item_matrix.columns, index=train_user_item_matrix.index)

# Example: predict the rating for a specific user-item pair
user_id = 196  # Example user
item_id = 242  # Example item

predicted_rating = predicted_ratings.loc[user_id, item_id]
print(f"Predicted rating for user {user_id} on item {item_id}: {predicted_rating}")


# Step 9: Evaluate the model using RMSE on the test set

# Create the test matrix (user-item matrix)
test_user_item_matrix = test_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Predict ratings for the test set
test_predicted_ratings = predicted_ratings.loc[test_user_item_matrix.index, test_user_item_matrix.columns]

# Flatten both matrices (true values and predicted values)
test_true_values = test_user_item_matrix.values.flatten()
test_predicted_values = test_predicted_ratings.values.flatten()

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(test_true_values, test_predicted_values))
print(f"RMSE: {rmse}")
