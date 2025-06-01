import pandas as pd
import itertools

# Load dataset (tab-separated, NO HEADERS)
df = pd.read_csv("data/ml-100k/u.data", delimiter="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])

# Drop unnecessary column (timestamp)
df = df.drop(columns=["timestamp"])

# List to store pairwise preferences
pairwise_prefs = []
# Store pairs with equal ratings
equal_prefs = []  


# Group by user (compare movies rated by the same user)
for user_id, user_group in df.groupby("user_id"):
    movies = user_group[["item_id", "rating"]].values  # Extract movie IDs and ratings
    
    # Generate all possible movie pairs for this user
    for (movie_a, rating_a), (movie_b, rating_b) in itertools.combinations(movies, 2):
        if rating_a > rating_b:
            pairwise_prefs.append((user_id, movie_a, movie_b))
        elif rating_b > rating_a:
            pairwise_prefs.append((user_id, movie_b, movie_a))
        else:
            equal_prefs.append((user_id, movie_a, movie_b))  # Store ties

# Convert to DataFrame
pairwise_df = pd.DataFrame(pairwise_prefs, columns=["User_ID", "Preferred", "Not_Preferred"])

# Save to CSV (No index column)
pairwise_df.to_csv("user_pairwise_preferences.csv", index=False)
equal_df = pd.DataFrame(equal_prefs, columns=["User_ID", "Movie_A", "Movie_B"])
equal_df.to_csv("equal_preferences.csv", index=False)

print("✅ User-specific pairwise preferences generated successfully!")

