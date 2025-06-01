import pandas as pd
import itertools

# Load dataset (tab-separated, NO HEADERS)
df = pd.read_csv("data/ml-100k/u.data", delimiter="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])

# Drop unnecessary column
df = df.drop(columns=["timestamp"])

# Initialize list to store filtered pairwise preferences
pairwise_prefs = []

# Optionally, track skipped or equal rating pairs
skipped_equal_or_midrange = []

# Group by user to compare only movies rated by the same user
for user_id, user_group in df.groupby("user_id"):
    movies = user_group[["item_id", "rating"]].values

    # Generate all combinations of movie pairs
    for (movie_a, rating_a), (movie_b, rating_b) in itertools.combinations(movies, 2):
        # Only include if one rating is in [4,5] and the other in [1,2,3]
        if rating_a >= 4 and rating_b <= 3:
            pairwise_prefs.append((user_id, movie_a, movie_b))
        elif rating_b >= 4 and rating_a <= 3:
            pairwise_prefs.append((user_id, movie_b, movie_a))
        else:
            skipped_equal_or_midrange.append((user_id, movie_a, movie_b))  # Not used but could be logged

# Convert to DataFrame
pairwise_df = pd.DataFrame(pairwise_prefs, columns=["User_ID", "Preferred", "Not_Preferred"])

# Save to CSV
pairwise_df.to_csv("filtered_user_pairwise_preferences.csv", index=False)

print(f"✅ Filtered pairwise preferences saved. Total pairs: {len(pairwise_df)}")
