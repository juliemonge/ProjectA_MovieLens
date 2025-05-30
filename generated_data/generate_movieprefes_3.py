import pandas as pd
import itertools

# Load dataset
df = pd.read_csv("data/ml-100k/u.data", delimiter="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])
df = df.drop(columns=["timestamp"])

pairwise_prefs = []

# Group by user
for user_id, user_group in df.groupby("user_id"):
    movies = user_group[["item_id", "rating"]].values

    for (movie_a, rating_a), (movie_b, rating_b) in itertools.combinations(movies, 2):
        # Ignore if same rating
        if rating_a == rating_b:
            continue

        # Determine preferred vs not preferred
        if rating_a > rating_b:
            pairwise_prefs.append((user_id, movie_a, movie_b))
        elif rating_b > rating_a:
            pairwise_prefs.append((user_id, movie_b, movie_a))
        # rating_a == rating_b is ignored automatically

# Save to DataFrame
pairwise_df = pd.DataFrame(pairwise_prefs, columns=["User_ID", "Preferred", "Not_Preferred"])
pairwise_df.to_csv("user_pairwise_preferences_newfilter.csv", index=False)

print(f"✅ Pairwise preferences (including 3s) saved. Total pairs: {len(pairwise_df)}")
