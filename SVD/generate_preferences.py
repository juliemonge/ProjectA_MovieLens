import pandas as pd
import itertools

# Load dataset (tab-separated, NO HEADERS)
df = pd.read_csv("data/ml-100k/u.data", delimiter="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])

# Drop unnecessary column (timestamp)
df = df.drop(columns=["timestamp"])

# Save file
df.to_csv("data/cleaned_ratings.csv", index=False, header=True)

