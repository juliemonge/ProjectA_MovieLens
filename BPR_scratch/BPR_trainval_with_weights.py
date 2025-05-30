from sklearn.model_selection import train_test_split
import pandas as pd
from BPR_with_weights import BPR  # Ensure you're using the updated BPR class with weight support

# === Load data ===
df = pd.read_csv("generated_data/user_pairwise_preferences_with_weights.csv")

# === Encode user and item IDs ===
user2id = {u: idx for idx, u in enumerate(df["User_ID"].unique())}
item_ids = pd.concat([df["Preferred"], df["Not_Preferred"]])
item2id = {i: idx for idx, i in enumerate(item_ids.unique())}

# Map to index and include weight
df["user_idx"] = df["User_ID"].map(user2id)
df["preferred_idx"] = df["Preferred"].map(item2id)
df["not_preferred_idx"] = df["Not_Preferred"].map(item2id)

# Ensure 'Weight' column exists (if not, default to 1.0)
if "Weight" not in df.columns:
    df["Weight"] = 1.0

# Create list of (user_idx, preferred_idx, not_preferred_idx, weight)
triplets = df[["user_idx", "preferred_idx", "not_preferred_idx", "Weight"]].values.tolist()

# === Split into training and validation ===
train_triplets, val_triplets = train_test_split(triplets, test_size=0.1, random_state=42)

# Validation DataFrame (for evaluation metrics — uses original user/item IDs)
val_df = pd.DataFrame(val_triplets, columns=["user_idx", "preferred_idx", "not_preferred_idx", "Weight"])
val_df["User_ID"] = val_df["user_idx"].map({v: k for k, v in user2id.items()})
val_df["Preferred"] = val_df["preferred_idx"].map({v: k for k, v in item2id.items()})
val_df["Not_Preferred"] = val_df["not_preferred_idx"].map({v: k for k, v in item2id.items()})

# === Initialize and train model ===
num_users = len(user2id)
num_items = len(item2id)
bpr = BPR(num_users=num_users, num_items=num_items)

bpr.train(data=train_triplets, eval_df=val_df, user2id=user2id, item2id=item2id)

# === Save the final model ===
bpr.save_bpr_model(user2id, item2id, path="final_model_weighted.pkl")
