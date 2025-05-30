from sklearn.model_selection import train_test_split
import pandas as pd
from BPR import BPR


# Read and prepare data
df = pd.read_csv("generated_data/user_pairwise_preferences_max300.csv")

# Map user and item IDs to indices
user2id = {u: idx for idx, u in enumerate(df["User_ID"].unique())}
item_ids = pd.concat([df["Preferred"], df["Not_Preferred"]])
item2id = {i: idx for idx, i in enumerate(item_ids.unique())}

# Encode data as (user_idx, preferred_item_idx, not_preferred_item_idx)
triplets = df.apply(lambda row: (user2id[row["User_ID"]],
                                 item2id[row["Preferred"]],
                                 item2id[row["Not_Preferred"]]), axis=1).tolist()

# Split triplets into training and validation
train_triplets, val_triplets = train_test_split(triplets, test_size=0.1, random_state=42)

# Create validation DataFrame
val_df = pd.DataFrame(val_triplets, columns=["User_ID", "Preferred", "Not_Preferred"])

# Initialize model
num_users = len(user2id)
num_items = len(item2id)
bpr = BPR(num_users=num_users, num_items=num_items)

# Train model on training data and validate on val_df
bpr.train(data=train_triplets, eval_df=val_df, user2id=user2id, item2id=item2id)

# Save final model
bpr.save_bpr_model(user2id, item2id)
