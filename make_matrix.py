import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
import random
import pickle
from scipy import sparse

# Load user pairwise preferences data
df = pd.read_csv("generated_data/user_pairwise_preferences.csv")

# Initialize LightFM Dataset
dataset = Dataset()
dataset.fit(df["User_ID"].unique(), np.concatenate([df["Preferred"].unique(), df["Not_Preferred"].unique()]))

# Create interaction matrix based on the pairwise preferences
interactions = []

for _, row in df.iterrows():
    user_id = row["User_ID"]
    preferred_item = row["Preferred"]
    not_preferred_item = row["Not_Preferred"]
    
    # Set interaction for the preferred item as 1, and not preferred as 0
    interactions.append((user_id, preferred_item, 1))  # User prefers this item
    interactions.append((user_id, not_preferred_item, 0))  # User does not prefer this item

# Build interaction matrix
(interaction_matrix, weights) = dataset.build_interactions(interactions)



# Save interaction matrix
sparse.save_npz("interaction_matrix.npz", interaction_matrix)

# Save dataset mappings (contains user/item ID mappings)
with open("lightfm_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)
