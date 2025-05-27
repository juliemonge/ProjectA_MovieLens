from BPR import BPR
import pandas as pd

df = pd.read_csv("generated_data/user_pairwise_preferences.csv")

# Map user and item IDs to indices
user2id = {u: idx for idx, u in enumerate(df["User_ID"].unique())}
item_ids = pd.concat([df["Preferred"], df["Not_Preferred"]])
item2id = {i: idx for idx, i in enumerate(item_ids.unique())}

# Encode the data
triplets = df.apply(lambda row: (user2id[row["User_ID"]],
                                 item2id[row["Preferred"]],
                                 item2id[row["Not_Preferred"]]), axis=1)
triplets = triplets.tolist()

num_users = len(user2id)
num_items = len(item2id)

bpr = BPR(num_users=num_users, num_items=num_items)
bpr.train(data=triplets, epochs=5, eval_df=df, user2id=user2id, item2id=item2id)
bpr.save_bpr_model(user2id, item2id, path="bpr_model1_with evaluations.pkl")