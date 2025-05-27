import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pickle

class BPR:
    def __init__(self, num_users, num_items, latent_dim=64, learning_rate=0.01, reg=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.reg = reg

        # Initialize user and item embeddings
        self.user_factors = np.random.normal(0, 0.1, (num_users, latent_dim))
        self.item_factors = np.random.normal(0, 0.1, (num_items, latent_dim))

    def train(self, data, epochs=10):
        for epoch in range(epochs):
            np.random.shuffle(data)
            epoch_loss = 0
            for user, preferred, not_preferred in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}"):
                user, preferred, not_preferred = int(user), int(preferred), int(not_preferred)
                loss = self.update(user, preferred, not_preferred)
                epoch_loss += loss
            print(f"Epoch {epoch+1} completed. Total Loss: {epoch_loss:.4f}")

    def update(self, user, preferred, not_preferred):
        # Get embeddings
        user_vec = self.user_factors[user]
        preferred_vec = self.item_factors[preferred]
        not_preferred_vec = self.item_factors[not_preferred]

        # Compute difference
        x_uij = np.dot(user_vec, preferred_vec - not_preferred_vec)
        sigmoid = 1 / (1 + np.exp(-x_uij))

        # Gradients
        grad_user = (1 - sigmoid) * (preferred_vec - not_preferred_vec) + self.reg * user_vec
        grad_preferred = (1 - sigmoid) * user_vec + self.reg * preferred_vec
        grad_not_preferred = -(1 - sigmoid) * user_vec + self.reg * not_preferred_vec

        # Update vectors
        self.user_factors[user] -= self.lr * grad_user
        self.item_factors[preferred] -= self.lr * grad_preferred
        self.item_factors[not_preferred] -= self.lr * grad_not_preferred

        # Return loss
        return -np.log(sigmoid + 1e-10) + (self.reg / 2) * (
            np.linalg.norm(user_vec)**2 + np.linalg.norm(preferred_vec)**2 + np.linalg.norm(not_preferred_vec)**2
        )

    def predict(self, user_id):
        user_vec = self.user_factors[user_id]
        scores = np.dot(self.item_factors, user_vec)
        return scores

    def recommend(self, user_id, known_items=set(), top_k=5):
        scores = self.predict(user_id)
        for item_id in known_items:
            scores[item_id] = -np.inf  # Mask known items
        top_items = np.argsort(-scores)[:top_k]
        return top_items.tolist()

    def save_bpr_model(bpr, user2id, item2id, path="bpr_model.pkl"):
        model_data = {
            "user_factors": bpr.user_factors,
            "item_factors": bpr.item_factors,
            "user2id": user2id,
            "item2id": item2id
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")


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

bpr = BPR(num_users=num_users, num_items=num_items, latent_dim=64, learning_rate=0.01, reg=0.01)
bpr.train(data=triplets, epochs=10)
bpr.save_bpr_model(bpr, user2id, item2id, path="bpr_model.pkl")



