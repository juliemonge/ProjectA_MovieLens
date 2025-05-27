import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pickle
from scipy.special import expit

class BPR:
    def __init__(self, num_users, num_items, latent_dim=64, learning_rate=0.005, reg=0.1):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.reg = reg

        # Initialize user and item embeddings
        self.user_factors = np.random.normal(0, 0.01, (num_users, latent_dim))
        self.item_factors = np.random.normal(0, 0.01, (num_items, latent_dim))

    def train(self, data, epochs=5, test_df=None, user2id=None, item2id=None):
        for epoch in range(epochs):
            np.random.shuffle(data)
            epoch_loss = 0
            for user, preferred, not_preferred in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}"):
                user, preferred, not_preferred = int(user), int(preferred), int(not_preferred)
                loss = self.update(user, preferred, not_preferred)
                epoch_loss += loss
            print(f"Epoch {epoch+1} completed. Total Loss: {epoch_loss:.4f}, Avg Loss: {epoch_loss / len(data):.6f}")

            if test_df is not None and user2id is not None and item2id is not None:
                precision = self.evaluate_precision_at_k(test_df, user2id, item2id, k=5)
                print(f"Precision@5 after Epoch {epoch+1}: {precision:.4f}")

    def update(self, user, preferred, not_preferred):
        # 1. Get current embeddings
        user_vec = self.user_factors[user]
        preferred_vec = self.item_factors[preferred]
        not_preferred_vec = self.item_factors[not_preferred]

        # 2. Compute predicted score difference (clip to avoid extreme exp)
        dot_product = np.dot(user_vec, preferred_vec - not_preferred_vec)
        dot_product = np.clip(dot_product, -35, 35)

        # 3. Compute sigmoid (numerically stable)
        sigmoid = expit(dot_product)

        # 4. Compute BPR loss BEFORE updates
        loss = -np.log(sigmoid + 1e-10) + (self.reg / 2) * (
            np.linalg.norm(user_vec)**2 +
            np.linalg.norm(preferred_vec)**2 +
            np.linalg.norm(not_preferred_vec)**2
        )

        # 5. Cap loss to avoid extreme accumulations
        loss = min(loss, 50)

        # 6. Compute gradients
        grad_user = (1 - sigmoid) * (preferred_vec - not_preferred_vec) + self.reg * user_vec
        grad_preferred = (1 - sigmoid) * user_vec + self.reg * preferred_vec
        grad_not_preferred = -(1 - sigmoid) * user_vec + self.reg * not_preferred_vec

        # 7. Clip gradients to prevent large updates
        max_grad = 5.0
        grad_user = np.clip(grad_user, -max_grad, max_grad)
        grad_preferred = np.clip(grad_preferred, -max_grad, max_grad)
        grad_not_preferred = np.clip(grad_not_preferred, -max_grad, max_grad)

        # 8. Update vectors
        self.user_factors[user] -= self.lr * grad_user
        self.item_factors[preferred] -= self.lr * grad_preferred
        self.item_factors[not_preferred] -= self.lr * grad_not_preferred

        # 9. Normalize embeddings to prevent vector explosion
        self.user_factors[user] /= np.linalg.norm(self.user_factors[user]) + 1e-10
        self.item_factors[preferred] /= np.linalg.norm(self.item_factors[preferred]) + 1e-10
        self.item_factors[not_preferred] /= np.linalg.norm(self.item_factors[not_preferred]) + 1e-10

        # 10. Check for numerical errors (optional)
        if not np.isfinite(loss):
            print(f"Warning: Non-finite loss! dot={dot_product:.2f}, sigmoid={sigmoid:.6f}")

        return loss



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

    def save_bpr_model(self, user2id, item2id, path="bpr_model.pkl"):
        model_data = {
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user2id": user2id,
            "item2id": item2id
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")

    def evaluate_precision_at_k(self, test_df, user2id, item2id, k=5):
        hits = 0
        total = 0

        for _, row in test_df.iterrows():
            user_orig = row["User_ID"]
            item_orig = row["Preferred"]

            user_id = user2id.get(user_orig)
            item_id = item2id.get(item_orig)

            if user_id is None or item_id is None:
                continue

            scores = self.predict(user_id)
            top_k = np.argsort(-scores)[:k]

            if item_id in top_k:
                hits += 1
            total += 1

        precision = hits / total if total > 0 else 0
        return precision



