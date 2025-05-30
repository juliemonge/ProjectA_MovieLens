import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm import tqdm
import pickle

class BPR:
    def __init__(self, num_users, num_items, latent_dim=64, learning_rate=0.01, reg=0.025, k=10, epochs=10, max_users=1000):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.reg = reg
        self.k = k
        self.epochs = epochs
        self.max_users = max_users

        self.user_factors = np.random.normal(0, 0.01, (num_users, latent_dim))
        self.item_factors = np.random.normal(0, 0.01, (num_items, latent_dim))

    def train(self, data, epochs=None, eval_df=None, user2id=None, item2id=None):
        best_precision = 0
        patience = 3
        no_improve_epochs = 0
        if epochs is None:
            epochs = self.epochs

        for epoch in range(epochs):
            np.random.shuffle(data)
            epoch_loss = 0
            for entry in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}"):
                if len(entry) == 4:
                    user, preferred, not_preferred, weight = map(float, entry)
                else:
                    user, preferred, not_preferred = map(int, entry)
                    weight = 1.0

                loss = self.update(int(user), int(preferred), int(not_preferred), float(weight))
                epoch_loss += loss

            print(f"Epoch {epoch+1} completed. Total Loss: {epoch_loss:.4f}, Avg Loss: {epoch_loss / len(data):.6f}")

            if eval_df is not None:
                metrics = self.evaluate_all_metrics(eval_df, user2id, item2id)
                print(f"Evaluation @ {self.k}: "
                      f"Precision={metrics['Precision@K']:.4f}, "
                      f"Recall={metrics['Recall@K']:.4f}, "
                      f"HitRate={metrics['HitRate@K']:.4f}, "
                      f"NDCG={metrics['NDCG@K']:.4f}")

                current_precision = metrics["Precision@K"]
                if current_precision > best_precision:
                    best_precision = current_precision
                    no_improve_epochs = 0
                    self.save_bpr_model(user2id, item2id, path="best_model_weighted.pkl")
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
                    break

    def update(self, user, preferred, not_preferred, weight=1.0):
        user_vec = self.user_factors[user]
        preferred_vec = self.item_factors[preferred]
        not_preferred_vec = self.item_factors[not_preferred]

        dot_product = np.dot(user_vec, preferred_vec - not_preferred_vec)
        dot_product = np.clip(dot_product, -35, 35)
        sigmoid = expit(dot_product)

        loss = -weight * np.log(sigmoid + 1e-10) + (self.reg / 2) * (
            np.dot(user_vec, user_vec) +
            np.dot(preferred_vec, preferred_vec) +
            np.dot(not_preferred_vec, not_preferred_vec)
        )
        loss = min(loss, 50)

        grad_common = weight * (1 - sigmoid)
        grad_user = grad_common * (preferred_vec - not_preferred_vec) + self.reg * user_vec
        grad_preferred = grad_common * user_vec + self.reg * preferred_vec
        grad_not_preferred = -grad_common * user_vec + self.reg * not_preferred_vec

        max_grad = 5.0
        grad_user = np.clip(grad_user, -max_grad, max_grad)
        grad_preferred = np.clip(grad_preferred, -max_grad, max_grad)
        grad_not_preferred = np.clip(grad_not_preferred, -max_grad, max_grad)

        self.user_factors[user] -= self.lr * grad_user
        self.item_factors[preferred] -= self.lr * grad_preferred
        self.item_factors[not_preferred] -= self.lr * grad_not_preferred

        self.user_factors[user] /= np.linalg.norm(self.user_factors[user]) + 1e-10
        self.item_factors[preferred] /= np.linalg.norm(self.item_factors[preferred]) + 1e-10
        self.item_factors[not_preferred] /= np.linalg.norm(self.item_factors[not_preferred]) + 1e-10

        if not np.isfinite(loss):
            print(f"Warning: Non-finite loss! dot={dot_product:.2f}, sigmoid={sigmoid:.6f}")

        return loss

    def predict(self, user_id):
        user_vec = self.user_factors[user_id]
        scores = np.dot(self.item_factors, user_vec)
        return scores

    def recommend(self, user_id, known_items=set(), top_k=10):
        scores = self.predict(user_id)
        for item_id in known_items:
            scores[item_id] = -np.inf
        top_items = np.argsort(-scores)[:top_k]
        return top_items.tolist()

    def save_bpr_model(self, user2id, item2id, path="bpr_model_weighted.pkl"):
        model_data = {
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user2id": user2id,
            "item2id": item2id
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")

    def evaluate_all_metrics(self, df, user2id, item2id, k=10, max_users=1000):
        from collections import defaultdict
        import random

        true_items_by_user = defaultdict(set)
        for _, row in df.iterrows():
            u = user2id.get(row["User_ID"])
            i = item2id.get(row["Preferred"])
            if u is not None and i is not None:
                true_items_by_user[u].add(i)

        all_user_ids = list(true_items_by_user.keys())
        if len(all_user_ids) > max_users:
            eval_user_ids = random.sample(all_user_ids, max_users)
        else:
            eval_user_ids = all_user_ids

        precisions, recalls, hits, ndcgs = [], [], [], []

        for user_id in eval_user_ids:
            true_items = true_items_by_user[user_id]
            scores = self.predict(user_id)

            top_k_idx = np.argpartition(-scores, k)[:k]
            top_k_scores = scores[top_k_idx]
            top_k_sorted = top_k_idx[np.argsort(-top_k_scores)]
            top_k_set = set(top_k_sorted)

            num_hits = len(top_k_set & true_items)
            precisions.append(num_hits / k)
            recalls.append(num_hits / len(true_items))
            hits.append(1 if num_hits > 0 else 0)

            dcg = 0
            for rank, item in enumerate(top_k_sorted, start=1):
                if item in true_items:
                    dcg += 1 / np.log2(rank + 1)
            idcg = sum(1 / np.log2(r + 1) for r in range(1, min(len(true_items), k) + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0)

        return {
            "Precision@K": np.mean(precisions),
            "Recall@K": np.mean(recalls),
            "HitRate@K": np.mean(hits),
            "NDCG@K": np.mean(ndcgs),
        }
