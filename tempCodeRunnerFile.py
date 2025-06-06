import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(filepath):
    """Load the dataset with correct format"""
    try:
        # Read space-separated file with no header
        data = pd.read_csv(filepath, sep='\s+', header=None, 
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        print("Data loaded successfully. First few rows:")
        print(data.head())
        print("\nData description:")
        print(data.describe())
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(data):
    """Clean and prepare the data"""
    if data is None or data.empty:
        raise ValueError("Input data is empty")
    
    print("\nInitial data shape:", data.shape)
    print("NaN values before cleaning:")
    print(data.isna().sum())
    
    # Convert all columns to numeric, coercing errors
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with any NaN values in critical columns
    data = data.dropna(subset=['user_id', 'item_id', 'rating'])
    
    print("\nData shape after cleaning:", data.shape)
    
    if data.empty:
        raise ValueError("No valid data remaining after cleaning")
    
    # Filter out users and items with too few interactions
    min_interactions = 5
    user_counts = data['user_id'].value_counts()
    item_counts = data['item_id'].value_counts()
    
    data = data[data['user_id'].isin(user_counts[user_counts >= min_interactions].index)]
    data = data[data['item_id'].isin(item_counts[item_counts >= min_interactions].index)]
    
    print("\nData shape after filtering sparse users/items:", data.shape)
    
    return data

# Load your dataset
filepath = 'data/ml-100k/u.data'  # Change this to your actual file path
data = load_data(filepath)

if data is not None:
    try:
        data = clean_data(data)
        
        # Only proceed if we have data
        if not data.empty:
            # Preprocessing
            user_ids = data['user_id'].unique()
            item_ids = data['item_id'].unique()

            user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
            item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

            data['user_idx'] = data['user_id'].map(user_id_to_idx)
            data['item_idx'] = data['item_id'].map(item_id_to_idx)

            # Normalize ratings to [0, 1] range
            data['normalized_rating'] = (data['rating'] - data['rating'].min()) / \
                                      (data['rating'].max() - data['rating'].min())

            # Split data into train and test
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            print("\nTraining data size:", len(train_data))
            print("Test data size:", len(test_data))
            
            # Rest of your BPR implementation would go here...
            # [Include the generate_pairs function and BPR model code from earlier]
            
        else:
            print("No valid data available after cleaning.")
            
    except Exception as e:
        print(f"Error during data cleaning: {e}")
else:
    print("Failed to load data.")

# Normalize ratings to [0, 1] range safely
def safe_normalize_ratings(data):
    scaler = MinMaxScaler()
    ratings = data['rating'].values.reshape(-1, 1)
    
    # Check if all ratings are identical (would cause division by zero)
    if np.all(ratings == ratings[0]):
        print("Warning: All ratings are identical. Using uniform scores.")
        data['normalized_rating'] = 0.5  # Midpoint for all
    else:
        data['normalized_rating'] = scaler.fit_transform(ratings).flatten()
    
    return data

data = safe_normalize_ratings(data)

# Split data into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Generate pairwise preferences for BPR with robust handling
def generate_pairs(data, num_negatives=1, rating_weighted=False):
    """
    Generate positive and negative item pairs for each user with robust NaN handling
    """
    user_items = defaultdict(list)
    item_users = defaultdict(list)
    
    # Get all items interacted by each user and users for each item
    for _, row in data.iterrows():
        # Ensure rating is not NaN
        if pd.notna(row['normalized_rating']):
            user_items[row['user_idx']].append((row['item_idx'], row['normalized_rating']))
            item_users[row['item_idx']].append(row['user_idx'])
    
    pairs = []
    all_items = set(data['item_idx'].unique())
    
    # Calculate item popularity for negative sampling
    item_popularity = {item: len(users) for item, users in item_users.items()}
    popular_items = np.array(list(item_popularity.keys()))
    popularity_weights = np.array(list(item_popularity.values()))
    
    # Invert popularity for sampling (less popular items more likely)
    inv_popularity_weights = 1.0 / (popularity_weights + 1e-6)  # Add small epsilon to avoid division by zero
    inv_popularity_weights = inv_popularity_weights / inv_popularity_weights.sum()
    
    for user, pos_items in user_items.items():
        for pos_item, rating in pos_items:
            # Ensure rating is finite
            if not np.isfinite(rating):
                continue
                
            # Sample negative items not interacted by the user
            user_pos_items = set([item for item, _ in pos_items])
            neg_candidates = list(all_items - user_pos_items)
            
            if rating_weighted:
                # Safely calculate number of negatives
                try:
                    num_neg = int(num_negatives * (1 + float(rating) * 2))
                    num_neg = max(1, num_neg)  # Ensure at least 1 negative
                except (ValueError, TypeError):
                    num_neg = num_negatives
            else:
                num_neg = num_negatives
                
            if len(neg_candidates) > 0:
                if rating_weighted and len(neg_candidates) > 1:
                    try:
                        # Get weights for candidate items
                        candidate_indices = [item_id_to_idx[item] for item in neg_candidates]
                        candidate_weights = inv_popularity_weights[candidate_indices]
                        candidate_weights = candidate_weights / candidate_weights.sum()
                        
                        neg_items = np.random.choice(
                            neg_candidates,
                            size=min(num_neg, len(neg_candidates)),
                            replace=False,
                            p=candidate_weights
                        )
                    except:
                        # Fallback to uniform sampling if weighted sampling fails
                        neg_items = np.random.choice(
                            neg_candidates,
                            size=min(num_neg, len(neg_candidates)),
                            replace=False
                        )
                else:
                    neg_items = np.random.choice(
                        neg_candidates,
                        size=min(num_neg, len(neg_candidates)),
                        replace=False
                    )
                
                for neg_item in neg_items:
                    pairs.append((user, pos_item, neg_item))
    
    return np.array(pairs)

# Generate training pairs with safe defaults
try:
    train_pairs = generate_pairs(train_data, num_negatives=3, rating_weighted=True)
except Exception as e:
    print(f"Error in generate_pairs: {e}")
    print("Falling back to unweighted sampling")
    train_pairs = generate_pairs(train_data, num_negatives=3, rating_weighted=False)

print(f"Generated {len(train_pairs)} training pairs")

# BPR Model Implementation
class BPRModel:
    def __init__(self, num_users, num_items, latent_dim=20):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        
    def build_model(self):
        # Inputs
        user_input = Input(shape=(1,), name='user_input')
        pos_item_input = Input(shape=(1,), name='pos_item_input')
        neg_item_input = Input(shape=(1,), name='neg_item_input')
        
        # Embedding layers
        user_embedding = Embedding(
            self.num_users, 
            self.latent_dim, 
            embeddings_initializer='he_normal',
            name='user_embedding'
        )
        item_embedding = Embedding(
            self.num_items, 
            self.latent_dim, 
            embeddings_initializer='he_normal',
            name='item_embedding'
        )
        
        # Get embeddings
        u = user_embedding(user_input)
        i_pos = item_embedding(pos_item_input)
        i_neg = item_embedding(neg_item_input)
        
        # Flatten embeddings
        u = Flatten()(u)
        i_pos = Flatten()(i_pos)
        i_neg = Flatten()(i_neg)
        
        # Compute scores
        pos_score = Dot(axes=1)([u, i_pos])
        neg_score = Dot(axes=1)([u, i_neg])
        
        # Subtract scores
        diff = pos_score - neg_score
        
        # Create model
        model = Model(
            inputs=[user_input, pos_item_input, neg_item_input],
            outputs=diff
        )
        
        return model

# Initialize model
num_users = len(user_ids)
num_items = len(item_ids)
latent_dim = 20

bpr = BPRModel(num_users, num_items, latent_dim)
model = bpr.build_model()

# Custom loss function for BPR
def bpr_loss(y_true, y_pred):
    return -tf.reduce_mean(tf.math.log(tf.sigmoid(y_pred)))

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss=bpr_loss)

# Prepare training data
users = train_pairs[:, 0]
pos_items = train_pairs[:, 1]
neg_items = train_pairs[:, 2]

# Train the model
history = model.fit(
    [users, pos_items, neg_items],
    np.ones(len(users)),  # Dummy output since we're using the pairs directly
    batch_size=256,
    epochs=30,
    verbose=1,
    validation_split=0.1
)

# Evaluation Functions
def evaluate_model(model, test_data, train_data, user_id_to_idx, item_id_to_idx, k=10):
    """
    Evaluate the model using various metrics:
    - AUC
    - Precision@K
    - Recall@K
    - NDCG@K
    """
    # Get embeddings
    user_embedding_layer = model.get_layer('user_embedding')
    item_embedding_layer = model.get_layer('item_embedding')
    
    user_embeddings = user_embedding_layer.weights[0].numpy()
    item_embeddings = item_embedding_layer.weights[0].numpy()
    
    # Prepare test data
    test_users = test_data['user_idx'].unique()
    
    # Initialize metrics
    auc_scores = []
    precisions = []
    recalls = []
    ndcgs = []
    
    # For each test user
    for user_idx in test_users:
        # Get items in training set (to exclude from recommendations)
        train_items = set(train_data[train_data['user_idx'] == user_idx]['item_idx'])
        
        # Get positive items in test set
        pos_test_items = set(test_data[test_data['user_idx'] == user_idx]['item_idx'])
        
        # Skip users without positive items in test set
        if len(pos_test_items) == 0:
            continue
            
        # Get all candidate items (not in training)
        all_items = set(item_id_to_idx.values())
        candidate_items = list(all_items - train_items)
        
        # Get user embedding
        user_embedding = user_embeddings[user_idx]
        
        # Compute scores for all candidate items
        item_scores = np.dot(item_embeddings[candidate_items], user_embedding)
        
        # Create labels (1 for positive test items, 0 otherwise)
        labels = np.zeros(len(candidate_items))
        for i, item_idx in enumerate(candidate_items):
            if item_idx in pos_test_items:
                labels[i] = 1
        
        # Calculate AUC
        if len(set(labels)) > 1:  # Need both positive and negative examples
            auc = roc_auc_score(labels, item_scores)
            auc_scores.append(auc)
        
        # Calculate Precision@K, Recall@K, NDCG@K
        top_k_indices = np.argsort(-item_scores)[:k]
        top_k_items = [candidate_items[i] for i in top_k_indices]
        
        # Precision@K: fraction of recommended items that are relevant
        relevant_count = sum(1 for item in top_k_items if item in pos_test_items)
        precision = relevant_count / k
        precisions.append(precision)
        
        # Recall@K: fraction of relevant items that are recommended
        recall = relevant_count / len(pos_test_items) if len(pos_test_items) > 0 else 0
        recalls.append(recall)
        
        # NDCG@K
        dcg = 0
        for i, item in enumerate(top_k_items):
            if item in pos_test_items:
                dcg += 1 / np.log2(i + 2)  # +2 because indexing starts at 0
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(pos_test_items))))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    # Aggregate metrics
    metrics = {
        'AUC': np.mean(auc_scores),
        'Precision@K': np.mean(precisions),
        'Recall@K': np.mean(recalls),
        'NDCG@K': np.mean(ndcgs)
    }
    
    return metrics

# Evaluate the model
eval_metrics = evaluate_model(model, test_data, train_data, user_id_to_idx, item_id_to_idx, k=10)
print("\nEvaluation Metrics:")
for metric, value in eval_metrics.items():
    print(f"{metric}: {value:.4f}")

