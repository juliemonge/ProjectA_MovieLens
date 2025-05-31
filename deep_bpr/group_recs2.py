import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import random
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

def bpr_loss(y_true, y_pred):
    return -tf.reduce_mean(tf.math.log(tf.sigmoid(y_pred)))

def load_model_and_mappings(model_path, mappings_path):
    """Load the saved model and user/item mappings."""
    print("Model path:", model_path)
    print("Mappings path:", mappings_path)
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'bpr_loss': bpr_loss}
        )

        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            user_id_to_idx = mappings['user_id_to_idx']
            item_id_to_idx = mappings['item_id_to_idx']
            item_ids = mappings['item_ids']

        return model, user_id_to_idx, item_id_to_idx, item_ids

    except Exception as e:
        print(f"Loading failed: {e}")
        return None, None, None, None



def load_and_prepare_data(data_path):
    """Load and prepare data with validation"""
    try:
        # Load data
        data = pd.read_csv(data_path, sep=r'\s+', header=None, 
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # Clean data
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna(subset=['user_id', 'item_id', 'rating'])
        
        # Filter sparse users/items
        min_interactions = 5
        user_counts = data['user_id'].value_counts()
        item_counts = data['item_id'].value_counts()
        
        data = data[data['user_id'].isin(user_counts[user_counts >= min_interactions].index)]
        data = data[data['item_id'].isin(item_counts[item_counts >= min_interactions].index)]
        
        if data.empty:
            raise ValueError("No valid data after cleaning")
            
        return data
        
    except Exception as e:
        print(f"Error loading/preparing data: {str(e)}")
        return None

def generate_group_recommendation(model, user_ids, user_id_to_idx, item_id_to_idx, item_ids, train_data, k=5):
    """Generate group recommendations with validation"""
    try:
        # Validate inputs
        if not user_ids or not isinstance(user_ids, list):
            raise ValueError("Invalid user_ids")
            
        # Get valid user indices
        user_indices = []
        for uid in user_ids:
            if uid not in user_id_to_idx:
                print(f"Warning: User ID {uid} not in mappings")
                continue
            user_indices.append(user_id_to_idx[uid])
            
        if not user_indices:
            raise ValueError("No valid users found")
            
        # Get embeddings
        user_emb = model.get_layer('user_embedding')(tf.constant(user_indices))
        item_emb = model.get_layer('item_embedding').weights[0].numpy()
        
        # Compute group scores
        group_emb = tf.reduce_mean(user_emb, axis=0)
        scores = tf.linalg.matmul(item_emb, tf.expand_dims(group_emb, -1))
        scores = tf.squeeze(scores).numpy()
        
        # Filter seen items
        seen_items = set()
        for uid in user_ids:
            if uid in train_data['user_id'].values:
                seen_items.update(train_data[train_data['user_id'] == uid]['item_id'])
                
        # Get top unseen items
        item_scores = list(zip(item_ids, scores))
        item_scores = [(iid, score) for iid, score in item_scores if iid not in seen_items]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [iid for iid, _ in item_scores[:k]]
        
    except Exception as e:
        print(f"Error in recommendation generation: {str(e)}")
        return None

def main():
    # Configure paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_PATH = os.path.join(BASE_DIR, 'data', 'ml-100k', 'u.data')
    MODEL_PATH = 'deep_bpr/deep_bpr_model.keras'
    MAPPINGS_PATH = 'deep_bpr/model_mappings.pkl'

    model, user_id_to_idx, item_id_to_idx, item_ids = load_model_and_mappings(MODEL_PATH, MAPPINGS_PATH)
    
    if model is None:
        return
    
    # Load and prepare data
    data = load_and_prepare_data(DATA_PATH)
    if data is None:
        return
    
    # Split data (consistent random state)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Select group - ensure we have test data for evaluation
    test_users = test_data['user_id'].unique()
    if len(test_users) < 5:
        print("Not enough users with test data")
        return
        
    group_users = random.sample(list(test_users), 5)
    print(f"\nSelected group users: {group_users}")
    
    # Generate and evaluate recommendations
    recommendations = generate_group_recommendation(
        model, group_users, user_id_to_idx, item_id_to_idx, item_ids, train_data, k=5
    )
    
    if recommendations:
        print(f"\nRecommendations: {recommendations}")
        evaluation = evaluate_group_recommendation(recommendations, test_data, group_users)
        print("\nEvaluation:")
        print(f"Precision: {evaluation['precision']:.2f}")
        print(f"Recall: {evaluation['recall']:.2f}")
        print(f"Liked items: {evaluation['liked_items']}")
    else:
        print("Failed to generate recommendations")

if __name__ == "__main__":
    main()