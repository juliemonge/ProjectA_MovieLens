import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import random
from collections import defaultdict

def bpr_loss(y_true, y_pred):
    """Define your custom loss function again for loading"""
    return -tf.reduce_mean(tf.math.log(tf.sigmoid(y_pred)))

def load_model_and_mappings(model_path):
    """Load the saved model and mappings"""
    try:
        # Load the saved model with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'bpr_loss': bpr_loss}  # Add your custom loss here
        )
        
        # Load the mappings
        with open('deep_bpr/model_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            user_id_to_idx = mappings['user_id_to_idx']
            item_id_to_idx = mappings['item_id_to_idx']
            item_ids = mappings['item_ids']
            
        return model, user_id_to_idx, item_id_to_idx, item_ids
    except Exception as e:
        print(f"Error loading model or mappings: {e}")
        return None, None, None, None

def load_data(filepath):
    """Load the dataset with correct format"""
    try:
        data = pd.read_csv(filepath, sep=r'\s+', header=None, 
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(data):
    """Clean and prepare the data"""
    if data is None or data.empty:
        raise ValueError("Input data is empty")
    
    # Convert all columns to numeric
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna(subset=['user_id', 'item_id', 'rating'])
    
    # Filter sparse users and items
    min_interactions = 5
    user_counts = data['user_id'].value_counts()
    item_counts = data['item_id'].value_counts()
    
    data = data[data['user_id'].isin(user_counts[user_counts >= min_interactions].index)]
    data = data[data['item_id'].isin(item_counts[item_counts >= min_interactions].index)]
    
    return data

def generate_group_recommendation(model, user_ids, user_id_to_idx, item_id_to_idx, item_ids, train_data, k=5):
    """Generate recommendations for a group of users"""
    # Get user indices
    user_indices = [user_id_to_idx[uid] for uid in user_ids if uid in user_id_to_idx]
    
    if not user_indices:
        raise ValueError("None of the provided user IDs exist in the dataset")
    
    # Get all item indices
    all_item_indices = np.array(list(item_id_to_idx.values()))
    
    # Get embeddings
    user_embedding_layer = model.get_layer('user_embedding')
    item_embedding_layer = model.get_layer('item_embedding')
    
    # Get group embedding (average of user embeddings)
    user_embeddings = user_embedding_layer(np.array(user_indices))
    group_embedding = tf.reduce_mean(user_embeddings, axis=0)
    
    # Get item embeddings
    item_embeddings = item_embedding_layer(all_item_indices)
    
    # Compute scores
    scores = tf.reduce_sum(group_embedding * item_embeddings, axis=-1).numpy()
    
    # Filter out items already seen by any group member
    group_train_items = set()
    for uid in user_ids:
        if uid in train_data['user_id'].values:
            group_train_items.update(train_data[train_data['user_id'] == uid]['item_id'].unique())
    
    candidate_mask = [item_id not in group_train_items for item_id in item_ids]
    candidate_indices = all_item_indices[candidate_mask]
    candidate_scores = scores[candidate_mask]
    
    # Get top k items
    top_k_indices = np.argsort(-candidate_scores)[:k]
    top_k_item_ids = [item_ids[idx] for idx in candidate_indices[top_k_indices]]
    
    return top_k_item_ids

def evaluate_group_recommendation(recommended_items, test_data, user_ids):
    """Evaluate the group recommendation"""
    # Get items liked by any group member in test set (rating >= 4)
    liked_items = set()
    for uid in user_ids:
        user_liked = test_data[(test_data['user_id'] == uid) & (test_data['rating'] >= 4)]['item_id'].unique()
        liked_items.update(user_liked)
    
    # Calculate metrics
    hit = any(item in liked_items for item in recommended_items)
    precision = sum(1 for item in recommended_items if item in liked_items) / len(recommended_items)
    
    return {
        'group_size': len(user_ids),
        'recommended_items': recommended_items,
        'liked_items': list(liked_items),
        'hit': hit,
        'precision': precision,
        'recall': sum(1 for item in recommended_items if item in liked_items) / len(liked_items) if liked_items else 0,
        'num_liked_items': len(liked_items)
    }

def main():
    # Load model and mappings
    model_path = 'deep_bpr/deep_bpr_model.h5'  # Change to your model path
    model, user_id_to_idx, item_id_to_idx, item_ids = load_model_and_mappings(model_path)
    
    if model is None:
        return
    
    # Load and clean data
    data_path = 'data/ml-100k/u.data'  # Change to your data path
    data = load_data(data_path)
    data = clean_data(data)
    
    if data is None or data.empty:
        print("No valid data available")
        return
    
    # Split data (consistent with training)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Select 5 random users for group recommendation
    all_user_ids = list(user_id_to_idx.keys())
    group_users = random.sample(all_user_ids, 5)
    print(f"\nSelected group users: {group_users}")
    
    # Generate group recommendation
    try:
        recommended_items = generate_group_recommendation(
            model, group_users, user_id_to_idx, item_id_to_idx, item_ids, train_data, k=5
        )
        print(f"\nGroup recommendations: {recommended_items}")
        
        # Evaluate the recommendation
        evaluation = evaluate_group_recommendation(recommended_items, test_data, group_users)
        
        print("\nEvaluation Results:")
        print(f"Number of items liked by group members: {evaluation['num_liked_items']}")
        print(f"Recommended items: {evaluation['recommended_items']}")
        print(f"Liked items: {evaluation['liked_items']}")
        print(f"At least one hit: {evaluation['hit']}")
        print(f"Precision: {evaluation['precision']:.2f}")
        print(f"Recall: {evaluation['recall']:.2f}")
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")

if __name__ == "__main__":
    main()