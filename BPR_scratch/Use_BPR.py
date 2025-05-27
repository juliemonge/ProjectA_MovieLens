import pickle 

path = "BPR_scratch/bpr_model.pkl"
def load_bpr_model(path):
    with open(path, "rb") as f:
        model_data = pickle.load(f)
    
    # Recreate BPR instance with loaded weights
    bpr = BPR(
        num_users=len(model_data["user_factors"]),
        num_items=len(model_data["item_factors"]),
        latent_dim=model_data["user_factors"].shape[1]
    )
    bpr.user_factors = model_data["user_factors"]
    bpr.item_factors = model_data["item_factors"]

    return bpr, model_data["user2id"], model_data["item2id"]

bpr_loaded, user2id_loaded, item2id_loaded = load_bpr_model(path)
