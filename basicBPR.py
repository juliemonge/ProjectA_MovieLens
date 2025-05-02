import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset

# Load your data
df = pd.read_csv("/Users/juliemongehallan/Documents/Datatek/Data_Science/Vår 2025/Data Mining/ProjectA_MovieLens/generated_data/user_pairwise_preferences.csv", header=None, names=["user", "pos_item", "neg_item"])

# Prepare dataset
dataset = Dataset()
dataset.fit(df['user'], pd.concat([df['pos_item'], df['neg_item']]))

(interactions, _) = dataset.build_interactions(((row['user'], row['pos_item']) for _, row in df.iterrows()))

# Train model
model = LightFM(loss='bpr')
model.fit(interactions, epochs=5, num_threads=2)
