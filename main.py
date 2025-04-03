import kaggle
import pandas as pd

# Download the dataset
kaggle.api.dataset_download_files('prajitdatta/movielens-100k-dataset', path='data/', unzip=True)


# Load the dataset (adjust filename based on the extracted files)
dataframe = pd.read_csv('data/ml-100k/u.data', delimiter="\t", header=None, engine="python")