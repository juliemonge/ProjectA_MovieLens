# ProjectA_MovieLens

This project implements a Bayesian Personalized Ranking (BPR) model from scratch to make movie recommendations to a random group.

## Project structure

To see our group recommendation results, you can run the BPR_group_recommendations.py file. It is set to groups of 3 members by default.

- BPR.py contain the core stucture of the model.
- BPR_train.py is the file used to start the training of the model.
- BPR_group_recommendations.py contains the strategy that we ended up using for our project, when the model had been trained.
- The generated_data folder contains the pairwise preference file used for model training

To run all three files the required libraries are the following:

- numpy
- pandas
- scipy
- tqdm
- pickle
- random
- collections
