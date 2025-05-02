def recommend_group_lms(model, dataset, user_ids, num_items = 5):
#     """Recommend top movies for the group based on least misery strategy"""

#      # Get item indices
#     item_ids = np.arange(len(dataset.mapping()[2]))

#     # Predict scores for each user
#     group_scores = np.zeros(len(item_ids))

#     for user_id in user_ids:
#         user_scores = model.predict(np.repeat(user_id, len(item_ids)), item_ids)
#         group_scores += user_scores  # Sum scores across users

