# Pick 5 random users from your dataset
unique_users = df["User_ID"].unique()
group_user_ids = random.sample(list(unique_users), 5)