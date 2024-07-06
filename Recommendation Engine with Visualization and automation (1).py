#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.data', sep='\t', names=column_names)
df = df.drop('timestamp', axis=1)
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating')

# Calculate user similarity
from sklearn.metrics.pairwise import cosine_similarity
user_item_matrix_filled = user_item_matrix.fillna(0)
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Predict ratings
def predict_ratings(user_item_matrix, user_similarity):
    pred = np.zeros(user_item_matrix.shape)
    for i in range(user_item_matrix.shape[0]):
        for j in range(user_item_matrix.shape[1]):
            if user_item_matrix.iloc[i, j] == 0:
                similar_users = user_similarity.iloc[i]
                user_ratings = user_item_matrix.iloc[:, j]
                weighted_sum = np.dot(similar_users, user_ratings)
                sum_of_weights = np.sum(np.abs(similar_users))
                pred[i, j] = weighted_sum / sum_of_weights if sum_of_weights != 0 else 0
            else:
                pred[i, j] = user_item_matrix.iloc[i, j]
    return pred

predicted_ratings = predict_ratings(user_item_matrix_filled, user_similarity_df)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Recommend items for a specific user
def recommend_items(predicted_ratings, user_id, num_recommendations):
    user_ratings = predicted_ratings.loc[user_id]
    recommended_items = user_ratings.sort_values(ascending=False)
    return recommended_items.head(num_recommendations)

# Visualization code
plt.figure(figsize=(10, 8))
sns.heatmap(user_similarity_df, cmap='coolwarm')
plt.title('User Similarity Matrix')
plt.xlabel('User ID')
plt.ylabel('User ID')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(predicted_ratings_df.iloc[:10, :10], cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Predicted Ratings (Sample)')
plt.xlabel('Item ID')
plt.ylabel('User ID')
plt.show()

def get_all_recommendations(predicted_ratings, num_recommendations):
    all_recommendations = {}
    for user_id in predicted_ratings.index:
        all_recommendations[user_id] = recommend_items(predicted_ratings, user_id, num_recommendations)
    return all_recommendations

num_recommendations = 5  # Number of recommendations per user
all_recommendations = get_all_recommendations(predicted_ratings_df, num_recommendations)
for user_id, recommendations in all_recommendations.items():
    print(f"Recommendations for User {user_id}:\n{recommendations}\n")
    
    
rows = []
for user_id, recommendations in all_recommendations.items():
    for item_id, rating in recommendations.items():
        rows.append([user_id, item_id, rating])

# Create DataFrame for CSV
df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'recommendation_rating'])

# Save to CSV
output_file = 'user_recommendations.csv'
df.to_csv(output_file, index=False)

print(f"All user recommendations saved to {output_file}")

