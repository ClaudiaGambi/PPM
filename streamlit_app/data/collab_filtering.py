# given the following set of synthetic user data from a music streaming platform
#
# class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2232 entries, 0 to 2231
# Data columns (total 33 columns):
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   user_id                 2232 non-null   int64
#  1   track_id                2232 non-null   object
#  2   interaction_type        2232 non-null   object
#  3   duration_listened       2232 non-null   int64
#  4   timestamp               2232 non-null   object
#  5   rating                  148 non-null    float64
#  6   like                    208 non-null    float64
#  7   not_like                208 non-null    float64
#  8   age                     2232 non-null   int64
#  9   gender                  2085 non-null   object
#  10  location                2013 non-null   object
#  11  preferred_genre         2232 non-null   object
#  12  track_name              2232 non-null   object
#  13  album_name              2232 non-null   object
#  14  artists                 2232 non-null   object
#  15  duration_ms             2232 non-null   int64
#  16  popularity              2232 non-null   int64
#  17  track_genre             2232 non-null   object
#  18  explicit                2232 non-null   bool
#  19  danceability            2232 non-null   float64
#  20  energy                  2232 non-null   float64
#  21  key                     2232 non-null   int64
#  22  loudness                2232 non-null   float64
#  23  mode                    2232 non-null   int64
#  24  speechiness             2232 non-null   float64
#  25  acousticness            2232 non-null   float64
#  26  instrumentalness        2232 non-null   float64
#  27  liveness                2232 non-null   float64
#  28  valence                 2232 non-null   float64
#  29  tempo                   2232 non-null   float64
#  30  time_signature          2232 non-null   int64
#  31  popularity_genre        2232 non-null   float64
#  32  duration_listened_perc  2232 non-null   float64
#
# The dataset contains both implicit feedback and explicit feedback and information on the tracks listened
#
# goal: write a collaborative filtering algorithm thatfor a particular user selects the closest other user and provide 5-10 recommendations based on the following restrictions
#
# - only based on overlap in  songs listened (track-id)
# - only based on audio features using approximate nearest neighbour search (faiss)
# - a combination of the 2


import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import StandardScaler
from collections import Counter

np.random.seed(42)


# Load the dataset

# user_data = pd.read_csv("streamlit_app/data/synthetic_user_data.csv") # if line is executed manually
user_data = pd.read_csv("synthetic_user_data_test.csv")

# user_data.describe()
# user_data.info()
# user_data.head()


# Step 1: Find the Most Similar User Based on Overlapping Songs
def find_most_similar_user(user_id, df):
    user_tracks = df[df['user_id'] == user_id]['track_id'].unique()

    # Compute Jaccard similarity with other users
    user_similarity = {}
    for other_user in df['user_id'].unique():
        if other_user == user_id:
            continue
        other_tracks = df[df['user_id'] == other_user]['track_id'].unique()
        intersection = len(set(user_tracks) & set(other_tracks))
        union = len(set(user_tracks) | set(other_tracks))
        if union > 0:
            user_similarity[other_user] = intersection / union

    # Get the most similar user
    most_similar_user = max(user_similarity, key=user_similarity.get, default=None)
    return most_similar_user


# Step 2: Recommend Songs from the Most Similar User
def recommend_from_similar_user(user_id, df, num_recommendations=5):
    similar_user = find_most_similar_user(user_id, df)
    if similar_user is None:
        return []

    user_tracks = set(df[df['user_id'] == user_id]['track_id'])
    similar_user_tracks = set(df[df['user_id'] == similar_user]['track_id'])

    # Recommend tracks the similar user has listened to but the target user has not
    recommendations = list(similar_user_tracks - user_tracks)

    # Retrieve track details
    recommended_tracks = df[df['track_id'].isin(recommendations)][
        ['track_id', 'track_name', 'artists']].drop_duplicates()

    return recommended_tracks.head(num_recommendations).to_dict(orient='records')


# Step 3: Use FAISS for Approximate Nearest Neighbor Search on Audio Features
def build_faiss_index(df, feature_cols):
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values.astype(np.float32))

    index = faiss.IndexFlatL2(features.shape[1])  # L2 distance
    index.add(features)

    return index, scaler


def recommend_similar_tracks(track_id, df, index, scaler, feature_cols, num_recommendations=5):
    track_features = df[df['track_id'] == track_id][feature_cols].values.astype(np.float32)
    if track_features.shape[0] == 0:
        return []

    track_features = scaler.transform(track_features)
    _, indices = index.search(track_features, num_recommendations + 1)  # +1 to exclude itself

    recommended_tracks = df.iloc[indices[0][1:]][['track_id', 'track_name', 'artists']].drop_duplicates()

    return recommended_tracks.head(num_recommendations).to_dict(orient='records')


# Step 4: Combine Both Approaches
def hybrid_recommendation(user_id, df, feature_cols, num_recommendations=5):
    # Get recommendations from similar user
    user_based_recommendations = recommend_from_similar_user(user_id, df, num_recommendations)

    # Build FAISS index
    index, scaler = build_faiss_index(df, feature_cols)

    # Get additional recommendations based on audio similarity
    content_based_recommendations = []
    for track in user_based_recommendations:
        similar_tracks = recommend_similar_tracks(track['track_id'], df, index, scaler, feature_cols,
                                                  num_recommendations=2)
        content_based_recommendations.extend(similar_tracks)

    # Combine and return unique recommendations
    final_recommendations = pd.DataFrame(
        user_based_recommendations + content_based_recommendations).drop_duplicates().head(num_recommendations)

    return final_recommendations.to_dict(orient='records')


# Define feature columns for FAISS
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Example Usage

user_id = 1

print(f'most similar user: {find_most_similar_user(user_id, user_data)}')
print(f'recommended from sim. user: {recommend_from_similar_user(user_id, user_data, num_recommendations=5)}')
# idx = build_faiss_index(user_data, audio_features)
# print(f'recommended similar tracks: {recommend_similar_tracks("1", user_data, idx[0], idx[1], audio_features, num_recommendations=5)})

recommendations = hybrid_recommendation(user_id, user_data, audio_features, num_recommendations=5)
print("Recommended Tracks:", recommendations)