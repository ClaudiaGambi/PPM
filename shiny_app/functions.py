import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from collections import Counter

data = pd.read_csv(Path(__file__).parent / "data/spotify_tracks_clean_clusters.csv")
data = data.sample(frac=0.1, random_state=42)

def knn_module(valence=0.5, energy=0.5, n=20):

    # Get user-selected values
    valence_target = valence
    energy_target = energy

    # Extract relevant features for KNN
    feature_data = data[["valence", "energy"]].to_numpy()

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=1000, metric='euclidean')
    knn.fit(feature_data)

    # Find 20 nearest neighbors to user-selected values
    distances, indices = knn.kneighbors(np.array([[valence_target, energy_target]]))

    # Use the indices to get the nearest neighbors from the original DataFrame
    nearest_neighbors = data.iloc[indices[0]]  # Indices is an array, so use [0] to get the correct row selection

    # Select 20 based on inverse popularity
    # Invert popularity for weighting (higher weight for lower popularity), ensuring all songs have a chance
    popularity = nearest_neighbors['popularity'].values
    weights = ((100 - popularity) + 1) / np.sum((100 - popularity) + 1)  # Normalize

    # Select n songs using weighted random sampling
    selected_indices = np.random.choice(nearest_neighbors.index, size=n, replace=False, p=weights)
    selected_songs = data.loc[selected_indices]

    # Return DataFrame with selected tracks
    return selected_songs.reset_index(drop=True)


def get_most_similar_tracks(df_track, df_users, user_id, top_n=10):
    # 1. Filter all interactions of the user
    user_data = df_users[df_users['user_id'] == user_id]

    print(f'STEP 1 {user_data.head()}')

    # 2. Calculate the average value for the selected audio features
    audio_features = ['danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']
    scaler = MinMaxScaler()

    user_scaled = scaler.fit_transform(user_data[audio_features])
    avg_user_features = np.mean(user_scaled, axis=0)

    # 3. Determine the most listened-to genre and artist
    most_common_genre = Counter(user_data['track_genre']).most_common(1)[0][0]
    most_common_artist = Counter(user_data['artists']).most_common(1)[0][0]

    # 4. Normalize the audio features of df_track
    track_scaled = scaler.transform(df_track[audio_features])

    # 5. Calculate the distance to each track
    distances = []
    for i, row in df_track.iterrows():
        track_features = track_scaled[i]
        audio_distance = euclidean(avg_user_features, track_features) * 0.6
        genre_distance = 0 if row['track_genre'] == most_common_genre else 1
        artist_distance = 0 if row['artists'] == most_common_artist else 1
        total_distance = audio_distance + genre_distance * 0.3 + artist_distance * 0.1
        distances.append(total_distance)

    # 6. Add the distance to the dataframe and sort by similarity
    df_track['similarity_score'] = distances
    similar_tracks = df_track.sort_values(by='similarity_score', ascending=True).head(top_n)

    return similar_tracks

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

    # Get the most similar user0
    most_similar_user = max(user_similarity, key=user_similarity.get, default=None)
    return most_similar_user


# Step 2: Recommend Songs from the Most Similar User
def recommend_from_similar_user(user_id, df, num_recommendations=5):
    similar_user = find_most_similar_user(user_id, df)
    if similar_user is None:
        return []

    user_tracks = set(df[df['user_id'] == user_id]['track_id'

    # Retrieve track details
    recommended_tracks = df[df['track_id'].isin(recommendations)][
        ['track_id', 'track_name', 'artists']].drop_duplicates()

    return recommended_tracks.head(num_recommendations).to_dict(orient='records')


# Step 3: Use FAISS for Approximate Nearest Neighbor Search on Audio Features
def build_faiss_index(df, feature_cols):
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values.astype(np.float32))

    index = faiss.IndexFlatL2(features.shape[1])  # L2 (Euclidean) distance
    index.add(features)

    # Save the FAISS index and scaler to a pickle file
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump((index, scaler), f)

    print("FAISS index saved successfully!")

    return index, scaler


def recommend_similar_tracks(track_id, df, index, scaler, feature_cols, num_recommendations=5):
    track_features = df[df['track_id'] == track_id][feature_cols].values.astype(np.float32)
    if track_features.shape[0] == 0:
        return []

    track_features = scaler.transform(track_features)
    _, indices = index.search(track_features, num_recommendations + 1)  # +1 to exclude itself in the nex line, as the top result is always the match to self

    recommended_tracks = df.iloc[indices[0][1:]][['track_id', 'track_name', 'artists']].drop_duplicates()

    return recommended_tracks.head(num_recommendations).to_dict(orient='records')


# Step 4: Combine Both Approaches

def hybrid_recommendation(user_id, df, feature_cols, num_recommendations=5, cf_threshold=3, faiss_index_path=None):
    """
    Generate hybrid recommendations using collaborative filtering (CF) and content-based filtering (CB).

    Parameters:
        user_id (int): The user ID for whom recommendations are generated.
        df (pd.DataFrame): The dataset containing track information.
        feature_cols (list): List of feature column names for content-based filtering.
        num_recommendations (int, optional): Total number of recommendations to return. Default is 5.
        cf_threshold (int, optional): Minimum number of CF-based tracks before adding CB-based tracks. Default is 3.
        faiss_index_path (str, optional): Path to a precomputed FAISS index pickle file.

    Returns:
        list: A list of recommended tracks in dictionary format.
    """

    # Step 1: Get recommendations from similar users (Collaborative Filtering)
    user_based_recommendations = recommend_from_similar_user(user_id, df, num_recommendations)

    # If CF recommendations meet or exceed the threshold, return them directly
    if len(user_based_recommendations) >= cf_threshold:
        return pd.DataFrame(user_based_recommendations).drop_duplicates().head(num_recommendations).to_dict(orient='records')

    # Step 2: Load FAISS index and scaler if provided
    if faiss_index_path:
        with open(faiss_index_path, 'rb') as f:
            index, scaler = pickle.load(f)
    else:
        # If no precomputed index is provided, build one (not recommended for large datasets)
        index, scaler = build_faiss_index(df, feature_cols)

    # Step 3: Get additional recommendations based on content similarity (only if CF is below threshold)
    content_based_recommendations = []
    for track in user_based_recommendations:
        similar_tracks = recommend_similar_tracks(track['track_id'], df, index, scaler, feature_cols,
                                                  num_recommendations=2)
        content_based_recommendations.extend(similar_tracks)

    # Step 4: Combine recommendations, ensuring uniqueness
    final_recommendations = pd.DataFrame(
        user_based_recommendations + content_based_recommendations
    ).drop_duplicates().head(num_recommendations)

    return final_recommendations.to_dict(orient='records')


# Define feature columns for FAISS
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]
