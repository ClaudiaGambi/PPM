import pandas as pd
from shinywidgets import output_widget, render_widget
import plotly.express as px
from shiny import reactive, render_text
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

