import pandas as pd
from shiny.ui import tags
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from collections import Counter
import socket

data = pd.read_csv(Path(__file__).parent / "data/spotify_tracks_clean_clusters.csv")
data = data.sample(frac=0.1, random_state=42)

def knn_module(data, valence=0.5, energy=0.5, n=1000):
    """
    Finds the n nearest neighbors in the dataset based on valence and energy.

    Parameters:
    - data (pd.DataFrame): The dataset containing 'valence' and 'energy'.
    - valence (float): Target valence value (0 to 1).
    - energy (float): Target energy value (0 to 1).
    - n (int): Number of nearest neighbors to return.

    Returns:
    - pd.DataFrame: Subset of the dataset with the n nearest neighbors.
    """
    # Check if required columns exist
    if not {'valence', 'energy'}.issubset(data.columns):
        raise ValueError("Data must contain 'valence' and 'energy' columns.")
    
    # Extract relevant features for KNN
    feature_data = data[["valence", "energy"]].to_numpy()

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=n, metric='euclidean')
    knn.fit(feature_data)

    # Find n nearest neighbors to user-selected values
    distances, indices = knn.kneighbors(np.array([[valence, energy]]))

    # Retrieve nearest neighbors from the original DataFrame
    nearest_neighbors = data.iloc[indices[0]]

    return nearest_neighbors.reset_index(drop=True)

def get_most_similar_tracks(df_track, df_users, user_id, top_n=200):
    """
    Recommends the most similar tracks to a given user based on their listening history.
    
    Parameters:
    df_track (DataFrame): Contains track metadata including audio features, genre, and artist.
    df_users (DataFrame): Contains user interactions with tracks.
    user_id (int or str): The ID of the user for whom recommendations are generated.
    top_n (int): Number of most similar tracks to return.

    Returns:
    DataFrame: A subset of df_track with the top_n most similar tracks.
    """
    # 1. Filter all interactions of the user
    user_data = df_users[df_users['user_id'] == user_id]

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

def inverse_popularity(df_tracks, top_n=20):
    """
    Selects a random sample of songs from df_tracks using inverse popularity weighting.

    Parameters:
    - df_tracks (pd.DataFrame): The dataset containing a 'popularity' column.
    - top_n (int): Number of songs to randomly select.

    Returns:
    - pd.DataFrame: Subset of df_tracks with top_n selected songs.
    """
    # Check if 'popularity' column exists
    if 'popularity' not in df_tracks.columns:
        raise ValueError("df_tracks must contain a 'popularity' column.")

    # Avoid division by zero: replace 0 popularity with small value
    df_tracks['popularity'] = df_tracks['popularity'].replace(0, 1e-6)

    # Compute inverse popularity weights
    weights = 1 / df_tracks['popularity']
    weights /= weights.sum()  # Normalize weights to sum to 1

    # Select top_n songs using weighted random sampling
    selected_indices = np.random.choice(df_tracks.index, size=top_n, replace=False, p=weights)
    selected_songs = df_tracks.loc[selected_indices]

    return selected_songs.reset_index(drop=True)

def is_connected():
    """Check if the system has an active internet connection."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def generate_recommended_tracks_list(tracks):
    """
    Generates a UI component displaying recommended tracks with album covers.

    Args:
        tracks (pd.DataFrame): DataFrame containing recommended tracks with album covers.

    Returns:
        shiny.ui.tags.div: A UI component displaying the recommended tracks.
    """
    if tracks.empty:
        return tags.p("No recommendations found.")

    # Check internet connection once before processing images
    internet_available = is_connected()

    items = []
    for _, row in tracks.iterrows():
        album_cover_src = row["album_cover"]

        # If no internet, use the placeholder image
        if not internet_available or album_cover_src == "album_cover_placeholder.png":
            album_cover_src = "static/album_cover_placeholder.png"

        item = tags.div(
            tags.img(src=album_cover_src, height="64px", width="64px", style="margin-right:10px;"),
            tags.div(
                tags.b(row["track_name"]),
                tags.div(f"by {row['artists']}"),
                tags.div(f"Album: {row['album_name']}"),
                style="display: inline-block; vertical-align: top;"
            ),
            style="background-color: #1e3352; padding: 10px; margin-bottom: 10px; border-radius: 5px;",
        )
        items.append(item)

    return tags.div(*items)


