import pandas as pd
import numpy as np
from shiny.ui import tags
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import faiss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import euclidean
from collections import Counter
import socket
import random


# functions to import track and user data in app.py
def process_tracks(data):
    print(f"Loading {len(data)} tracks")
    return data.copy()

def process_users(data):
    print(f"Loading {len(data)} user interactions")
    return data.copy()

# track data loading

data = pd.read_csv(Path(__file__).parent / "data/spotify_tracks.csv")
data = data.sample(frac=0.3, random_state=42)

tracks_data = process_tracks(data) # for use in app.py
# user data loading

user_data = pd.read_csv(Path(__file__).parent / "data/synthetic_user_data.csv")

user_data = process_users(user_data)  # for use in app.py


# audio_features = ['danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

scaler = MinMaxScaler()
track_scaled = scaler.fit_transform(data[audio_features])


def knn_module(data, valence=0.5, energy=0.5, max_knn=500):
    """
    Finds the nearest neighbors in the dataset based on valence and energy.

    Parameters:
    - data (pd.DataFrame): The filtered dataset containing 'valence' and 'energy'.
    - valence (float): Target valence value (0 to 1).
    - energy (float): Target energy value (0 to 1).
    - max_knn (int): Maximum number of nearest neighbors to return.

    Returns:
    - pd.DataFrame: Subset of the dataset with the nearest neighbors.
    """
    # Check if required columns exist
    if not {'valence', 'energy'}.issubset(data.columns):
        raise ValueError("Data must contain 'valence' and 'energy' columns.")

    # Extract relevant features for KNN
    feature_data = data[["valence", "energy"]].to_numpy()

    n_neighbors = min(len(data), max_knn) # Ensure n_neighbors does not exceed available data

    if n_neighbors < 1:
        print("WARNING: No data available for KNN search!", flush=True)
        return pd.DataFrame()  # Return empty DataFrame if no data is available

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(feature_data)

    # Find nearest neighbors to user-selected values
    distances, indices = knn.kneighbors(np.array([[valence, energy]]))

    # Retrieve nearest neighbors from the original DataFrame
    nearest_neighbors = data.iloc[indices[0]]
    print(f"knn: Returning {len(nearest_neighbors)} nearest neighbors", flush=True)

    return nearest_neighbors.reset_index(drop=True)

def filter_christmas_songs(tracks_df):
    christmas_keywords = ["christmas", "santa", "saint", "frosty", "snowman"]

    # Zet alles om naar kleine letters voor case-insensitive filtering
    mask = tracks_df["track_name"].str.lower().str.contains('|'.join(christmas_keywords), na=False) | \
           tracks_df["album_name"].str.lower().str.contains('|'.join(christmas_keywords), na=False)

    # Houd alleen niet-kerstliedjes over
    filtered_df = tracks_df[~mask]

    # Reset de index en verwijder de oude indexkolom
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df

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

    user_scaled = scaler.fit_transform(user_data[audio_features])
    avg_user_features = np.mean(user_scaled, axis=0)

    # 3. Determine the most listened-to genre and artist
    most_common_genre = Counter(user_data['track_genre']).most_common(1)[0][0]
    print(f'most common genre: {most_common_genre}', flush=True)
    most_common_artist = Counter(user_data['artists']).most_common(1)[0][0]
    print(f'most common artist: {most_common_artist}\n', flush=True)

    # 4. Calculate the distance to each track
    distances = []
    for i, row in df_track.iterrows():
        track_features = track_scaled[i]
        audio_distance = euclidean(avg_user_features, track_features)
        # print(f'audio distance: {audio_distance}\n', flush=True)
        genre_distance = 0 if row['track_genre'] == most_common_genre else 1
        artist_distance = 0 if row['artists'] == most_common_artist else 1
        total_distance = audio_distance * 0.5 + genre_distance * 0.3 + artist_distance * 0.2

        distances.append(total_distance)

    print(f'avg total distance: {np.mean(distances)}\n', flush=True)

    # 6. Add the distance to the dataframe and sort by similarity
    df_track['similarity_score'] = distances
    similar_tracks = df_track.sort_values(by='similarity_score', ascending=True).head(top_n)
    print(f'sim track: returning {len(similar_tracks)} similar tracks', flush=True)

    return similar_tracks


def inverse_popularity(df_tracks, top_n):
    """
    Selects top_n tracks using weighted random sampling based on inverse popularity.

    Parameters:
    - df_tracks (pd.DataFrame): DataFrame containing tracks.
    - top_n (int): Number of tracks to select.

    Returns:
    - pd.DataFrame: Subset of df_tracks with selected tracks.
    """
    if df_tracks.empty:
        print("WARNING: No tracks available for inverse popularity selection!", flush=True)
        return df_tracks  # Return empty DataFrame

    # Normalize popularity weights
    weights = df_tracks['popularity_weight'] / df_tracks['popularity_weight'].sum()

    #  Ensure `top_n` does not exceed available data
    top_n = min(top_n, len(df_tracks))

    # Select tracks using weighted random sampling
    selected_indices = np.random.choice(df_tracks.index, size=top_n, replace=False, p=weights)

    return df_tracks.loc[selected_indices]

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

    # print("DEBUG: Available columns in tracks DataFrame:", tracks.columns, flush=True)

    # Check internet connection once before processing images
    internet_available = is_connected()

    items = []
    for _, row in tracks.iterrows():
        album_cover_src = row["album_cover"]

        # If no internet, use the placeholder image
        if not internet_available or album_cover_src == "album_cover_placeholder.png":
            album_cover_src = "album_cover_placeholder.png"
            # album_cover_src = str(Path(__file__).parent / "/static/album_cover_placeholder.png")

        item = tags.div(
            tags.img(src=album_cover_src, height="64px", width="64px", style="margin-right:10px;"),
            tags.div(
                tags.b(row["track_name"]),
                tags.div(f"by {row['artists']}"),
                tags.div(f"Album: {row['album_name']}"),
                tags.div(f"genre: {row['track_genre']}"),
                style="display: inline-block; vertical-align: top;"
            ),
            style="background-color: #1e3352; padding: 10px; margin-bottom: 10px; border-radius: 5px;",
        )
        items.append(item)

    return tags.div(*items)

def buddy_recommendations(user_id, df, num_recommendations=5):
    # Get unique tracks for the user
    user_tracks = set(df[df['user_id'] == user_id]['track_id'].unique())

    # Compute Jaccard similarity with other users
    user_similarity = {}
    for other_user in df['user_id'].unique():
        if other_user == user_id:
            continue
        other_tracks = set(df[df['user_id'] == other_user]['track_id'].unique())
        intersection = len(user_tracks & other_tracks)
        union = len(user_tracks | other_tracks)
        if union > 0:
            similarity = intersection / union

            # Select broad range for similarity buddy system for proto type (since there are no real users)
            if 0.0 <= similarity <= 0.6:
                user_similarity[other_user] = similarity

    # Pick the most similar user within the 40-60% similarity range
    if user_similarity:
        chosen_user = max(user_similarity, key=user_similarity.get)
        print(f'cf: current user: {user_id}', flush=True)
        print(f'cf: chosen buddy (40-60% similar): {chosen_user}', flush=True)

        # Recommend songs the chosen buddy has listened to but the user hasn't
        similar_user_tracks = set(df[df['user_id'] == chosen_user]['track_id'])
        recommendations = list(similar_user_tracks - user_tracks)

        # Filter the dataframe to include only recommended tracks
        recommended_tracks_df = df[df['track_id'].isin(recommendations)].drop_duplicates()
        print(f'cf: returning {len(recommended_tracks_df)} recommended tracks from similar user', flush=True)

        return recommended_tracks_df.head(num_recommendations)

    else:
        print(f'cf: current user: {user_id}', flush=True)
        print('cf: no buddy found', flush=True)
        return pd.DataFrame()  # Return an empty DataFrame if no buddy is found



# Step 3: Use FAISS for Approximate Nearest Neighbor Search on Audio Features


def build_faiss_index(df, feature_cols, name="faiss_index"):
    """
    Builds a FAISS index for approximate nearest neighbor search and saves it to a pickle file.

    Parameters:
        df (pd.DataFrame): The dataset containing feature columns.
        feature_cols (list): List of feature column names to use for indexing.
        name (str, optional): Name of the dataset (used in the filename). Default is "faiss_index".

    Returns:
        tuple: (FAISS index, Scaler)
    """
    # check if feature_cols are in the dataframe

    if not set(feature_cols).issubset(df.columns):
        raise ValueError("Feature columns not found in the DataFrame")

    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values.astype(np.float32))
    features = np.ascontiguousarray(features)  # Ensure contiguous memory

    index = faiss.IndexFlatL2(features.shape[1])  # L2 (Euclidean) distance
    index.add(features)

    # # Sanitize the name to create a safe filename
    # safe_name = re.sub(r'\W+', '_', name)  # Replace non-alphanumeric characters with "_"
    #
    # Save the FAISS index and scaler with the dataset name
    # filename = f"data/{safe_name}.pkl"
    # with open(filename, "wb") as f:
    #     pickle.dump((index, scaler), f)

    print(f"Built FAISS index with {features.shape[0]} rows", flush=True)
    # print(f"FAISS index saved successfully as {filename}")

    return index, scaler


def recommend_similar_tracks(track_id, df, index, scaler, feature_cols, num_recommendations=5):
    track_features = df[df['track_id'] == track_id][feature_cols].values.astype(np.float32)
    if track_features.shape[0] == 0:
        return pd.DataFrame()  # Return an empty DataFrame if track_id is not found

    track_features = scaler.transform(track_features)
    _, indices = index.search(track_features, num_recommendations + 1)  # +1 to exclude itself

    # Retrieve all columns for the recommended tracks
    recommended_tracks_df = df.iloc[indices[0][1:]].drop_duplicates()
    print(f'cb: returning {len(recommended_tracks_df)} similar tracks', flush=True)

    return recommended_tracks_df.head(num_recommendations)  # Return full DataFrame subset


# Step 4: Combine Both Approaches
# NOT USING THIS
def hybrid_recommendation(user_id, df, user_faiss, feature_cols, num_recommendations=5, cf_threshold=3):
    """
    Generate hybrid recommendations using collaborative filtering (CF) and content-based filtering (CB).

    Parameters:
        user_id (int): The user ID for whom recommendations are generated.
        df (pd.DataFrame): The dataset containing user and track information.
        user_faiss (tuple): Tuple containing the FAISS index and scaler for content-based filtering.
        feature_cols (list): List of feature column names for content-based filtering.
        num_recommendations (int, optional): Total number of recommendations to return. Default is 5.
        cf_threshold (int, optional): Minimum number of CF-based tracks before adding CB-based tracks. Default is 3.

    Returns:
        pd.DataFrame: A DataFrame of recommended tracks.
    """

    # Step 1: Get recommendations from similar users (Collaborative Filtering)
    user_based_recommendations = buddy_recommendations(user_id, df, num_recommendations)

    # If CF recommendations meet or exceed the threshold, return them directly
    if len(user_based_recommendations) >= cf_threshold:
        return pd.DataFrame(user_based_recommendations).drop_duplicates().head(num_recommendations)

    # Step 2: Load FAISS index and scaler
    index, scaler = user_faiss  # build_faiss_index(df, feature_cols)

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

    print(f'hybrid: returning {len(final_recommendations)} hybrid recommendations', flush=True)

    return final_recommendations


def recommend_similar_tracks_audio_ft(track_id, df, faiss_index, scaler, feature_cols, num_recommendations=5):
    """
    Recommend songs based on audio features using a pre-compiled FAISS index.

    Parameters:
        track_id (int or str): The track ID for which recommendations are generated.
        df (pd.DataFrame): The dataset containing track metadata and audio features.
        faiss_index (faiss.IndexFlatL2): The pre-compiled FAISS index.
        scaler (sklearn.preprocessing.StandardScaler): The scaler used to normalize features.
        feature_cols (list): List of feature column names used in FAISS.
        num_recommendations (int, optional): Number of recommendations to return. Default is 5.

    Returns:
        pd.DataFrame: A DataFrame of recommended tracks.
    """

    # Step 1: Retrieve the track's audio features
    track_row = df[df['track_id'] == track_id]

    if track_row.empty:
        print(f"ERROR: Track ID {track_id} not found in dataset!", flush=True)
        return pd.DataFrame()  # Return an empty DataFrame if track is not found

    track_features = track_row[feature_cols].values.astype(np.float32)

    # Step 2: Normalize the track's features using the same scaler
    track_features = scaler.transform(track_features)
    track_features = np.ascontiguousarray(track_features)  # Ensure contiguous memory

    # Step 3: Search for similar tracks in the FAISS index
    _, indices = faiss_index.search(track_features, num_recommendations + 1)  # +1 to exclude itself

    # Step 4: Retrieve recommended tracks
    recommended_tracks = df.iloc[indices[0][1:]]  # Exclude the first result (itself)

    print(f"FAISS: Found {len(recommended_tracks)} similar tracks for track ID {track_id}", flush=True)

    # Step 5: select least popular track from set
    inv_pop = inverse_popularity(recommended_tracks, 10)

    # Ensure inv_pop is not empty before updating recc_tracks
    if inv_pop.empty:
        print("WARNING: No recommendations after applying inverse popularity filter.", flush=True)
        return  # Stop execution if no valid recommendations

    print(f"FAISS: Found {len(inv_pop)} least popular tracks for track ID {track_id}", flush=True)

    return inv_pop

# Callback for when a marker is clicked directly on the figure
def on_point_click(trace, points, state):
    if points.point_inds:  # Only proceed if a point is clicked
        idx = points.point_inds[0]  # Get index of first clicked point
        valence = trace.x[idx]
        energy = trace.y[idx]
        valence_selected.set(valence)
        energy_selected.set(energy)
