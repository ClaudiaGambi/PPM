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
tracks_data = process_tracks(data) # for app.py

# user data loading
user_data = pd.read_csv(Path(__file__).parent / "data/synthetic_user_data.csv")
user_data = process_users(user_data)  # for app.py

# save audio_features
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# scale audio features
scaler = MinMaxScaler()
track_scaled = scaler.fit_transform(data[audio_features])

# define knn function based on valence and energy
def knn_module(data, valence=0.5, energy=0.5, max_knn=500):

    # extract valence and energy
    feature_data = data[["valence", "energy"]].to_numpy()

    # ensure n_neighbors does not exceed available data
    n_neighbors = min(len(data), max_knn)

    if n_neighbors < 1:
        print("WARNING: No data available for KNN search!", flush=True)
        return pd.DataFrame()

    # fit KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(feature_data)

    # find nearest neighbors to user-selected values
    distances, indices = knn.kneighbors(np.array([[valence, energy]]))

    # retrieve nearest neighbors from original df
    nearest_neighbors = data.iloc[indices[0]]
    print(f"knn: Returning {len(nearest_neighbors)} nearest neighbors", flush=True)

    return nearest_neighbors.reset_index(drop=True)

# function to remove Christmas songs
def filter_christmas_songs(tracks_df):
    christmas_keywords = ["christmas", "santa", "saint", "frosty", "snowman"]
    
    # change to lowercase
    mask = tracks_df["track_name"].str.lower().str.contains('|'.join(christmas_keywords), na=False) | \
           tracks_df["album_name"].str.lower().str.contains('|'.join(christmas_keywords), na=False)

    # remove Christmas songs
    filtered_df = tracks_df[~mask]
    
    # reset index
    filtered_df = filtered_df.reset_index(drop=True)
    
    return filtered_df

# function to get most similar tracks based on listening history
def get_most_similar_tracks(df_track, df_users, user_id, top_n=200):

    # filter all interactions of user
    user_data = df_users[df_users['user_id'] == user_id]

    # calc average value for selected audio features
    user_scaled = scaler.fit_transform(user_data[audio_features])
    avg_user_features = np.mean(user_scaled, axis=0)

    # get & print most listened to genre and artist
    most_common_genre = Counter(user_data['track_genre']).most_common(1)[0][0]
    print(f'most common genre: {most_common_genre}', flush=True)
    most_common_artist = Counter(user_data['artists']).most_common(1)[0][0]
    print(f'most common artist: {most_common_artist}\n', flush=True)

    # calc distance to each track
    distances = []
    for i, row in df_track.iterrows():
        track_features = track_scaled[i]
        audio_distance = euclidean(avg_user_features, track_features)
        genre_distance = 0 if row['track_genre'] == most_common_genre else 1
        artist_distance = 0 if row['artists'] == most_common_artist else 1
        total_distance = audio_distance * 0.5 + genre_distance * 0.3 + artist_distance * 0.2

        distances.append(total_distance)

    # print average distance
    print(f'avg total distance: {np.mean(distances)}\n', flush=True)

    # add the distance to df & sort by similarity
    df_track['similarity_score'] = distances
    similar_tracks = df_track.sort_values(by='similarity_score', ascending=True).head(top_n)
    print(f'sim track: returning {len(similar_tracks)} similar tracks', flush=True)

    return similar_tracks

# function inverse popularity
def inverse_popularity(df_tracks, top_n):
    # normalize popularity weights
    weights = df_tracks['popularity_weight'] / df_tracks['popularity_weight'].sum()

    #  ensure `top_n` does not exceed available data
    top_n = min(top_n, len(df_tracks))

    # select tracks using weighted random sampling
    selected_indices = np.random.choice(df_tracks.index, size=top_n, replace=False, p=weights)

    return df_tracks.loc[selected_indices]

# check if system has an active internet connection (for laoding images)
def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# function generate recommended tracks
def generate_recommended_tracks_list(tracks):
    if tracks.empty:
        return tags.p("No recommendations found.")

    # check internet connection before processing images
    internet_available = is_connected()

    # generate recommended track list
    items = []
    for _, row in tracks.iterrows():

        # get album covers
        album_cover_src = row["album_cover"]

        # no internet -> use placeholder
        if not internet_available or album_cover_src == "album_cover_placeholder.png":
            album_cover_src = "album_cover_placeholder.png"

        # div for recommended track list
        item = tags.div(
            tags.img(src=album_cover_src, height="64px", width="64px", style="margin-right:10px;"),
            tags.div(
                tags.b(row["track_name"]),
                tags.div(f"by {row['artists']}"),
                tags.div(f"Album: {row['album_name']}"),
                tags.div(f"genre: {row['track_genre']}"),
                style="display: inline-block; vertical-align: top;"
            ),
            style="background-color: #1e3352; padding: 10px; margin-bottom: 10px; border-radius: 5px; font-size: 15px; white-space: nowrap; overflow: scroll;",
        )
        items.append(item)

    return tags.div(*items)

# function buddy recommendations
def buddy_recommendations(user_id, df, num_recommendations=5):

    # get unique tracks for user
    user_tracks = set(df[df['user_id'] == user_id]['track_id'].unique())

    # compute Jaccard similarity with other users
    user_similarity = {}
    for other_user in df['user_id'].unique():
        if other_user == user_id:
            continue
        other_tracks = set(df[df['user_id'] == other_user]['track_id'].unique())
        intersection = len(user_tracks & other_tracks)
        union = len(user_tracks | other_tracks)
        if union > 0:
            similarity = intersection / union
            if 0.0 <= similarity <= 0.6:
                user_similarity[other_user] = similarity

    # pick most similar user (within the 40-60% similarity range)
    if user_similarity:
        chosen_user = max(user_similarity, key=user_similarity.get)
        print(f'cf: current user: {user_id}', flush=True)
        print(f'cf: chosen buddy (40-60% similar): {chosen_user}', flush=True)

        # recommend songs the chosen buddy has listened to but the user hasn't
        similar_user_tracks = set(df[df['user_id'] == chosen_user]['track_id'])
        recommendations = list(similar_user_tracks - user_tracks)

        # filter the dataframe to include only recommended tracks
        recommended_tracks_df = df[df['track_id'].isin(recommendations)].drop_duplicates()
        print(f'cf: returning {len(recommended_tracks_df)} recommended tracks from similar user', flush=True)

        return recommended_tracks_df.head(num_recommendations)

    # no buddy found
    else:
        print(f'cf: current user: {user_id}', flush=True)
        print('cf: no buddy found', flush=True)
        return pd.DataFrame()

# function FAISS index for approximate nearest neighbor search
def build_faiss_index(df, feature_cols, name="faiss_index"):

    # scale feature columns
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values.astype(np.float32))
    features = np.ascontiguousarray(features)  # ensure contiguous memory

    # create FAISS index
    index = faiss.IndexFlatL2(features.shape[1])  # L2 (Euclidean) distance
    index.add(features)
    print(f"Built FAISS index with {features.shape[0]} rows", flush=True)

    return index, scaler

# function similar tracks recommendations
def recommend_similar_tracks(track_id, df, index, scaler, feature_cols, num_recommendations=5):

    # get & transform track features
    track_features = df[df['track_id'] == track_id][feature_cols].values.astype(np.float32)
    track_features = scaler.transform(track_features)
    _, indices = index.search(track_features, num_recommendations + 1)  # +1 to exclude itself

    # retrieve all columns for the recommended tracks
    recommended_tracks_df = df.iloc[indices[0][1:]].drop_duplicates()
    print(f'cb: returning {len(recommended_tracks_df)} similar tracks', flush=True)

    return recommended_tracks_df.head(num_recommendations)

# function recommendations based on audio features
def recommend_similar_tracks_audio_ft(track_id, df, faiss_index, scaler, feature_cols, num_recommendations=5):

    # retrieve track's audio features
    track_row = df[df['track_id'] == track_id]
    track_features = track_row[feature_cols].values.astype(np.float32)

    # normalize the track's features
    track_features = scaler.transform(track_features)
    track_features = np.ascontiguousarray(track_features)  # Ensure contiguous memory

    # search for similar tracks in the FAISS index
    _, indices = faiss_index.search(track_features, num_recommendations + 1)  # +1 to exclude itself

    # retrieve recommended tracks
    recommended_tracks = df.iloc[indices[0][1:]]  # Exclude the first result (itself)

    print(f"FAISS: Found {len(recommended_tracks)} similar tracks for track ID {track_id}", flush=True)

    # select least popular track from set
    inv_pop = inverse_popularity(recommended_tracks, 10)

    # ensure inv_pop is not empty before updating recc_tracks
    if inv_pop.empty:
        print("WARNING: No recommendations after applying inverse popularity filter.", flush=True)
        return

    print(f"FAISS: Found {len(inv_pop)} least popular tracks for track ID {track_id}", flush=True)

    return inv_pop

# callback for when a marker is clicked directly on the figure
def on_point_click(trace, points, state):
    if points.point_inds:

        # get index of first clicked point
        idx = points.point_inds[0]
        valence = trace.x[idx]
        energy = trace.y[idx]
        valence_selected.set(valence)
        energy_selected.set(energy)
