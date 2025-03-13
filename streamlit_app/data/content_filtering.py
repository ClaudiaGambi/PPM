# create similarties to allow conent-based filtering

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("streamlit_app/data/spotify_tracks.csv")

# sample 1000 rows to test code
df = df.sample(1000, random_state=42).reset_index(drop=True)


# Some EDA
print(df.head())

# remove first column (duplicate index)

df = df.drop(df.columns[0], axis=1)

df.info()
print(df.columns)
df.describe()

# only 1 value missing in some colums (see df.info()). We can drop these rows
df = df.dropna()

# Selecting numerical audio features for similarity calculation
features = ["danceability", "energy", "valence", "tempo", "acousticness",
            "instrumentalness", "liveness", "speechiness", "loudness"]

# Normalize features using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=features)
df_scaled.index = df.index  # Keep original index

import faiss

# Convert data to numpy array
df_numpy = np.ascontiguousarray(df_scaled.values, dtype=np.float32)

# Build FAISS index for fast similarity search
index = faiss.IndexFlatL2(df_numpy.shape[1])  # L2 distance (can also use cosine similarity)
index.add(df_numpy)  # Add track vectors to index

# Search for nearest neighbors
def recommend_faiss(song_name, df, index, top_n=5):
    song_idx = df[df["track_name"] == song_name].index[0]
    song_vector = df_numpy[song_idx].reshape(1, -1)

    # Search for similar songs
    _, indices = index.search(song_vector, top_n + 1)  # +1 to ignore itself

    return df.iloc[indices[0][1:]]["track_name"].tolist()

# Example
print(recommend_faiss("No Other Name", df, index))

# Genre and Artist embeddings

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder

# Combine multiple genres into a single string per track
df["track_genre"] = df["track_genre"].fillna("")
df["artists"] = df["artists"].fillna("")

# Use TF-IDF for genre & artist representation
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
genre_matrix = tfidf_vectorizer.fit_transform(df["track_genre"])
artist_matrix = tfidf_vectorizer.fit_transform(df["artists"])

# One-Hot Encode Artists (alternative to TF-IDF)
# encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
# artist_matrix = encoder.fit_transform(df[["artists"]])

# Convert to DataFrames for easier use
genre_df = pd.DataFrame(genre_matrix.toarray(), index=df.index)
artist_df = pd.DataFrame(artist_matrix.toarray(), index=df.index)
# artist_df = pd.DataFrame(artist_matrix, index=df.index)

# Ensure column names match unique genres & artists
genre_df.columns = [f"genre_{i}" for i in range(genre_df.shape[1])]
artist_df.columns = [f"artist_{i}" for i in range(artist_df.shape[1])]

# convert

# combine audio features with genre and artist embeddings
df_final = pd.concat([df_scaled, genre_df, artist_df], axis=1)

# Compute Weighted Similarity

# Define weights for different feature groups (sum up to 1)
audio_weight = 0.1
genre_weight = 0.4
artist_weight = 0.5

if audio_weight + genre_weight + artist_weight != 1:
    raise ValueError("Weights must sum up to 1.")

# Scale genre and artist matrices to match feature importance
genre_df *= genre_weight
artist_df *= artist_weight
df_scaled *= audio_weight

# Recombine features with weights applied
df_weighted = pd.concat([df_scaled, genre_df, artist_df], axis=1)

# Compute new similarity matrix
weighted_similarity_matrix = cosine_similarity(df_weighted, df_weighted)

# Convert to DataFrame
weighted_similarity_df = pd.DataFrame(weighted_similarity_matrix, index=df["track_name"], columns=df["track_name"])

# Recommend songs based on weighted similarity

def recommend_songs_weighted(song_name, df, similarity_df, top_n=5):
    if song_name not in similarity_df.index:
        return "Song not found in the dataset."

    # Get similarity scores for the song
    similar_songs = similarity_df[song_name].sort_values(ascending=False)[1:top_n+1]

    return list(similar_songs.index)

# Example usage
song_name = "Daughters"
recommended_songs = recommend_songs_weighted(song_name, df, weighted_similarity_df)
print(f"Songs similar to '{song_name}': {recommended_songs}")
