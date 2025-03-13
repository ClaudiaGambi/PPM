import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import faiss

# Load the dataset
# df = pd.read_csv("streamlit_app/data/spotify_tracks.csv") # if manual executed
df = pd.read_csv("spotify_tracks.csv") # if using run

# Sample 1000 rows to test code
df = df.sample(10000, random_state=42).reset_index(drop=True)

# Remove first column (duplicate index)
df = df.drop(df.columns[0], axis=1)

# Drop rows with missing values
df = df.dropna()

# Selecting numerical audio features for similarity calculation
features = ["danceability", "energy", "valence", "tempo", "acousticness",
            "instrumentalness", "liveness", "speechiness", "loudness"]

# Normalize features using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Convert to sparse matrix
df_scaled_sparse = csr_matrix(df_scaled)

# Genre and Artist embeddings
df["track_genre"] = df["track_genre"].fillna("")
df["artists"] = df["artists"].fillna("")

# Use TF-IDF for genre & artist representation
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
genre_matrix = tfidf_vectorizer.fit_transform(df["track_genre"])
artist_matrix = tfidf_vectorizer.fit_transform(df["artists"])

# Compute Weighted Similarity
audio_weight = 0.2
genre_weight = 0.1
artist_weight = 0.7

if not np.isclose(audio_weight + genre_weight + artist_weight, 1.0):
    raise ValueError("Weights must sum up to 1.")

# Scale genre and artist matrices to match feature importance
df_scaled_sparse *= audio_weight
genre_matrix *= genre_weight
artist_matrix *= artist_weight

# Recombine features with weights applied
df_weighted_sparse = hstack([df_scaled_sparse, genre_matrix, artist_matrix])

# Convert sparse matrix to dense numpy array for FAISS
df_numpy = df_weighted_sparse.toarray().astype(np.float32)

# Build FAISS index for fast similarity search
index = faiss.IndexFlatL2(df_numpy.shape[1])  # L2 distance
index.add(df_numpy)  # Add track vectors to index

# Search for nearest neighbors using FAISS
def recommend_faiss(song_name, df, index, top_n=5):
    if song_name not in df['track_name'].values:
        return "Song not found in the dataset."

    song_idx = df[df["track_name"] == song_name].index[0]
    song_vector = df_numpy[song_idx].reshape(1, -1)

    # Search for similar songs
    _, indices = index.search(song_vector, top_n + 1)  # +1 to ignore itself

    # Retrieve song names and artists for the recommended indices
    recommended_tracks = df.iloc[indices[0][1:]][["track_name", "artists"]]

    # Format the output as a list of tuples (track_name, artist)
    recommendations = [(row["track_name"], row["artists"]) for _, row in recommended_tracks.iterrows()]

    return recommendations

# Example usage
song_name = "Daughters" # John Mayer
recommended_songs = recommend_faiss(song_name, df, index)
print(f"Songs similar to '{song_name}': {recommended_songs}")