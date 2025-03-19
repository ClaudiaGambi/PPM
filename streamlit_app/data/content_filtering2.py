
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import faiss

# Load the dataset
# df = pd.read_csv("streamlit_app/data/spotify_tracks_clean.csv") # if line is manually executed
df = pd.read_csv("spotify_tracks_clean.csv") # if using run

# Remove first column (duplicate index)
# df = df.drop(df.columns[0], axis=1)

# Drop rows with missing values
df = df.dropna()

# drop rows with duplicated in track_id column
df = df.drop_duplicates(subset=['track_id'])

# group by track_genre and add column 'popularity_genre' by calculating popularity of each track by genre and rescale between 0-100

# Define a function to rescale and round the popularity values for each genre
def rescale_and_round_popularity(series):
    scaler = MinMaxScaler(feature_range=(0, 100))
    # Reshape the series to fit the scaler
    scaled_values = scaler.fit_transform(series.values.reshape(-1, 1))
    # Flatten the scaled values and round to zero decimals
    rounded_values = scaled_values.flatten().round(0).astype(int)
    return rounded_values

# Apply the rescaling and rounding function to each group
df['popularity_genre'] = df.groupby('track_genre')['popularity'].transform(rescale_and_round_popularity)


# Define parameters for the log-normal distribution
mu = 8  # Mean of the underlying normal distribution
sigma = 0.1  # Standard deviation of the underlying normal distribution

# Generate a base log-normal distribution
base_plays = np.random.lognormal(mean=mu, sigma=sigma, size=len(df))

# Apply a non-linear transformation to the popularity_genre to make the decrease faster
# For example, use a power transformation
power_factor = 2  # You can adjust this factor to control the steepness of the curve
popularity_scale = (df['popularity_genre'] / 100) ** power_factor

# Calculate tracks_played by scaling the base log-normal distribution
df['tracks_played'] = base_plays * popularity_scale

# Round the values to the nearest integer
df['tracks_played'] = df['tracks_played'].round(0).astype(int)

# Ensure that the minimum number of plays is at least 1
df['tracks_played'] = df['tracks_played'].clip(lower=1)

# Group by 'track_genre' and sort each group by 'popularity_genre'
df = df.groupby('track_genre', group_keys=False).apply(lambda x: x.sort_values('popularity_genre', ascending=False)).reset_index(drop=True)

# Sample 1000 rows to test code
df = df.sample(10000, random_state=42).reset_index(drop=True)

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
audio_weight = 0.6
genre_weight = 0.3
artist_weight = 0.1

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
song_name = "Papaoutai" # Stromae
recommended_songs = recommend_faiss(song_name, df, index)
print(f"Songs similar to '{song_name}': {recommended_songs}")