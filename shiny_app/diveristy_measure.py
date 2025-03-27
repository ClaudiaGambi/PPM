from scipy.spatial.distance import cosine
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Load your user data
user_data = pd.read_csv(Path(__file__).parent / "data/synthetic_user_data.csv")

def jaccard_distance(set1, set2):
    if not set1 and not set2:
        return 0.0
    return 1 - len(set1 & set2) / len(set1 | set2)

def combined_intra_list_diversity(feature_list, genre_list, weight=0.5):
    """
    weight: between 0 (genre only) and 1 (audio only)
    """
    if len(feature_list) < 2:
        return 0.0

    distances = []
    for (i, j) in combinations(range(len(feature_list)), 2):
        audio_d = cosine(feature_list[i], feature_list[j])
        genre_d = jaccard_distance(genre_list[i], genre_list[j])

        # weigh audio features against genres
        combined = weight * audio_d + (1 - weight) * genre_d
        distances.append(combined)

    return np.mean(distances) if distances else 0.0

audio_cols = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Remove rows with missing audio features
user_data_filtered = user_data.dropna(subset=audio_cols)

user_ild_scores = []

# Get ILD
for user_id, group in user_data_filtered.groupby('user_id'):
    audio = group[audio_cols].values
    genres = group['track_genre'].fillna('').map(lambda g: {g.lower()}).tolist()
    ild = combined_intra_list_diversity(audio, genres, weight=0.6)
    user_ild_scores.append({'user_id': user_id, 'ild': ild})

# Create the dataframe
ild_df = pd.DataFrame(user_ild_scores)

# Plot ILD Distribution
plt.figure(figsize=(10, 5))
plt.hist(ild_df['ild'], bins=30, edgecolor='black')
plt.title('Distribution of Intra-List Diversity (ILD) Across Users')
plt.xlabel('ILD Score')
plt.ylabel('Number of Users')
plt.grid(True)
plt.tight_layout()
plt.savefig("ild_distribution.png")
