from scipy.spatial.distance import cosine
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Load your user data
user_data = pd.read_csv(Path(__file__).parent / "data/synthetic_user_data.csv")

# indicating how spread-out the user’s genre listening is
def genre_entropy(genre_list):
    genre_counts = pd.Series(genre_list).value_counts(normalize=True)

    return entropy(genre_counts)

def audio_only_intra_list_diversity(feature_list):
    if len(feature_list) < 2:
        return 0.0

    distances = []
    for (i, j) in combinations(range(len(feature_list)), 2):
        audio_d = cosine(feature_list[i], feature_list[j])
        distances.append(audio_d)

    return np.mean(distances) if distances else 0.0

audio_cols = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Remove rows with missing audio features
user_data_filtered = user_data.dropna(subset=audio_cols)

user_ild_scores = []

for user_id, group in user_data_filtered.groupby('user_id'):
    audio = group[audio_cols].values
    genres = group['track_genre'].fillna('').str.lower().tolist()

    audio_div = audio_only_intra_list_diversity(audio)
    genre_div = genre_entropy(genres)

    user_ild_scores.append({
        'user_id': user_id,
        'audio_ild': audio_div,
        'genre_entropy': genre_div
    })

ild_df = pd.DataFrame(user_ild_scores)

ild_df['genre_entropy_norm'] = ild_df['genre_entropy'] / ild_df['genre_entropy'].max()

weight = 0.6
ild_df['combined_ild'] = weight * ild_df['audio_ild'] + (1 - weight) * ild_df['genre_entropy_norm']


# Plot ILD Distribution
plt.figure(figsize=(10, 5))
plt.hist(ild_df['combined_ild'], bins=30, edgecolor='black')
plt.title('Distribution of Intra-List Diversity (ILD) Across Users')
plt.xlabel('ILD Score')
plt.ylabel('Number of Users')
plt.grid(True)
plt.tight_layout()
plt.savefig("ild_distribution.png")
