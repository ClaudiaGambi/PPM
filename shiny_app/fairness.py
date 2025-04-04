# Script to visualize the fairness of track recommendations

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from shiny_app.functions import knn_module
from shiny_app.functions import get_most_similar_tracks
from shiny_app.functions import inverse_popularity
from shiny_app.functions import filter_christmas_songs

matplotlib.use('Qt5Agg')

# Load the tracks data
track_data = pd.read_csv("data/spotify_tracks.csv")
# track_data = pd.read_csv("shiny_app/data/spotify_tracks.csv") # manual code execution
track_data = track_data.sample(frac=0.3, random_state=42) # same dataset as used in app

# Load the synthetic user data

user_data = pd.read_csv("data/synthetic_user_data.csv")
# user_data = pd.read_csv("shiny_app/data/synthetic_user_data.csv") # manual code execution
user_id = 1 # same user as used in app
smpl = 1000 # number of tracks to sample
np.random.seed(42) # set random seed for reproducibility

# sample 100 tracks from data based on popularity

def sample_tracks(data, n=100):
    """
    Sample n tracks from the data based on popularity.
    """
    sampled_tracks = []
    for _ in range(n):
        sampled_track = data.sample(n=1, weights=data['popularity'])
        sampled_tracks.append(sampled_track)
    return pd.concat(sampled_tracks, ignore_index=True)

# Sample 1000 tracks from the data

sampled_tracks = sample_tracks(track_data, n=smpl)

# plot the proportion of sampled tracks by popularity

def plot_proportion_by_popularity(sampled_tracks):
    """
    Plot the proportion of sampled tracks by popularity.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(sampled_tracks['popularity'], bins=20, color='blue', alpha=0.7)
    plt.title('Proportion of Sampled Tracks by Popularity')
    plt.xlabel('Popularity')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

# plot_proportion_by_popularity(sampled_tracks)


def recommended_tracks(df_track, df_user, usr_id, valence, energy, top_n):


    # Call KNN function with updated values
    nn = knn_module(df_track, valence, energy)

    # apply filter function to remove christmas songs
    nc = filter_christmas_songs(nn)

    # Get similar tracks based on KNN results
    sim = get_most_similar_tracks(nn, df_user, usr_id, top_n=top_n)
    sim = sim.head(5)

    # check max length if top_n == 200
    if top_n == 200:
        max_tracks = len(sim)
        top_n = min(max_tracks, 200)

    # Check if sim is empty before applying inverse_popularity
    if sim.empty:
        print("WARNING: No similar tracks found. No recommendations available.", flush=True)
        return  # Stop execution if no recommendations are found

    # Apply inverse popularity filter
    inv_pop = inverse_popularity(sim, top_n)

    # Ensure inv_pop is not empty before updating recc_tracks
    if inv_pop.empty:
        print("WARNING: No recommendations after applying inverse popularity filter.", flush=True)
        return  # Stop execution if no valid recommendations

    return inv_pop.iloc[[0]]

# return min and max energy and valence values in tracks_data

def get_min_max_values(df):
    """
    Get the minimum and maximum values of valence and energy in the dataset.
    """
    min_valence = df['valence'].min()
    max_valence = df['valence'].max()
    min_energy = df['energy'].min()
    max_energy = df['energy'].max()

    return min_valence, max_valence, min_energy, max_energy

min_valence, max_valence, min_energy, max_energy = get_min_max_values(track_data)

# sample 1000 tracks using recommended_tracks function, with random valence and energy values based on min and max values


def sample_recommended_tracks(df_track, df_user, usr_id, n=100):
    """
    Sample n recommended tracks based on random valence and energy values.
    """
    sampled_recommended_tracks = []
    for _ in range(n):
        # Generate random valence and energy values
        valence = np.random.uniform(min_valence, max_valence)
        energy = np.random.uniform(min_energy, max_energy)

        # Get recommended tracks
        recommended = recommended_tracks(df_track, df_user, usr_id, valence, energy, top_n=200)
        if recommended is not None:
            sampled_recommended_tracks.append(recommended)

    return pd.concat(sampled_recommended_tracks, ignore_index=True) if sampled_recommended_tracks else pd.DataFrame()

# Sample 1000 recommended tracks

sampled_recommended_tracks = sample_recommended_tracks(track_data, user_data, user_id, n=smpl)

# plot the proportion of sampled recommended tracks by popularity

# plot_proportion_by_popularity(sampled_recommended_tracks)

# overlay the two histograms
def overlay_histograms(sampled_tracks, sampled_recommended_tracks):
    """
    Overlay histograms of sampled tracks and sampled recommended tracks.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(sampled_tracks['popularity'], bins=20, color='blue', alpha=0.7, label='Sampled Tracks')
    plt.hist(sampled_recommended_tracks['popularity'], bins=20, color='orange', alpha=0.7, label='Sampled Recommended Tracks')
    plt.title('Overlay of Sampled Tracks and Sampled Recommended Tracks by Popularity')
    plt.xlabel('Popularity')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Sampled tracks and sampled recommended tracks are already generated above
    # You can call the overlay_histograms function here if needed
    overlay_histograms(sampled_tracks, sampled_recommended_tracks)