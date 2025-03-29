# create synthetic data for user data

# are you aware of this dataset?
#
# https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data

# Yes, I'm aware of the Spotify Tracks Dataset on Kaggle.
# This dataset typically contains information about various tracks available on Spotify, including features like track
# name, artist, album, release date, and various audio features such as danceability, energy, loudness, tempo, and more.
# These features can be used for a variety of data analysis and machine learning tasks, such as music recommendation systems,
# trend analysis, and audio feature exploration. If you have specific questions or need help with this dataset, feel free to ask!
#
# how would you create a synthetic user data set from users that interacted with this track data set?#

# Creating a synthetic user dataset based on interactions with the Spotify Tracks Dataset involves simulating user
# behaviors and interactions with the tracks.Here's a step-by-step approach to create such a synthetic dataset:#
# ### Step 1: Define User Profiles
#
# 1. ** User Demographics: **
# - Create attributes such as user ID, age, gender, location, and preferred genres.
#
# 2. ** Listening Habits: ** - Define attributes like average listening time per day, preferred listening times, and
# device used(e.g., mobile, desktop).
#
# ### Step 2: Simulate User Interactions
#
# 1. ** Select  Tracks: **
# - Randomly assign a subset of tracks from the Spotify# dataset to each user, ensuring a mix of popular and niche
# tracks.
#
# 2. ** Interaction Types: **
# - Define interaction types such as play, skip, like, and share.
#
# 3. ** Interaction# Frequency: **
# - Determine how often each user interacts with tracks, which can be based on their listening habits.#

# ### Step 3: Generate Interaction Data
#
# 1. ** Create a Timeframe: **
# - Decide on a timeframe for the interactions, such as a month or a year.
#
# 2. ** Simulate # Interactions: **
# - For each user, simulate interactions with tracks over the chosen timeframe.Use random distributions to model
# realistic behavior, such as:
#     - ** Normal  distribution **
#     for daily listening times.
#     - ** Poisson   distribution **
#     for the number of interactions per day.
#     - ** Exponential
#     distribution **
#     for time between interactions.
#
# 3. ** Incorporate Track Features: **
# - Use track features like danceability, energy, and tempo to influence interaction likelihood.For example, a user who
# prefers high - energy music might interact more with tracks having high energy scores.
#
# ### Step 4: Compile the Dataset
#
# 1. ** Structure the Data: **
# - Organize the data into a tabular format with columns such as user ID, track ID, interaction type, timestamp, and any
# other relevant features.

# 2. ** Ensure Diversity: **
# - Make sure the dataset reflects a diverse range of user behaviors and interactions.#

# ### Step 5: Validate the Dataset
#
# 1. ** Check for Realism: ** -
#
# Ensure that the synthetic data reflects realistic user behavior patterns.

# 2. ** Adjust Parameters: **
# - Tweak the parameters and distributions used in the simulation to better match expected user behaviors.
#

# modify the code in the following ways
#
# - the age distribution mean is around 30, median at 25 and long tail up to 75
# - per user id, user interactions should be clustered in genres, for example a user that listens to rock music has mainly
# interaction with other rock music genre tracks
# - the track id chosen under user interactions is based on popularity score in track dataset, with highly popular tracks overrepresented.
# - a fraction (~10%)  of the interactions should have a rating from 1 to 5 (integers only) for tracks

# - add empty options to both location and gender

# - ratings should be normally distributed around 2.5 and no rating if interaction type is skip - move like in
# interaction type colum to separate column called like that has binary values 0 and 1
# - create not _ike column that has# binary values 0 and 1.  1 in both like and no_like column is not possible
# - number of tracks per user follows a log distribution with a few users listening to relatively a lot of tracks


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('Qt5Agg')

np.random.seed(42)

# Load the Spotify tracks dataset
tracks = pd.read_csv("spotify_tracks_clean_clusters_v2.csv")

# tracks = tracks.drop(columns=['Unnamed: 0']) #remove columm 'unnamed: 0'
tracks = tracks.drop_duplicates(subset='track_id') # drop duplicate rows based on track_id

# Define parameters
num_users = 100

# Generate synthetic users
age_skewness = 10
ages = np.clip(np.random.normal(30, 10, num_users), 18, 75).astype(int)

gender_options = ['Male', 'Female', None]
location_options = ['USA', 'UK', 'Canada', 'Australia', None]

genre_distribution = tracks['track_genre'].value_counts(normalize=True)

users = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'age': ages,
    'gender': np.random.choice(gender_options, size=num_users, p=[0.45, 0.45, 0.10]),
    'location': np.random.choice(location_options, size=num_users, p=[0.25, 0.25, 0.25, 0.15, 0.10]),
    'preferred_genre': np.random.choice(genre_distribution.index, size=num_users, p=genre_distribution.values)
})

# Normalize track popularity for weighted selection within genres
tracks['popularity_genre'] = tracks.groupby('track_genre')['popularity'].transform(lambda x: x / x.sum())

# We will collect all interactions here
interactions = []

# Number of tracks per user follows a log-normal distribution
num_tracks_per_user = np.random.lognormal(mean=2.0, sigma=0.5, size=num_users).astype(int)

# A global counter to ensure each interaction has a unique microsecond offset
global_interaction_counter = 0

# --- Function to replicate each row in a DataFrame based on track popularity ---
def replicate_by_popularity(df, genre_scale=10):
    """
    For each row (track) in df, replicate the row based on its genre popularity.
    A higher 'popularity_genre' value (normalized popularity within the genre) results in more replicates.

    Parameters:
      df: DataFrame containing track rows (must include 'popularity_genre' column).
      genre_scale: A scaling factor to amplify the effect of genre popularity.

    Returns:
      A new DataFrame with replicated rows.
    """
    repeated_rows = []
    for idx, row in df.iterrows():
        # Compute replicate count based on genre popularity.
        replicate_count = max(1, np.random.poisson(lam=row['popularity_genre'] * genre_scale))
        repeated_rows.extend([row] * replicate_count)
    return pd.DataFrame(repeated_rows, columns=df.columns)


# Loop through each user to simulate interactions
for user_id, num_tracks in zip(users['user_id'], num_tracks_per_user):
    preferred_genre = users.loc[users['user_id'] == user_id, 'preferred_genre'].values[0]
    num_preferred_tracks = int(num_tracks * np.random.uniform(0.6, 0.8))
    num_other_tracks = num_tracks - num_preferred_tracks

    # Select ~60-80% tracks from the user's preferred genre, rest from other genres
    preferred_genre_tracks = tracks[tracks['track_genre'] == preferred_genre]
    selected_preferred_tracks = preferred_genre_tracks.sample(
        n=num_preferred_tracks,
        replace=True,
        weights=preferred_genre_tracks['popularity_genre']
    )

    other_genre_tracks = tracks[tracks['track_genre'] != preferred_genre]
    selected_other_tracks = other_genre_tracks.sample(
        n=num_other_tracks,
        replace=True,
        weights=other_genre_tracks['popularity_genre']
    )

    # Combine the selected tracks for the user and replicate based on genre popularity
    selected_tracks = pd.concat([selected_preferred_tracks, selected_other_tracks])
    repeated_tracks = replicate_by_popularity(selected_tracks, genre_scale=1000)

    # Iterate over each row of repeated_tracks
    for idx, track in repeated_tracks.iterrows():
        global_interaction_counter += 1

        track_id = track['track_id']

        # Randomly pick an interaction type
        interaction_type = np.random.choice(['play', 'skip', 'share'])

        # Random day + random second offset, plus an incremental microsecond offset
        timestamp = (
                datetime.now()
                - timedelta(days=np.random.randint(0, 30), seconds=np.random.randint(0, 86400))
                + timedelta(microseconds=global_interaction_counter)
        )

        # Determine if 'like' and 'not_like' should be set (10% chance)
        if np.random.rand() < 0.1:
            like = np.random.choice([0, 1], p=[0.8, 0.2])
            not_like = 1 - like
        else:
            like = np.nan
            not_like = np.nan

        # Assign rating for ~10% of interactions (if not skip)
        rating = None
        if interaction_type != 'skip' and np.random.rand() < 0.1:
            if like == 1:
                # Higher ratings for liked tracks
                rating = np.random.randint(3, 6)
            elif not_like == 1:
                # Lower ratings for not liked tracks
                rating = np.random.randint(1, 3)
            else:
                # Normally distributed rating around 3
                rating = int(np.clip(round(np.random.normal(loc=3, scale=1.0)), 1, 5))

        # Calculate duration_listened based on interaction
        track_duration = track['duration_ms']
        if interaction_type in ['play', 'share'] or like == 1:
            duration_listened = int(skewnorm.rvs(a=10, loc=0, scale=track_duration))
            duration_listened = np.clip(duration_listened, 0, track_duration)
        elif not_like == 1:
            duration_listened = int(skewnorm.rvs(a=-10, loc=0, scale=track_duration))
            duration_listened = np.clip(duration_listened, 0, track_duration)
        else:
            duration_listened = 0

        interactions.append({
            'user_id': user_id,
            'track_id': track_id,
            'interaction_type': interaction_type,
            'duration_listened': duration_listened,
            'timestamp': timestamp,
            'rating': rating,
            'like': like,
            'not_like': not_like
        })

# Convert interactions to DataFrame
interactions_df = pd.DataFrame(interactions)

# Merge user and interaction data
synthetic_data = interactions_df.merge(users, on='user_id')

# Merge synthetic data with tracks data
synthetic_data = synthetic_data.merge(tracks, on='track_id', how='left', sort=False)

# Calculate duration_listened fraction
synthetic_data['duration_listened_perc'] = synthetic_data['duration_listened'] / synthetic_data['duration_ms']

# Show the first few rows
# print(synthetic_data.head())

pass

## QC

# column names
print(synthetic_data.columns)

# total number of user interactions
total_interactions = synthetic_data.shape[0]
print(f'Total number of user interactions: {total_interactions}')

# plot histogram of number of tracks per user, sorted from highest to lowest using seaborn

# Plot histogram of number of total tracks per user
# sns.histplot(num_tracks_per_user, bins=30, kde=True)
# plt.xlabel('Number of Total Tracks per User')
# plt.ylabel('Frequency')
# plt.title('Distribution of Number of Total Tracks per User')
# plt.show()

# Plot histogram of number of unique tracks per user
unique_tracks_per_user = synthetic_data.groupby('user_id')['track_id'].nunique()

# sns.histplot(unique_tracks_per_user, bins=30, kde=True)
# plt.xlabel('Number of Unique Tracks per User')
# plt.ylabel('Frequency')
# plt.title('Distribution of Number of Unique Tracks per User')
# plt.show()

# proportion of interactions with a like or not_like score
like_proportion = synthetic_data['like'].notna().mean()
not_like_proportion = synthetic_data['not_like'].notna().mean()

print(f'Proportion of interactions with a like score: {like_proportion:.2f}')
print(f'Proportion of interactions with a not_like score: {not_like_proportion:.2f}')

# plot distribution of ratings

# Filter out interactions with ratings
# rated_interactions = synthetic_data['rating'].dropna()

# Plot distribution of ratings
# sns.histplot(synthetic_data['rating'].dropna(), bins=5, discrete=True) # Filter out interactions with ratings
# plt.xlabel('Rating')
# plt.ylabel('Frequency')
# plt.title('Distribution of Ratings')
# plt.show()

# number of genres
# Calculate the number of unique genres
num_genres = synthetic_data['track_genre'].nunique()
print(f'number of genres: {num_genres}')

# number of tracks
num_tracks = synthetic_data['track_id'].nunique()
print(f'number of tracks: {num_tracks}')

# number of artists
num_artists = synthetic_data['artists'].nunique()
print(f'number of artists: {num_artists}')

# number of albums
num_albums = synthetic_data['album_name'].nunique()
print(f'number of albums: {num_albums}')

# number of tracks per genre
tracks_per_genre = synthetic_data['track_genre'].value_counts()
print(f'tracks per genre: {tracks_per_genre}')

# number of tracks per artist
tracks_per_artist = synthetic_data['artists'].value_counts()
print(f'tracks per artist: {tracks_per_artist}')

# mean number of total tracks per user
mean_total_tracks = num_tracks_per_user.mean()
print(f'Mean number of total tracks per user: {mean_total_tracks:.2f}')

# mean number of unique tracks per user
mean_unique_tracks = unique_tracks_per_user.mean()
print(f'Mean number of unique tracks per user: {mean_unique_tracks:.2f}')

#  mean number of replicate tracks grouped by user
mean_replicate_tracks = synthetic_data.groupby(['user_id', 'track_id']).size().mean()
print(f'Mean number of replicate tracks per user: {mean_replicate_tracks:.2f}')

# rating distribution
rating_distribution = synthetic_data['rating'].value_counts(normalize=True)
print(f'Rating distribution:\n{rating_distribution}')

# number of users (users dataframe)

num_users = users['user_id'].nunique()
print(f'Number of unique users: {num_users}')

# user age distribution
age_distribution = users['age'].value_counts(normalize=True)
print(age_distribution)

# user gender distribution
gender_distribution = users['gender'].value_counts(normalize=True)
print(gender_distribution)

# users preferred genre distribution
preferred_genre_distribution = users['preferred_genre'].value_counts(normalize=True)
print(preferred_genre_distribution)

synthetic_data.to_csv('synthetic_user_data.csv', index=False)
# print('Synthetic user data saved to CSV file.')




# ### ADD A USER WHO IS SIMILAR TO OTHER USER FOR DEMO BUDDY SYSTEM

# --- Add two synthetic users who share listening history ---

# Step 1: Get original user_ids
original_user_ids = users['user_id'].tolist()
user1_id = original_user_ids[0]  # First user
user2_id = original_user_ids[1]  # Second user

# Step 2: Get their interaction data
user1_data = synthetic_data[synthetic_data['user_id'] == user1_id]
user2_data = synthetic_data[synthetic_data['user_id'] == user2_id]

# Step 4: Create new user ids for shared users
shared_user_1_id = users['user_id'].max() + 1
shared_user_2_id = shared_user_1_id + 1

# Step 5: Combine halves for shared users and assign new IDs
shared_user_1_data = pd.concat([user2_data, user1_data]).copy()
shared_user_2_data = pd.concat([user1_data, user2_data]).copy()

shared_user_1_data['user_id'] = shared_user_1_id
shared_user_2_data['user_id'] = shared_user_2_id

# Step 6: Append to synthetic_data
synthetic_data = pd.concat([synthetic_data, shared_user_1_data, shared_user_2_data], ignore_index=True)
print(synthetic_data)

# Save updated dataset
synthetic_data.to_csv('synthetic_user_data.csv', index=False)
# print(f"Added new user {new_user_id} with 50% similarity to user[1]")
