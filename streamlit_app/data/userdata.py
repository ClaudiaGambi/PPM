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
from scipy.stats import skewnorm, lognorm

# Load the Spotify tracks dataset
tracks = pd.read_csv("spotify_tracks.csv")
# tracks = pd.read_csv("streamlit_app/data/spotify_tracks.csv") # when executing lines manually
# Define parameters
num_users = 1000
num_interactions = 10000

# Generate synthetic users with a skewed age distribution
age_skewness = 10
ages = skewnorm.rvs(a=age_skewness, loc=25, scale=10, size=num_users)
ages = np.clip(ages, 18, 75).astype(int)

# Define possible values for gender and location, including None for missing data
gender_options = ['Male', 'Female', None]
location_options = ['USA', 'UK', 'Canada', 'Australia', None]

# Calculate the distribution of genres in the tracks dataset
genre_distribution = tracks['track_genre'].value_counts(normalize=True) # in this dataset each genre has exactly 1000 tracks

users = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'age': ages,
    'gender': np.random.choice(gender_options, size=num_users, p=[0.45, 0.45, 0.10]),
    'location': np.random.choice(location_options, size=num_users, p=[0.25, 0.25, 0.25, 0.15, 0.10]),
    'preferred_genre': np.random.choice(genre_distribution.index, size=num_users,
        p=genre_distribution.values)
})

# Normalize track popularity for weighted selection within genres
tracks['popularity_normalized'] = tracks.groupby('track_genre')['popularity'].transform(lambda x: x / x.sum())
# Generate synthetic interactions
interactions = []

# Determine number of tracks per user using a log-normal distribution
num_tracks_per_user = np.random.lognormal(mean=4.5, sigma=0.5, size=num_users).astype(int)

for user_id, num_tracks in zip(users['user_id'], num_tracks_per_user):
    preferred_genre = users.loc[users['user_id'] == user_id, 'preferred_genre'].values[0]
    genre_tracks = tracks[tracks['track_genre'] == preferred_genre]

    # Select tracks based on popularity within the preferred genre
    selected_tracks = genre_tracks.sample(n=num_tracks, replace=True, weights=genre_tracks['popularity_normalized'])

    for track_id in selected_tracks['track_id']:
        interaction_type = np.random.choice(['play', 'skip', 'share'])
        timestamp = datetime.now() - timedelta(days=np.random.randint(0, 30))

        # Determine if 'like' and 'not_like' should be set
        if np.random.rand() < 0.1:  # 10% chance to assign a like/not_like
            like = np.random.choice([0, 1], p=[0.8, 0.2])
            not_like = 1 - like
        else:
            like = np.nan
            not_like = np.nan

        # Initialize rating to None
        rating = None

        # Assign ratings for 10% of all user interactions
        # if like =1, rating is randomly chosen between 3 and 5
        # if not_like =1, rating is randomly chosen between 1 and 2
        # if both like and not_like are 0, rating is randomly chosen between 1 and 5
        if np.random.rand() < 0.1:
            if like == 1:
                rating = np.random.randint(3, 6)
            elif not_like == 1:
                rating = np.random.randint(1, 3)
            else:
                rating = int(np.clip(round(np.random.normal(loc=2.5, scale=1.0)), 1, 5))

        # add colum duration_listened. Conditions:
        # if interaction_type is play or share or like is 1, duration_listened is between 0 and duration_ms value as present in tracks dataframe
        # duration_listened is right skewed, i.e close to duration_ms value
        # if not_like is 1, duration_listened is between 0 and duration_ms value as present in tracks dataframe, left skewed towards 0
        # if interaction_type is skip duration_listened is o milliseconds

        if interaction_type in ['play', 'share'] or like == 1:
            duration_listened = np.random.randint(0, selected_tracks.loc[selected_tracks['track_id'] == track_id, 'duration_ms'].values[0])

        elif not_like == 1:
            duration_listened = np.random.randint(0, selected_tracks.loc[selected_tracks['track_id'] == track_id, 'duration_ms'].values[0])
            duration_listened = int(np.clip(round(np.random.exponential(scale=duration_listened / 2)), 0, duration_listened))

        else:
            duration_listened = 0

        interactions.append({
            'user_id': user_id,
            'track_id': track_id,
            'interaction_type': interaction_type,
            'timestamp': timestamp,
            'rating': rating,
            'like': like,
            'not_like': not_like
        })

# Convert interactions to DataFrame
interactions_df = pd.DataFrame(interactions)

# Merge user and interaction data
synthetic_data = interactions_df.merge(users, on='user_id')

# Display the synthetic dataset
print(synthetic_data.head())


# plot histogram of number of tracks per user, sorted from highest to lowest using seaborn

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

# Plot histogram of number of tracks per user
sns.histplot(num_tracks_per_user, bins=30, kde=True)
plt.xlabel('Number of Tracks per User')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Tracks per User')
plt.show()

# proportion of interactions with a like or not_like score
like_proportion = interactions_df['like'].notna().mean()
not_like_proportion = interactions_df['not_like'].notna().mean()

print(f'Proportion of interactions with a like score: {like_proportion:.2f}')
print(f'Proportion of interactions with a not_like score: {not_like_proportion:.2f}')

# plot distribution of ratings

# Filter out interactions with ratings
rated_interactions = interactions_df[interactions_df['rating'].notna()]

# Plot distribution of ratings
sns.histplot(rated_interactions['rating'], bins=5, discrete=True)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.show()








