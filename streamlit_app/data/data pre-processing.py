
### DATA
import pandas as pd

# save data
df = pd.read_csv("streamlit_app/data/spotify_tracks.csv")   # save dataset
print(df.shape)                                 # shape: 114000, 21

# column names
print(df.columns)                               # column names
df = df.drop("Unnamed: 0", axis = 1)            # drop number column

# missing data
print(df.isnull().sum())                        # 1 missing on artist, album, and track
df = df.dropna()                                # remove 1 row

# check variables
print(df.dtypes)                                # column types
df_descriptives = df.describe()                 # save descriptives

print(df['mode'].unique())                      # unique values in mode: 0 / 1
df["mode"] = df["mode"].astype("category")      # change to category

# find duplicates on all columns
df_duplicates_all = df[df.duplicated(keep = "first")]
df_cleaned = df.drop_duplicates(keep = "first")

# find duplicates on track_id (all columns same but track_id)
df_duplicates_all_id = df_cleaned[df_cleaned.duplicated(subset = df_cleaned.columns.difference(["track_id"]), keep = "first")]
df_cleaned = df_cleaned.drop_duplicates(subset = df_cleaned.columns.difference(["track_id"]), keep = "first")

# find duplicates on track_id (same track_id, possible differences on other columns)
df_duplicates_id = df_cleaned[df_cleaned.duplicated(subset = ["track_id"], keep = False)]
df_duplicates = df_cleaned[df_cleaned.duplicated(subset = df_cleaned.columns.difference(["popularity", "track_genre"]), keep = False)]
print(df_duplicates.shape)  # all differences found in popularity and/or genre

df_cleaned = df_cleaned.groupby('track_id', as_index=False).agg({                       # group by track_id
    'popularity': 'max',                                                                # get max value for popularity
    **{col: 'first' for col in df_cleaned.columns if col not in ["popularity"]}})       # for all other columns get first value

# reorder dataframe
df_cleaned = df_cleaned[["track_id", "track_name", "album_name", "artists", "duration_ms", "popularity", "track_genre",
                         "explicit", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
                         "instrumentalness", "liveness", "valence", "tempo", "time_signature"]]

### add cover image links to df
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# authorization
# client_id = # removed by hku1
# client_secret = # removed by hku1
auth_manager = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)
sp = spotipy.Spotify(auth_manager = auth_manager)

# split df into chunks
num_chunks = 10
chunk_size = len(df_cleaned) // num_chunks
chunks = []

for i in range(num_chunks):
    start_idx = i * chunk_size
    if i == num_chunks - 1:
        end_idx = len(df_cleaned)
    else:
        end_idx = (i + 1) * chunk_size

    chunks.append(df_cleaned.iloc[start_idx:end_idx])

# get album covers
album_covers = []

for i, part in enumerate(chunks, 1):
    list_track_id = part["track_id"].tolist()       # get list of track id's per df chunk

    for j in range(0, len(list_track_id), 50):
        chunk_id = list_track_id[j:j + 50]          # create chucks of 50 track_id's
        track_info = sp.tracks(chunk_id)            # get track info per chunk

        for track in track_info["tracks"]:
            track_id = track["id"]                  # save track id
            if track["album"]["images"]:            # save album cover
                album_cover = track["album"]["images"][0]["url"]
            else:
                album_cover = "no image available"  # no album cover -> "no image available"

            album_covers.append({"track_id": track_id, "album_cover": album_cover})

# add album covers to df
album_covers = pd.DataFrame(album_covers)
df_cleaned = df_cleaned.merge(album_covers, on = 'track_id', how = 'left')

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
df_cleaned['popularity_genre'] = df_cleaned.groupby('track_genre')['popularity'].transform(rescale_and_round_popularity)

# Add tracks_played column based on popularity_genre

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
df_cleaned['tracks_played'] = base_plays * popularity_scale

# Round the values to the nearest integer
df_cleaned['tracks_played'] = df['tracks_played'].round(0).astype(int)

# Ensure that the minimum number of plays is at least 1
df_cleaned['tracks_played'] = df['tracks_played'].clip(lower=1)

df_cleaned.to_csv("spotify_tracks_clean.csv", index=False)  # save cleaned dataset
