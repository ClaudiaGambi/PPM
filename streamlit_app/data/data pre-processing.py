
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

df_cleaned = df_cleaned.groupby('track_id', as_index=False).agg({                               # group by track_id
    'popularity': 'max',                                                                       # calculate mean of popularity
    'track_genre': ', '.join,                                                                   # join genres
    **{col: 'first' for col in df_cleaned.columns if col not in ["popularity", "track_genre"]}})    # for all other columns get first value

# reorder dataframe
df_cleaned = df_cleaned[["track_id", "track_name", "album_name", "artists", "duration_ms", "popularity", "track_genre",
                         "explicit", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
                         "instrumentalness", "liveness", "valence", "tempo", "time_signature"]]

# write file
df_cleaned.to_csv("streamlit_app/data/spotify_tracks_clean.csv", index = False)