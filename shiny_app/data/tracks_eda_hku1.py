import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

# Load the Spotify tracks dataset
tracks = pd.read_csv("spotify_tracks_clean_clusters_v2.csv")

audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# number of rows and columns in the dataset
num_rows, num_cols = tracks.shape
print(f'rows: {num_rows}, columns: {num_cols}')

# column names
column_names = tracks.columns
print(f'column names: {column_names}')

# distribution of audio features in a single boxplot x-axis labels 45 degrees rotation
# plt.figure(figsize=(10, 5))
sns.boxplot(data=tracks[audio_features])
plt.xticks(rotation=45)
plt.show()

# number of genre_clusters
num_genre_clusters = tracks['genre_cluster'].nunique()
print(f'number of genre_clusters: {num_genre_clusters}')
# boxplot of popularity by genre_cluster in a single boxplot
sns.boxplot(x='genre_cluster', y='popularity', data=tracks)
plt.xticks(rotation=45)
plt.show()

# number of genres
num_genres = tracks['track_genre'].nunique()
print(f'number of genres: {num_genres}')
# boxplot of popularity by track_genre in a single boxplot
# plt.figure(figsize=(10, 5))
sns.boxplot(x='track_genre', y='popularity', data=tracks)
plt.xticks(rotation=45)
plt.show()

# number of tracks
num_tracks = tracks['track_id'].nunique()
print(f'number of tracks: {num_tracks}')

# number of artists
num_artists = tracks['artists'].nunique()
print(f'number of artists: {num_artists}')

# number of albums
num_albums = tracks['album_name'].nunique()
print(f'number of artists: {num_albums}')

# number of tracks per genre
tracks_per_genre = tracks['track_genre'].value_counts()
print(f'tracks per genre: {tracks_per_genre}')

# number of tracks per artist
tracks_per_artist = tracks['artists'].value_counts()
print(f'tracks per artist: {tracks_per_artist}')

# check for missing values in album_cover column

missing_album_cover = tracks['album_cover'].isnull().sum()

print(f'missing album cover: {missing_album_cover}')
# check if album_cover colum contains non-URL values

non_url_album_cover_count = tracks['album_cover'].apply(lambda x: not x.startswith('http')).sum()
print(f'non-url album cover: {non_url_album_cover_count}')

# print the first 5 rows of the tracks dataset album_cover column wtih non-URL values
non_url_album_cover_rows = tracks.loc[~tracks['album_cover'].astype(str).str.startswith('http', na=False)].head()
print(non_url_album_cover_rows['album_cover'])

# check if all non_url_album_cover_rows contain same value in album_cover column
same_album_cover = non_url_album_cover_rows['album_cover'].nunique() == 1
print(f'same album cover: {same_album_cover}')






