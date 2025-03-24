import pandas as pd

# Load the Spotify tracks dataset
tracks = pd.read_csv("spotify_tracks_clean_clusters.csv")


# number of genre_clusters

num_genre_clusters = tracks['genre_cluster'].nunique()
print(f'number of genre_clusters: {num_genre_clusters}')

# number of genres
num_genres = tracks['track_genre'].nunique()
print(f'number of genres: {num_genres}')

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






