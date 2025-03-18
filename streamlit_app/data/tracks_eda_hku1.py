import pandas as pd

# Load the Spotify tracks dataset
tracks = pd.read_csv("spotify_tracks.csv")
# tracks = pd.read_csv("streamlit_app/data/spotify_tracks.csv") # when executing lines manually

# number of genres
# Calculate the number of unique genres
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




