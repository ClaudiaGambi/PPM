# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from collections import Counter
from pathlib import Path

# %%
def get_most_similar_tracks(df_track, df_users, user_id, top_n=10):
    # 1. Filter alle interacties van de gebruiker
    user_data = df_users[df_users['user_id'] == user_id]
    
    # 2. Bereken de gemiddelde waarde voor de geselecteerde audiofeatures
    audio_features = ['danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']
    scaler = MinMaxScaler()
    user_scaled = scaler.fit_transform(user_data[audio_features])
    avg_user_features = np.mean(user_scaled, axis=0)
    
    # 3. Bepaal het meest geluisterde genre en artiest
    most_common_genre = Counter(user_data['track_genre']).most_common(1)[0][0]
    most_common_artist = Counter(user_data['artists']).most_common(1)[0][0]
    
    # 4. Normaliseer de audio features van df_track
    track_scaled = scaler.transform(df_track[audio_features])
    
    # 5. Bereken de afstand tot elke track
    distances = []
    for i, row in df_track.iterrows():
        track_features = track_scaled[i]
        audio_distance = euclidean(avg_user_features, track_features) * 0.6
        genre_distance = 0 if row['track_genre'] == most_common_genre else 1
        artist_distance = 0 if row['artists'] == most_common_artist else 1
        total_distance = audio_distance + genre_distance * 0.3 + artist_distance * 0.1
        distances.append(total_distance)
    
    # 6. Voeg de afstand toe aan de dataframe en sorteer op gelijkenis
    df_track['similarity_score'] = distances
    similar_tracks = df_track.sort_values(by='similarity_score', ascending=True).head(top_n)
    
    return similar_tracks

# %%
df_tracks = pd.read_csv(r'C:\Documenten\ADS\PPM\Assignment2\Git\PPM\streamlit_app\data\spotify_tracks_clean.csv')
#Random selection of 1000
df_tracks_sample = df_tracks.sample(1000)
df_tracks_sample = df_tracks_sample.dropna(subset=['danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness', 'track_genre', 'artists'])
df_tracks_sample = df_tracks_sample.reset_index(drop=True)

# %%
df_users = pd.read_csv(r'C:\Documenten\ADS\PPM\Assignment2\Git\PPM\streamlit_app\data\synthetic_user_data.csv')
df_users = df_users.dropna(subset=['danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness', 'track_genre', 'artists'])
df_users = df_users.reset_index(drop=True)

# %%
get_most_similar_tracks(df_tracks, df_users,77, top_n=10)


