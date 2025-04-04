import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import lognorm, skewnorm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os


# Get the current working directory
current_dir = os.getcwd()
#Make sure you are in the shiny_app/data folder

# Path to the CSV file in the 'shiny_app/data' folder
file_path = os.path.join(current_dir, 'spotify_track.csv')
tracks = pd.read_csv(file_path)


# Define genre clusters
genre_clusters = {
    "acoustic & traditional": ['acoustic', 'country', 'folk', 'honky-tonk', 'bluegrass', 'singer-songwriter'],
    "calm": ['sleep', 'ambient', 'study', 'chill', 'new-age'],
    "children": ['kids', 'children'],
    "classical": ['piano', 'classical', 'opera'],
    "comedy": ['comedy'],
    "dance & electronic": ['trip-hop', 'idm', 'hardstyle', 'electronic', 'club', 'detroit-techno', 'happy', 'disco',
                            'techno', 'house', 'chicago-house', 'trance', 'j-dance', 'dance', 'edm', 'breakbeat',
                            'drum-and-bass', 'minimal-techno', 'electro', 'progressive-house', 'dubstep', 'deep-house'],
    "global": ['world-music', 'afrobeat'],
    "hip-hop & r&b": ['hip-hop', 'r-n-b'],
    "jazz, soul & funk": ['blues', 'jazz', 'funk', 'soul', 'gospel'],
    "latin & brazilian": ['latin', 'salsa', 'pagode', 'brazil', 'sertanejo', 'forro', 'latino', 'mpb', 'tango', 'samba'],
    "metal & heavy": ['goth', 'death-metal', 'industrial', 'grindcore', 'metalcore', 'metal', 'black-metal', 'hardcore', 'groove', 'heavy-metal'],
    "pop & mainstream": ['power-pop', 'j-idol', 'j-pop', 'mandopop', 'synth-pop', 'k-pop', 'cantopop', 'indie-pop', 'pop', 'pop-film'],
    "reggae, ska & dub": ['ska', 'reggae', 'dub', 'dancehall', 'reggaeton'],
    "rock & alternative": ['rockabilly', 'emo', 'alternative', 'punk-rock', 'garage', 'guitar', 'rock', 'rock-n-roll', 'alt-rock', 'psych-rock', 'grunge', 'punk', 'indie', 'hard-rock', 'j-rock'],
    "soundtrack": ['anime', 'show-tunes', 'disney']
}

# Define personas
personas = [
    {"name": "Ambitious Youth", "age_group": (14, 22), "listening_frequency": "often", "preferred_clusters": ["dance & electronic", "hip-hop & r&b", "pop & mainstream"], "share": 0.08},
    {"name": "Adventurous City Dwellers", "age_group": (22, 46), "listening_frequency": "often", "preferred_clusters": ["rock & alternative", "pop & mainstream"], "share": 0.08},
    {"name": "Self-Aware Family People", "age_group": (22, 46), "listening_frequency": "sometimes", "preferred_clusters": ["hip-hop & r&b", "pop & mainstream"], "share": 0.11},
    {"name": "Technical Doers", "age_group": (43, 56), "listening_frequency": "often", "preferred_clusters": ["rock & alternative", "pop & mainstream", "dance & electronic", "metal & heavy"], "share": 0.2},
    {"name": "Caring Multitaskers", "age_group": (33, 57), "listening_frequency": "rarely", "preferred_clusters": ["pop & mainstream"], "share": 0.22},
    {"name": "Authentic Believers", "age_group": (30, 58), "listening_frequency": "rarely", "preferred_clusters": ["classical"], "share": 0.08},
    {"name": "Wealthy In-Depth Seekers", "age_group": (54, 73), "listening_frequency": "often", "preferred_clusters": ["classical", "jazz, soul & funk"], "share": 0.09},
    {"name": "Cautious Seniors", "age_group": (63, 76), "listening_frequency": "sometimes", "preferred_clusters": ["acoustic & traditional", "classical"], "share": 0.14}
]

# Gender- en locatie-opties
gender_options = ['Male', 'Female', None]
location_options = ['BE', 'NL', 'Other', None]

# Luisterfrequentie mapping
listening_frequency_mapping = {
    'rarely': (1, 5),
    'sometimes': (5, 15),
    'often': (15, 25),
    'very often': (25, 40)
}

# Functie om gemiddelde luistertijd per week te berekenen
def get_avg_hours_per_week(frequency):
    return np.random.uniform(*listening_frequency_mapping[frequency])

# Genereer gebruikersdata
num_users = 100
users = []
for i in range(num_users):
    persona = np.random.choice(personas, p=[p["share"] for p in personas])
    min_age, max_age = persona["age_group"]
    age = np.random.randint(min_age, max_age + 1)
    gender = np.random.choice(gender_options, p=[0.45, 0.45, 0.10])
    location = np.random.choice(location_options, p=[0.25, 0.50, 0.10, 0.15])
    preferred_genre_cluster = np.random.choice(persona["preferred_clusters"])
    preferred_genre = np.random.choice(genre_clusters[preferred_genre_cluster])
    listen_freq = persona["listening_frequency"]
    avg_hours_per_week = get_avg_hours_per_week(listen_freq)

    buddy_consent = np.random.choice([0, 1], p=[0.25, 0.75])
    users.append([i + 1, persona["name"], age, gender, location, preferred_genre_cluster, 
                  preferred_genre, listen_freq, avg_hours_per_week, buddy_consent])

# Maak DataFrame met gebruikersdata
users_df = pd.DataFrame(users, columns=['user_id', 'persona', 'age', 'gender', 'location', 
                                        'preferred_genre_cluster', 'preferred_genre', 'listen_freq', 
                                        'avg_hours_per_week', 'buddy_consent'])

#definieer popularity weight
tracks['popularity_weight'] = tracks['popularity'] / 100


# Simuleer sessies en trackselectie
sessions = []
listening_data = []

for _, user in users_df.iterrows():
    user_id = user['user_id']
    avg_hours = user['avg_hours_per_week']
    total_tracks = int(avg_hours * 17)  # 17 tracks per uur
    preferred_cluster = user['preferred_genre_cluster']
    preferred_genre = user['preferred_genre']
    
    # Tracks selecteren met gewogen kans op basis van populariteit
    # Zorg ervoor dat de populariteit gewichten correct zijn

    # Selecteer tracks op basis van de voorkeuren
    preferred_tracks = tracks[tracks['genre_cluster'] == preferred_cluster]
    preferred_genre_tracks = preferred_tracks[preferred_tracks['track_genre'] == preferred_genre]
    other_tracks = tracks[tracks['genre_cluster'] != preferred_cluster]

    num_preferred_genre = int(total_tracks * 0.5)
    num_preferred_cluster = int(total_tracks * 0.3)
    num_other = total_tracks - (num_preferred_genre + num_preferred_cluster)

    selected_tracks = []

    if not preferred_genre_tracks.empty:
        selected_tracks.extend(preferred_genre_tracks.sample(n=num_preferred_genre, weights='popularity_weight', replace=True).to_dict('records'))
    if not preferred_tracks.empty:
        selected_tracks.extend(preferred_tracks.sample(n=num_preferred_cluster, weights='popularity_weight', replace=True).to_dict('records'))
    if not other_tracks.empty:
        selected_tracks.extend(other_tracks.sample(n=num_other, weights='popularity_weight', replace=True).to_dict('records'))

    # Debug: Controleer hoeveel tracks zijn geselecteerd
    if len(selected_tracks) == 0:
        print(f"No tracks selected for user {user_id}")
        continue  # Skip gebruiker als er geen tracks geselecteerd zijn
    
    # Genereer sessies
    session_id = 1
    start_date = datetime(2025, 3, 1)

    while total_tracks > 0 and selected_tracks:
        session_tracks = np.random.randint(3, 15)  # Willekeurig aantal tracks per sessie
        session_tracks = min(session_tracks, total_tracks)
        total_tracks -= session_tracks
        
        session_time = start_date + timedelta(days=np.random.randint(0, 30), 
                                              hours=np.random.randint(0, 24))
        
        sessions.append([user_id, session_id, session_time])
        
        for _ in range(session_tracks):
            track = np.random.choice(selected_tracks)
            like = np.random.choice([1, 0, -1], p=[0.6, 0.3, 0.1])
            
            # Calculate the duration listened
            duration_listened = int(skewnorm.rvs(a=10 if like == 1 else -10 if like == -1 else 0, 
                                                 loc=0, scale=track['duration_ms']))
            duration_listened = np.clip(duration_listened, 0, track['duration_ms'])
            
            # Calculate the completion rate
            completion_rate = duration_listened / track['duration_ms']
            
            # Append the data to listening_data
            listening_data.append([user_id, session_id, track['track_id'], track['track_name'], track['duration_ms'], 
                                   duration_listened, completion_rate, like, session_time])
        
        session_id += 1

# Zet data om in DataFrames
#sessions_df = pd.DataFrame(sessions, columns=['user_id', 'session_id', 'session_time'])
interactions_df = pd.DataFrame(listening_data, columns=['user_id', 'session_id', 'track_id', 'track_name', 
                                                      'duration_ms', 'duration_listened', 'completion_rate', 'like','session_time'])


print(interactions_df.head(20))

# Merge user_df en listening_df op 'user_id'
final_df = pd.merge(interactions_df, users_df, on='user_id', how='left')

# Merge het resultaat met tracks_df op 'track_id'
final_df = pd.merge(final_df, tracks, on='track_id', how='left')

# Drop de dubbele track_name kolom
final_df = final_df.drop(columns=['track_name_y'])

# Verwijder '_x' uit de kolomnamen
final_df.columns = [col.replace('_x', '') for col in final_df.columns]

print(final_df.columns)  # Check of het correct is aangepast

#writing the final_df to a csv file
#give it name synthetic_user_data.csv
final_df.to_csv('synthetic_user_data.csv', index=False)






#######################################################################
# Quick Check
# Bekijk het resultaat
print(final_df.head(20))
print(final_df.columns)
print(final_df['time_signature'].head())
#total numbers of interactions
total_interactions = final_df.shape[0]
print(f'Total number of user interactions: {total_interactions}')

# plot histogram of number of tracks per user, sorted from highest to lowest using seaborn
num_tracks_per_user = final_df['user_id'].value_counts().sort_values(ascending=False)

# Plot histogram of number of total tracks per user
sns.histplot(num_tracks_per_user, bins=30, kde=True)
plt.xlabel('Number of Total Tracks per User')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Total Tracks per User')
plt.show()

# Plot histogram of number of unique tracks per user
unique_tracks_per_user = final_df.groupby('user_id')['track_id'].nunique()

sns.histplot(unique_tracks_per_user, bins=30, kde=True)
plt.xlabel('Number of Unique Tracks per User')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Unique Tracks per User')
plt.show()

#show range of popularity in tracks
popularity_range = final_df['popularity'].max() - final_df['popularity'].min()


num_genres = final_df['track_genre'].nunique()
print(f'number of genres: {num_genres}')

# number of tracks
num_tracks = final_df['track_id'].nunique()
print(f'number of tracks: {num_tracks}')

#total number of tracks in final_df
total_tracks = final_df.shape[0]
print(f'Total number of tracks: {total_tracks}')


# number of artists
num_artists = final_df['artists'].nunique()
print(f'number of artists: {num_artists}')

#number of artists in tracks
num_artists_tracks = tracks['artists'].nunique()
print(f'number of artists in tracks: {num_artists_tracks}')

# number of albums
num_albums = final_df['album_name'].nunique()
print(f'number of albums: {num_albums}')

# number of tracks per genre
tracks_per_genre = final_df['track_genre'].value_counts()
print(f'tracks per genre: {tracks_per_genre}')

#average popularity in tracks of classical

#show for each genre the average popularity and the populrity weight in tracks
average_popularity = final_df.groupby('track_genre')['popularity'].mean()
print(f'average popularity per genre: {average_popularity}')

# # plot within this plot the number of tracks per genre in final_df
# plt.figure(figsize=(10, 6))
# plt.bar(average_popularity.index, average_popularity.values, alpha=0.5, label='Average Popularity')
# plt.bar(tracks_per_genre.index, tracks_per_genre.values, alpha=0.5, label='Number of Tracks')
# plt.xlabel('Genre')
# plt.ylabel('Count')
# plt.title('Average Popularity and Number of Tracks per Genre')
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.show()

# popularity weight
popularity_weight = final_df.groupby('track_genre')['popularity_weight'].mean()
print(f'popularity weight per genre: {popularity_weight}')

# number of tracks per artist
tracks_per_artist = final_df['artists'].value_counts()
print(f'tracks per artist: {tracks_per_artist}')

# mean number of total tracks per user
mean_total_tracks = num_tracks_per_user.mean()
print(f'Mean number of total tracks per user: {mean_total_tracks:.2f}')

# mean number of unique tracks per user
mean_unique_tracks = unique_tracks_per_user.mean()
print(f'Mean number of unique tracks per user: {mean_unique_tracks:.2f}')

#  mean number of replicate tracks grouped by user
mean_replicate_tracks = final_df.groupby(['user_id', 'track_id']).size().mean()
print(f'Mean number of replicate tracks per user: {mean_replicate_tracks:.2f}')

# number of users (users dataframe)

num_users = users['user_id'].nunique()
print(f'Number of unique users: {num_users}')

num_none_genders = users_df['gender'].isna().sum()
print(f"Aantal None in gender: {num_none_genders}")


# user age distribution
age_distribution = users_df['age'].value_counts(normalize=True)
print(age_distribution)
#plot age distribution
plt.figure(figsize=(10, 6))
plt.hist(users_df['age'], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Users')
plt.show()


# user gender distribution
gender_distribution = users_df['gender'].value_counts(normalize=True)
print(gender_distribution)

# check how many 'None' in gender


# users preferred genre distribution
preferred_genre_distribution = users['preferred_genre'].value_counts(normalize=True)
print(preferred_genre_distribution)

#final_df.to_csv('synthetic_user_data.csv', index=False)