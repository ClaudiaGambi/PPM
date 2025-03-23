import pandas as pd
from shiny import module, ui, render, reactive, event, run_app
from shinywidgets import output_widget, render_widget
import plotly.express as px
from shiny import reactive, render_text
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np

data = pd.read_csv(Path.cwd() / "streamlit_app" / "shiny_app" / "spotify_tracks_clean.csv")
#data = pd.read_csv(Path(__file__).parent / "spotify_tracks_clean.csv")
data = data.sample(frac=0.1, random_state=42)

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def knn_module(valence=0.5, energy=0.5, n=20):
    # Get user-selected values
    valence_target = valence
    energy_target = energy

    # Extract relevant features for KNN
    feature_data = data[["valence", "energy"]].to_numpy()

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=1000, metric='euclidean')
    knn.fit(feature_data)

    # Find 1000 nearest neighbors to user-selected values
    distances, indices = knn.kneighbors(np.array([[valence_target, energy_target]]))

    # Use the indices to get the nearest neighbors from the original DataFrame
    nearest_neighbors = data.iloc[indices[0]]  # Indices is an array, so use [0] to get the correct row selection


    ### (ILSE) Het lijkt mij goed om hier van 1000 knn naar 200 nummers te komen die matchen met luistergeschiedenis.
    # ik heb daarvoor een script geschreven user_track_filtering in data folder
    # get_most_similar_tracks(df_track, df_users, user_id, top_n=200)
    # maar ik weet niet hoe ik die hier kan aanroepen

    # Select 20 based on inverse popularity
    # Invert popularity for weighting (higher weight for lower popularity), ensuring all songs have a chance
    popularity = nearest_neighbors['popularity'].values
    weights = ((100 - popularity) + 1) / np.sum((100 - popularity) + 1)  # Normalize

    # Select n songs using weighted random sampling
    selected_indices = np.random.choice(nearest_neighbors.index, size=n, replace=False, p=weights)
    selected_songs = data.loc[selected_indices, ['artists', 'track_name', 'album_name']]

    return selected_songs.reset_index(drop=True)


knn_module(0.5, 0.5, 20)

