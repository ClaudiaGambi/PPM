import pandas as pd
from shiny import module, ui, render, reactive, event, run_app
from shinywidgets import output_widget, render_widget
import plotly.express as px
from shiny import reactive, render_text
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np

data = pd.read_csv(Path(__file__).parent / "spotify_tracks_clean.csv")
data = data.sample(frac=0.1, random_state=42)

def knn_module(dance = 0.5, energy = 0.5):

    # Get user-selected values
    dance_target = dance
    energy_target = energy

    # Extract relevant features for KNN
    feature_data = data[["danceability", "energy"]].to_numpy()

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
    knn.fit(feature_data)

    # Find 20 nearest neighbors to user-selected values
    distances, indices = knn.kneighbors(np.array([[dance_target, energy_target]]))

    # check plot
    print(data.iloc[indices[0]][["track_name", "album_name", "artists"]])

    # Return DataFrame with selected tracks
    return data.iloc[indices[0]][["track_name", "album_name", "artists"]]
