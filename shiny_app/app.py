import pandas as pd
from shiny import App, module, ui, render, reactive, event, run_app
from shinywidgets import output_widget, render_widget
import plotly.express as px
from shiny import reactive, render_text
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np
from shiny_app.functions import knn_module
from shiny_app.functions import get_most_similar_tracks


# Predifined user id as our current user
user_id = 1

# User data
user_data = pd.read_csv(Path(__file__).parent / "data/synthetic_user_data.csv")

# Tracks data
tracks_data = pd.read_csv(Path(__file__).parent / "data/spotify_tracks_clean_clusters.csv")
tracks_data = tracks_data.sample(frac=0.1, random_state=42)

# UI
ui = ui.page_fluid(

    ui.h2("Spotify Track Analysis"),

    # Dropdown to select a genre group
    ui.input_selectize("genre_cluster_filter", "Select Genre group:",
                    choices=["All"] + sorted(tracks_data["genre_cluster"].unique().tolist()), multiple=True),

    # Dropdown to select a specific genre
    ui.input_selectize("genre_filter", "Select Genre:",
                    choices=["All"] + sorted(tracks_data["track_genre"].unique().tolist()), multiple=True),

    # Dropdown to select a specific artist
    ui.input_selectize("artist_filter", "Select Artist:",
                    choices=["All"] + sorted(tracks_data["artists"].unique().tolist()), multiple=True),

    output_widget("plot"),

    ui.h2("Sliders for Energy and Valence"),

    # Add sliders below the plot
    ui.input_slider("valence_filter", "Valence", min=0.0, max=1.0, value=0, step=0.01),
    ui.input_slider("energy_filter", "Energy", min=0.0, max=1.0, value=0, step=0.01),
)

# SERVER
def server(input, output, session):

    # REACTIVE FUNCTION: Filters Data Based on dropdown menu Selection
    @reactive.Calc
    def filtered_data():

        data = tracks_data.copy()
        selected_genre = input.genre_filter()

        # Genre cluster filter
        selected_clusters = input.genre_cluster_filter()
        if isinstance(selected_clusters, str):
            selected_clusters = [selected_clusters]
        if selected_clusters and "All" not in selected_clusters:
            data = data[data["genre_cluster"].isin(selected_clusters)]

        # Genre filter
        selected_genres = input.genre_filter()
        if isinstance(selected_genres, str):
            selected_genres = [selected_genres]
        if selected_genres and "All" not in selected_genres:
            data = data[data["track_genre"].isin(selected_genres)]

        # Artist filter
        selected_artists = input.artist_filter()
        if isinstance(selected_artists, str):
            selected_artists = [selected_artists]
        if selected_artists and "All" not in selected_artists:
            data = data[data["artists"].isin(selected_artists)]


        # Valence and energy filters
        valence = input.valence_filter()
        energy = input.energy_filter()
        data = data[(data["valence"] >= valence) & (data["energy"] >= energy)]

        return data


    # REACTIVE FUNCTION: Get KNN recommendations based on slider values
    @reactive.Calc
    def recommended_tracks():

        # Get valence value from slider
        valence = input.valence_filter()

        # Get energy value from slider
        energy = input.energy_filter()

        # Call your KNN function with updated values for first selection
        NN = knn_module(valence, energy)

        # Only recommend tracks that are somewhat similar to tracks history
        return get_most_similar_tracks(NN, user_data, user_id)

    # PLOT FUNCTION
    @render_widget
    def plot():
        scatterplot = px.scatter(

            # Call the filtered data function to only display the correct data
            data_frame=filtered_data(),
            x="valence",
            y="energy",
            hover_data=["track_name", "artists", "album_name"]
        ).update_traces(
            hovertemplate="<b>Song:</b> %{customdata[0]}<br><b>Artist:</b> %{customdata[1]}<br><b>Album:</b> %{customdata[2]}<extra></extra>"
        ).update_layout(
            title={"text": "Valence vs. Energy"},
            yaxis_title="Energy",
            xaxis_title="Valence",
        )

        return scatterplot

    # PRINT RECOMMENDED TRACKS TO CONSOLE
    @reactive.Effect
    def _():
        print(recommended_tracks())


# run app
app = App(ui, server)

if __name__ == "__main__":
    run_app(app)
