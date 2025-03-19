import pandas as pd
from shiny import App, ui
from shiny import run_app
from shinywidgets import output_widget, render_widget
import plotly.express as px
from shiny import reactive, render_text
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np

# data
data = pd.read_csv(Path(__file__).parent / "spotify_tracks_clean.csv")
data = data.sample(frac=0.1, random_state=42)

# UI
ui = ui.page_fluid(
    ui.h2("Spotify Track Analysis"),

    # Dropdown to select an artist
    ui.input_selectize("artist_filter", "Select Artist:",
                    choices=["All"] + sorted(data["artists"].unique().tolist()), multiple=True),
    output_widget("plot"),

    ui.h2("Sliders for Energy and Danceability"),

    # Add sliders below the plot
    ui.input_slider("danceability_filter", "Danceability Range", min=0.0, max=1.0, value=(0.0, 1.0), step=0.01),
    ui.input_slider("energy_filter", "Energy Range", min=0.0, max=1.0, value=(0.0, 1.0), step=0.01),

)

# SERVER
def server(input, output, session):

    #REACTIVE FUNCTION: Filters Data Based on dropdown menu Selection
    @reactive.Calc
    def filtered_data():
        selected_artists = input.artist_filter()

        # Ensure it's always a list
        if isinstance(selected_artists, str):
            selected_artists = [selected_artists]

        # If no artist selected, show all data
        if not selected_artists:
            return data

        return data[data["artists"].isin(selected_artists)]

    @reactive.Calc
    def knn_tracks():

        # Get user-selected values
        dance_target = input.danceability_filter()
        energy_target = input.energy_filter()

        # Extract relevant features for KNN
        feature_data = data[["danceability", "energy"]].to_numpy()

        # Fit KNN model
        knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
        knn.fit(feature_data)

        # Find 20 nearest neighbors to user-selected values
        distances, indices = knn.kneighbors(np.array([[dance_target, energy_target]]))

        # Return DataFrame with selected tracks
        return data.iloc[indices[0]][["track_name", "album_name", "artists"]]


    # plot
    @render_widget
    def plot():
        scatterplot = px.scatter(

            # call the filtered data function to only display the correct data
            data_frame=filtered_data(),
            x = "danceability",
            y = "energy",
            hover_data = ["track_name", "artists", "album_name"]
        ).update_traces(
            hovertemplate = "<b>Song:</b> %{customdata[0]}<br><b>Artist:</b> %{customdata[1]}<br><b>Album:</b> %{customdata[2]}<extra></extra>"
        ).update_layout(
            title = {"text": "Danceability vs. Energy"},
            yaxis_title = "Energy",
            xaxis_title = "Danceability",
        )

        return scatterplot

    # @render.data_table
    # def knn_output():
    #     return knn_tracks()

# run app
app = App(ui, server)

if __name__ == "__main__":
    run_app(app)
