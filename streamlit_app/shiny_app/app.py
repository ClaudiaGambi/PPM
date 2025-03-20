import pandas as pd
from shiny import App, module, ui, render, reactive, event, run_app
from shinywidgets import output_widget, render_widget
import plotly.express as px
from shiny import reactive, render_text
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np
from streamlit_app.shiny_app.counter import knn_module

# data
data = pd.read_csv(Path(__file__).parent / "spotify_tracks_clean.csv")
data = data.sample(frac=0.1, random_state=42)

# UI
ui = ui.page_fluid(
    ui.h2("Spotify Track Analysis"),

    # Dropdown to select an artist
    ui.input_selectize("genre_filter", "Select Genre:",
                    choices=["All"] + sorted(data["track_genre"].unique().tolist()), multiple=True),
    output_widget("plot"),

    ui.h2("Sliders for Energy and Danceability"),

    # Add sliders below the plot
    ui.input_slider("danceability_filter", "Danceability", min=0.0, max=1.0, value=0.5, step=0.01),
    ui.input_slider("energy_filter", "Energy", min=0.0, max=1.0, value=0.5, step=0.01),

)

# SERVER
def server(input, output, session):

    # REACTIVE FUNCTION: Filters Data Based on dropdown menu Selection
    @reactive.Calc
    def filtered_data():
        selected_genre = input.genre_filter()

        # Ensure it's always a list
        if isinstance(selected_genre, str):
            selected_genre = [selected_genre]

        # If no genre selected, show all data
        if not selected_genre:
            return data

        return data[data["track_genre"].isin(selected_genre)]

    # REACTIVE FUNCTION: Get KNN recommendations based on slider values
    @reactive.Calc
    def recommended_tracks():
        dance = input.danceability_filter()  # Get danceability value from slider
        print(dance)
        energy = input.energy_filter()  # Get energy value from slider
        print(energy)

        return knn_module(dance, energy)  # Call your KNN function with updated values

    # PLOT FUNCTION
    @render_widget
    def plot():
        scatterplot = px.scatter(

            # Call the filtered data function to only display the correct data
            data_frame=filtered_data(),
            x="danceability",
            y="energy",
            hover_data=["track_name", "artists", "album_name"]
        ).update_traces(
            hovertemplate="<b>Song:</b> %{customdata[0]}<br><b>Artist:</b> %{customdata[1]}<br><b>Album:</b> %{customdata[2]}<extra></extra>"
        ).update_layout(
            title={"text": "Danceability vs. Energy"},
            yaxis_title="Energy",
            xaxis_title="Danceability",
        )

        return scatterplot

    # PRINT RECOMMENDED TRACKS TO CONSOLE (FOR DEBUGGING)
    @reactive.Effect
    def _():
        print(recommended_tracks())


# run app
app = App(ui, server)

if __name__ == "__main__":
    run_app(app)
