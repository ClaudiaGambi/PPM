import pandas as pd
from shiny import App, module, ui, render, reactive, event, run_app
from shinywidgets import output_widget, render_widget
from shiny.ui import tags
import plotly.express as px
import plotly.graph_objects as go
from shiny import reactive, render_text
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import numpy as np
from shiny_app.functions import knn_module
from shiny_app.functions import get_most_similar_tracks
from shiny_app.functions import inverse_popularity
from shiny_app.functions import generate_recommended_tracks_list
from shiny_app.functions import filter_christmas_songs

# Predifined user id as our current user
user_id = 1

# User data
user_data = pd.read_csv(Path(__file__).parent / "data/synthetic_user_data.csv")

# Tracks data
tracks_data = pd.read_csv(Path(__file__).parent / "data/spotify_tracks_clean_clusters.csv")
tracks_data = tracks_data.sample(frac=0.1, random_state=42)

# Compute fixed axis ranges from the data (used for coordinate conversion)
x_min = tracks_data["valence"].min()
x_max = tracks_data["valence"].max()
y_min = tracks_data["energy"].min()
y_max = tracks_data["energy"].max()

# Reactive values for selected valence & energy
valence_selected = reactive.Value(0.5)
energy_selected = reactive.Value(0.5)

# Callback for when a marker is clicked directly on the figure
def on_point_click(trace, points, state):
    if points.point_inds:  # Only proceed if a point is clicked
        idx = points.point_inds[0]  # Get index of first clicked point
        valence = trace.x[idx]
        energy = trace.y[idx]
        valence_selected.set(valence)
        energy_selected.set(energy)
        print(f"Updated Valence: {valence}, Energy: {energy}")

# UI
ui = ui.page_fluid(

    ui.tags.head(
    ui.tags.style(
        """
        body {
            background-color: #0b1c36;
            color: #ffffff;
        }

        .form-control, .selectize-input {
            background-color: ##ffffff !important;
            color: #ffffff !important;
            border: 0px solid #333;
        }

        .selectize-dropdown-content {
            background-color: #1e3352 !important;
            color: #ffffff !important;


        .shiny-input-container {
            margin-bottom: 1rem;
        }

        .widget-output, .plotly {
            background-color: #1e1e1e !important;
        }
        """
    )),

    ui.h2("NPO Luister"),

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
    # Custom JS: Attach a click listener on the plot element.
    # It converts the click position (using fixed margins and dimensions)
    # into data coordinates and sends them to the Shiny server.
    ui.tags.script(f"""
        function attachPlotClick() {{
            var plotEl = document.getElementById("plot");
            if (plotEl) {{
                plotEl.addEventListener("click", function(event) {{
                    // These values must match those used in the plot layout:
                    var left_margin = 50;
                    var top_margin = 50;
                    var total_width = 800;
                    var total_height = 800;
                    var inner_width = total_width - 50 - 50;  // left and right margins
                    var inner_height = total_height - 50 - 50; // top and bottom margins

                    // Convert the click's offset position (relative to the plot element)
                    // into data coordinates assuming a linear mapping.
                    var dataX = {x_min} + ((event.offsetX - left_margin) / inner_width) * ({x_max} - {x_min});
                    var dataY = {y_max} - ((event.offsetY - top_margin) / inner_height) * ({y_max} - {y_min});
                    Shiny.setInputValue("plot_click_any", {{x: dataX, y: dataY}}, {{priority: "event"}});
                }});
            }} else {{
                setTimeout(attachPlotClick, 500);
            }}
        }}
        attachPlotClick();
    """),

    ui.h2("Your recommendations:"),

    # output list tracks
    ui.output_ui("recommended_tracks_list"),

)

# SERVER
def server(input, output, session):

    # REACTIVE FUNCTION: Filters Data Based on dropdown menu Selection
    @reactive.Calc
    def filtered_data():

        data = tracks_data.copy()
        selected_genres = input.genre_filter()

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

        return data
    
    @render_widget
    def plot():
        # Create a Plotly Express scatter plot with fixed dimensions and margins.
        scatterplot = px.scatter(
            filtered_data(),
            x="valence",
            y="energy",
            hover_data=["track_name", "artists", "album_name"]
        ).update_traces(
            marker=dict(size=10, color="blue", opacity=0.7)
        ).update_layout(
            title="Valence vs. Energy",
            yaxis_title="Energy",
            xaxis_title="Valence",
            width=800,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50),
            clickmode="event+select"
        )
        # Convert the figure to a FigureWidget so we can attach Python callbacks.
        w = go.FigureWidget(scatterplot.data, scatterplot.layout)
        # Attach the on_click callback to capture direct clicks on markers.
        w.data[0].on_click(on_point_click)
        return w

    # React to any click (even if not on a marker) by selecting the nearest point.
    @reactive.Effect
    def update_nearest_point():
        data = input.plot_click_any()
        if data is not None:
            df = filtered_data()
            # Compute the Euclidean distance from the click to each point.
            distances = ((df["valence"] - data["x"])**2 + (df["energy"] - data["y"])**2)**0.5
            nearest_idx = distances.idxmin()
            nearest_valence = df.loc[nearest_idx, "valence"]
            nearest_energy = df.loc[nearest_idx, "energy"]
            valence_selected.set(nearest_valence)
            energy_selected.set(nearest_energy)
            print(f"Nearest point selected: Valence: {nearest_valence}, Energy: {nearest_energy}")


    # REACTIVE FUNCTION: Get KNN recommendations based on slider values
    @reactive.Calc
    def recommended_tracks(top_n=5):

        # Get valence value from nearest point
        valence = valence_selected.get()

        # Get energy value from nearest point
        energy = energy_selected.get()

        # Call your KNN function with updated values for first selection
        NN = knn_module(tracks_data, valence, energy)
        
        #Christmas filter
        NC = filter_christmas_songs(NN)

        # Only recommend tracks that are somewhat similar to tracks history
        SIM = get_most_similar_tracks(NC, user_data, user_id=1)
        
        # Apply inverse popularity filter
        return inverse_popularity(SIM, top_n)

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
            marker=dict(size=10, color="white", opacity=1.0)
            # hovertemplate="<b>Song:</b> %{customdata[0]}<br><b>Artist:</b> %{customdata[1]}<br><b>Album:</b> %{customdata[2]}<extra></extra>"
        ).update_layout(
            title={"text": "Valence vs. Energy", "font": {"color": "white"}},
            yaxis_title="Energy",
            xaxis_title="Valence",
            width=800,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50),
            clickmode="event+select",
            plot_bgcolor="#0b1c36",
            paper_bgcolor="#0b1c36",
            font={"color": "white"},
            xaxis=dict(color="white"),
            yaxis=dict(color="white"),
        )

        return scatterplot

    @render.ui
    def recommended_tracks_list():
        tracks = recommended_tracks()  # Get recommended tracks
        return generate_recommended_tracks_list(tracks)


# run app
app = App(ui, server)

if __name__ == "__main__":
    run_app(app)
