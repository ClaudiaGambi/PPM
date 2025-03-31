
import pandas as pd
import numpy as np
from shiny import App, module, ui, render, reactive, event, run_app
from shinywidgets import output_widget, render_widget
from shiny.ui import tags
import plotly.express as px
import plotly.graph_objects as go
from shiny import reactive, render_text
from shiny.types import ImgData
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances
from shiny_app.functions import knn_module
from shiny_app.functions import get_most_similar_tracks
from shiny_app.functions import inverse_popularity
from shiny_app.functions import generate_recommended_tracks_list
from shiny_app.functions import buddy_recommendations
from shiny_app.functions import build_faiss_index
from shiny_app.functions import recommend_similar_tracks_audio_ft
from shiny_app.functions import on_point_click
from shiny_app.functions import tracks_data # import data from functions.py
from shiny_app.functions import user_data # import data from functions.py

# audio_features = ['danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# build FAISS index from tracks_data and user_data
tracks_faiss = build_faiss_index(tracks_data, audio_features, name='faiss_tracks')
user_faiss = build_faiss_index(user_data, audio_features, name='faiss_users')

# Compute fixed axis ranges from the data (used for coordinate conversion)
x_min = tracks_data["valence"].min()
x_max = tracks_data["valence"].max()
y_min = tracks_data["energy"].min()
y_max = tracks_data["energy"].max()

# Reactive values
valence_selected = reactive.Value(0.5)
energy_selected = reactive.Value(0.5)
recc_tracks_plot = reactive.Value(pd.DataFrame())
recc_tracks_buddy = reactive.Value(pd.DataFrame())
recc_tracks_track = reactive.Value(pd.DataFrame())
# user_id = reactive.Value(5)


# @reactive.Effect
# def print_user_id():
#     print(f'current user: {user_id.get()}')


# UI
ui = ui.page_fixed(

    ui.tags.head(
        ui.tags.style(
            """
            body {
                background-color: #0B223E;
                color: #ffffff;
                width: 100vw;
                height: 100vh;
                font-family: NPO Sans;
            }

            .container-fluid {
                padding: 0px;
            }

            .row.title-bar {
                padding: 15px;
                display: flex;
                height: 10hv;
            }

            /* Style for all numeric input fields */
            .form-control, .selectize-input, .shiny-input-container input[type="number"] {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ffffff;
                padding: 8px;
                border-radius: 5px;
                font-size: 14px;
            }

            /* Style for all output text boxes */
            .shiny-output-text-verbatim, .shiny-text-output {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            }

            .selectize-dropdown-content {
                background-color: #1e3352 !important;
                color: #ffffff !important;
            }

            .shiny-input-container {
                margin-bottom: 1rem;
            }

            .widget-output, .plotly {
                background-color: #1e1e1e !important;
            }
            """
        )),

    # ROW 1 (title)
    ui.row(
        ui.column(6, ui.output_image("npo_logo", width="auto", height="auto")),
        # ui.column(6, ui.h1("NPO Music")),
        class_="title-bar"),

    # Row 2 (description)
    ui.row(ui.h1("Explore new artists", class_='mb-5'),
           ui.h2("Select your favorite tracks based on energy and valence "),
            class_ = "description"),

    # ROW 3 (filter + plot + recommendations)
    ui.row(
        ui.column(3,

                  # text
                   ui.p('You can select certain genre clusters, specific genres, or no filter at all.\n'),
                         ui.p("Valence: the musical positiveness conveyed by a track"),
                         ui.p("Energy: a perceptual measure of intensity and activity"),

                  # Dropdown to select a genre group
                  ui.input_selectize("genre_cluster_filter", "Select Genre cluster:",
                                     choices=["All"] + sorted(tracks_data["genre_cluster"].unique().tolist()),
                                     multiple=True),

                  # Dropdown to select a specific genre
                  ui.input_selectize("genre_filter", "Select Genre:",
                                     choices=["All"] + sorted(tracks_data["track_genre"].unique().tolist()),
                                     multiple=True),

                  # Dropdown to select a specific artist
                  ui.input_selectize("artist_filter", "Select Artist:",
                                     choices=["All"] + sorted(tracks_data["artists"].unique().tolist()),
                                     multiple=True),

                  # Slider for diversity
                  ui.input_slider("slider_diversity", "Select diversity level:",
                                  min=1, max=3, step=1, value=2),

                  # column class
                  class_="select-menu"),

            ui.column(5,

                      # plot widget
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
                            var total_width = 570;
                            var total_height = 570;
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

                # column class
                class_="plot-container"),


            ui.column(4,
                ui.h2("Your recommendations:"),

                    # output list tracks
                     ui.output_ui("recommended_tracks_plot"),

                    # column class
                    class_="recommendations-column"
            ),
        # row class
        class_ = "filter-plot-recommendations mb-5"),

        #ROW 4 (buddy recommendations)
    ui.row(
    ui.column(4,
            ui.h2("...or select a track from your buddy's playlist"),

            # HET SELECTEREN VAN USER ID NIET IN DE INTERFACE, LIVER COMMANDLINE COMMAND OF NIET
            # ui.input_numeric("user_id", "User ID", 1, min=1, max=max(user_data["user_id"])),
            # ui.output_text_verbatim("value"),
            ui.output_image("clickable_img", inline=True),
            ui.tags.script(f"""
            function attachImageClick() {{
                var imgEl = document.getElementById("clickable_img");
                if (imgEl) {{
                    imgEl.addEventListener("click", function() {{
                        Shiny.setInputValue("img_clicked", Math.random(), {{priority: "event"}});
                    }});
                }}   else {{
                    setTimeout(attachImageClick, 500);
                }}
            }}
            attachImageClick();
            """),
            class_="image-column"),
        ui.column(8,
                  ui.h2("Your buddy's recommendations:"),
            ui.output_ui("recommended_tracks_list_buddy"),
            class_="buddy_recommendations-column"),
        # row class
        class_="buddy-recommendations mb-5"),

    # ROW 5 (track selection)
    ui.row(
        ui.column(4,
            ui.h2('...or select a track from the catalog'),
            ui.input_selectize(
                "track_selection", "Search for a Track:",
                choices=["Select a track"] + sorted(
                    tracks_data.fillna("").apply(lambda row: f"{row['track_name']} - {row['artists']} ({row['album_name']})", axis=1).unique().tolist()
                ),
                multiple=False,
                options={"create": True} ), # Allows free typing with autocomplete
            # column class
            class_="recommendations-column"),
        ui.column(8,
               ui.h2("Your recommendations based on selected track:"),
            ui.output_ui("recommended_tracks_track"),
            class_="track_recommendations-column"),
        # row class
        class_="track-recommendation mb-5"),

)

# SERVER
def server(input, output, session):

    # logo
    @render.image
    def npo_logo():
        dir = Path(__file__).resolve().parent
        logo: ImgData = {"src": str(dir / "static/NPO_logo.png"), "width": "auto", "height": "auto"}
        return logo

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

    # debug function to print selected categories to console
    @reactive.Effect
    def print_selected_categories():
        print(f"Selected genres: {input.genre_filter()}")
        print(f"Selected genre clusters: {input.genre_cluster_filter()}")
        print(f"Selected artists: {input.artist_filter()}")

    @render_widget
    def plot():
        # Create a Plotly Express scatter plot with fixed dimensions and margins.
        scatterplot = px.scatter(
            filtered_data(),
            x="valence",
            y="energy",
            hover_data=["track_name", "artists", "album_name", "track_genre"]
        ).update_traces(
            marker=dict(size=10, color="blue", opacity=0.7)
        ).update_layout(
            title="Valence vs. Energy",
            yaxis_title="Energy",
            xaxis_title="Valence",
            width=570,
            height=570,
            margin=dict(l=50, r=50, t=50, b=50),
            clickmode="event+select"
        )
        # Convert the figure to a FigureWidget, so we can attach Python callbacks.
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
            # Reset track selection dropdown
            session.send_input_message("track_selection", {"value": "Select a track"})

    # track selection
    @reactive.Calc
    def selected_track():
        track_info = input.track_selection()
        print(f'Selected track: {track_info}')

        if not track_info or track_info == "Select a track":
            return None  # No track selected

        # Extract track name and artist from the selection
        track_name, artist_album = track_info.split(" - ", 1)
        artist, album = artist_album.rsplit(" (", 1)
        album = album.rstrip(")")

        # Find the track in the dataset
        track_row = tracks_data[
            (tracks_data["track_name"] == track_name) &
            (tracks_data["artists"] == artist) &
            (tracks_data["album_name"] == album)
            ]

        if track_row.empty:
            print(f"WARNING: Selected track '{track_info}' not found in dataset!", flush=True)
            return None

        return track_row.iloc[0]  # Return the selected track as a Series

    @reactive.Effect
    @reactive.event(input.track_selection)  # Runs only when a track is selected
    def execute_function_on_track_selection():
        track = selected_track()

        if track is None:
            print("ERROR: No valid track selected.", flush=True)
            return  # Stop execution if no valid track is found

        track_id = track["track_id"]  # Extract track_id
        print(f"catalogue track selected: {track_id}", flush=True)

        # Get recommendations using FAISS
        track_sel_rec = recommend_similar_tracks_audio_ft(
            track_id, tracks_data, tracks_faiss[0], tracks_faiss[1], audio_features, num_recommendations=5
        )

        recc_tracks_track.set(track_sel_rec)

    # calculate diversity
    def calculate_diversity(recommended_tracks):
        feature_cols = ["valence", "energy"]
        feature_matrix = recommended_tracks[feature_cols].to_numpy()

        if len(feature_matrix) < 2:
            return 0

        # compute pairwise Euclidean distance
        pairwise_distances = euclidean_distances(feature_matrix)
        upper_triangle_values = pairwise_distances[np.triu_indices(len(pairwise_distances), k=1)]
        diversity_score = np.mean(upper_triangle_values) if len(upper_triangle_values) > 0 else 0

        return diversity_score, pairwise_distances

    # REACTIVE FUNCTION: Get KNN recommendations based on valence and energy values
    @reactive.Effect
    def recommended_tracks(top_n=5):
        # current_user_id = user_id.get()
        current_user_id = 1

        # get slider input
        slider_value = int(input.slider_diversity())
        k_value = {1: 5, 2: 50, 3: 500}[slider_value]

        # Get valence and energy values from the nearest point
        valence = valence_selected.get()
        energy = energy_selected.get()

        # Call KNN function with updated values
        nn = knn_module(filtered_data(), valence, energy)

        # Get similar tracks based on KNN results
        sim = get_most_similar_tracks(nn, user_data, current_user_id)

        # Check if sim is empty before applying inverse_popularity
        if sim.empty:
            print("WARNING: No similar tracks found. No recommendations available.", flush=True)
            return  # Stop execution if no recommendations are found

        # calculate diversity
        diversity_score, pairwise_distances = calculate_diversity(sim)
        diversity_score = diversity_score / np.max(pairwise_distances)

        # Apply inverse popularity filter
        inv_pop = inverse_popularity(sim, top_n)

        # Ensure inv_pop is not empty before updating recc_tracks
        if inv_pop.empty:
            print("WARNING: No recommendations after applying inverse popularity filter.", flush=True)
            return  # Stop execution if no valid recommendations

        # Update reactive value
        recc_tracks_plot.set(inv_pop)
        recc_tracks_track.set(inv_pop)
        print(f"Updated recommendations using 'recommended_tracks' with {len(inv_pop)} tracks", flush=True)

    # PLOT FUNCTION
    @render_widget
    def plot():
        scatterplot = px.scatter(

            # Call the filtered data function to only display the correct data
            data_frame=filtered_data(),
            x="valence",
            y="energy",
            custom_data=[
                filtered_data()["valence"].apply(lambda v: "Negative" if v <= 0.33 else "Neutral" if v <= 0.66 else "Positive"),
                filtered_data()["energy"].apply(lambda e: "Low" if e <= 0.33 else "Moderate" if e <= 0.66 else "High"),
                filtered_data()["track_name"],
                filtered_data()["artists"],
                filtered_data()["album_name"],
                filtered_data()["track_genre"],
        ]).update_traces(
            marker=dict(size=8, color="#DEDEDE", opacity=0.6),
            hovertemplate="<b>Valence:</b> %{customdata[0]}<br>"
                          "<b>Energy:</b> %{customdata[1]}<br>"
                          "<b>Song:</b> %{customdata[2]}<br>"
                          "<b>Artist:</b> %{customdata[3]}<br>"
                          "<b>Album:</b> %{customdata[4]}<br>"
                          "<b>Genre:</b> %{customdata[5]}<br>"
        ).update_layout(
            title={
                "text": "\nSelect your favorite tracks based on energy and valence to get recommendations!",
                "font": {"size": 12, "color": "white"},
                "x": 0.5,
                "y": 0.98},
            yaxis_title="Energy",
            xaxis_title="Valence",
            width=570,
            height=570,
            margin=dict(l=50, r=50, t=50, b=50),
            clickmode="event+select",
            plot_bgcolor="#0b1c36",
            paper_bgcolor="#0b1c36",
            font={"color": "white"},
            xaxis=dict(
                title=dict(text="<b>Valence</b>", font=dict(size=16)),
                color="white",
                ticklen=0,
                tickvals=[0, 0.5, 1],
                ticktext=["Negative", "Neutral", "Positive"]),
            yaxis=dict(
                title=dict(text="<b>Energy</b>", font=dict(size=16)),
                color="white",
                ticklen=0,
                tickvals=[0, 0.5, 1],
                ticktext=["Low", "Moderate", "High"])
        )

        return scatterplot

    # @reactive.Effect
    # def update_user_id():
    #     new_user_id = input.user_id()  # Get the new value from the UI

    #     # Validate input: Ensure it's a valid user ID
    #     if new_user_id is None or not isinstance(new_user_id, (int, float)):
    #         print("ERROR: Invalid input! User ID must be a number.", flush=True)
    #         return  # Do not update user_id

    #     if new_user_id < 1 or new_user_id > user_data["user_id"].max():
    #         print(f"ERROR: User ID {new_user_id} is out of range!", flush=True)
    #         return  # Do not update user_id

    #     # Input is valid, update reactive value
    #     print(f"updated user: {new_user_id}", flush=True)
    #     user_id.set(int(new_user_id))  # Ensure it's stored as an integer

    # @render.text
    # def value():
    #     return f"Current User ID: {input.user_id()}"

    @render.image
    def clickable_img():
        # Return a dictionary with at least src and one of width/height.
        # "src" is relative to the app directory or the provided static_dir.
        dir = Path(__file__).resolve().parent
        return {
            "src": str(dir / "static/buddy_3.png"),  # Ensure that buddy_3.png is in your app directory (or a served static folder)
            "width": "200px",
            "height": "auto",
            "alt": "buddy image",
            "id": "clickable_img",
            "style": "cursor: pointer;"
        }

    # Reactive effect: Execute the external function when the image is clicked
    @reactive.Effect
    @reactive.event(input.img_clicked)
    def execute_function_on_image_click():
        print(f'image clicked', flush=True)
        current_user_id = 1
        buddy_rec = buddy_recommendations(current_user_id, user_data, num_recommendations=5)
        recc_tracks_buddy.set(buddy_rec)
        session.send_input_message("track_selection", {"value": "Select a track"})

    @render.ui
    def recommended_tracks_list_buddy():
        tracks = recc_tracks_buddy.get()  # Retrieve the DataFrame from the reactive value
        return generate_recommended_tracks_list(tracks)

    @render.ui
    def recommended_tracks_plot():
        tracks = recc_tracks_plot.get()  # Retrieve the DataFrame from the reactive value
        return generate_recommended_tracks_list(tracks)

    @render.ui
    def recommended_tracks_track():
        tracks = recc_tracks_track.get()  # Retrieve the DataFrame from the reactive value
        return generate_recommended_tracks_list(tracks)

# run app
app = App(ui, server, static_assets=Path(__file__).parent/"static")


if __name__ == "__main__":
    run_app(app)

# TODO: remove artist filter
# TODO logic genre&cluster filter
# TODO diversity slider (Rosalie)
# TODO tooltip explanation app
# TODO: privacy checkbox + modal
