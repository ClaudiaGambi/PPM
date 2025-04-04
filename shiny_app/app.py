
# import packages
import pandas as pd
import numpy as np
from shiny import App, ui, render, reactive, run_app
from shinywidgets import output_widget, render_widget
from shiny.ui import modal, tags, modal_show, modal_button
import plotly.express as px
import plotly.graph_objects as go
from shiny.types import ImgData
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances

# import function from functions.py file
from shiny_app.functions import knn_module
from shiny_app.functions import get_most_similar_tracks
from shiny_app.functions import inverse_popularity
from shiny_app.functions import generate_recommended_tracks_list
from shiny_app.functions import buddy_recommendations
from shiny_app.functions import build_faiss_index
from shiny_app.functions import recommend_similar_tracks_audio_ft
from shiny_app.functions import on_point_click
from shiny_app.functions import filter_christmas_songs
from shiny_app.functions import tracks_data
from shiny_app.functions import user_data

# save audio features to list
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# build FAISS index from tracks_data and user_data
tracks_faiss = build_faiss_index(tracks_data, audio_features, name='faiss_tracks')
user_faiss = build_faiss_index(user_data, audio_features, name='faiss_users')

# compute fixed axis ranges for valence & energy (used for coordinate conversion)
x_min = tracks_data["valence"].min()
x_max = tracks_data["valence"].max()
y_min = tracks_data["energy"].min()
y_max = tracks_data["energy"].max()

# reactive values
valence_selected = reactive.Value(0.5)
energy_selected = reactive.Value(0.5)
recc_tracks_plot = reactive.Value(pd.DataFrame())
recc_tracks_buddy = reactive.Value(pd.DataFrame())
recc_tracks_track = reactive.Value(pd.DataFrame())
# user_id = reactive.Value(5)


# @reactive.Effect
# def print_user_id():
#     print(f'current user: {user_id.get()}')


# ui
ui = ui.page_fixed(

    # app customization
    ui.tags.head(
        ui.tags.style(
            """
            /* body */
            body {
                background-color: #0B223E;
                color: #ffffff;
                width: 100vw;
                height: 100vh;
                font-family: 'NPO Sans';
            }
            
            @font-face {
                font-family: 'NPO Sans';
                src: url('NPO_Sans_Regular.otf') format('opentype');
                font-weight: normal;
                font-style: normal;
            }
        
            .container-fluid {
                padding: 0px;
            }
            
            /* title */
            .row.title-bar {
                padding: 0px;
            }
            
            .col-sm-6.column-title {
                padding: 15px;
                width: auto;
                height: auto;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            
            #npo_logo.shiny-image-output.shiny-bound-output {
                width: auto !important;
                height: auto !important;
            }
            
            h1.mb-5 {
                width: auto;
                height: auto;
                margin-bottom: 0px !important;
            }
            
            .col-sm-8.description {
                padding: 0px;
            }
            
            h2.description-text {
                padding: 0px;
                margin-top: 10px;
                margin-bottom: 15px;
                font-size: 26px;
            }

            /* input select */
            .col-sm-3.select-menu {
                padding: 0px;
                padding-right: 10px;
                display: flex;
                flex-direction: column;
            }
            
            .row.row-explanation {
                display: flex;
                flex-direction: column;
                padding: 10px;
                background-color: #1F3352;
                border-radius: 5px;
            }
            
            p.text-explanation {
                padding: 5px;
                padding-left: 10px;
                padding-right: 10px;
                margin: 0px;
            }
            
            .row.row-sliders {
                margin-top: 10px;
                display: flex;
                flex-direction: column;
                align-content: stretch;
                align-items: stretch;
                padding: 0px;
            }
            
            .form-group.shiny-input-container {
                margin-bottom: 5px; 
                width: auto;
            }
      
            .form-control, .selectize-input, .shiny-input-container input[type="number"] {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ffffff;
                padding: 8px;
                border-radius: 5px;
                font-size: 14px;
            }
            
            .row.row-slider-explanation {
                font-size: 14px;
                display: flex;
            }
            
            p.slider-explanation {
                padding-left: 10px;
                padding-right: 10px;
                padding-bottom: 5px;
                padding-top: 0px;
                margin: 0px;
            }
            
            /* output boxes */
            .col-sm-4.recommendations-column {
                padding: 0px;
                padding-left: 10px;
            }
            
            h2.your-recommendations {
                font-size: 26px;
            }
            
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
            
            .row.track-recommendation.mb-5 {
                padding-bottom: 25px !important;
            }
            
            /* modal */
            .modal-content,
            .modal-header,
            .modal-body,
            .modal-footer {
                background-color: #1e3352 !important;
                color: #ffffff !important;
            }
            
            .modal-btn-default {
                background-color: #333333;
                color: white;
            }
            """
        )),

    # row 1 (title)
    ui.row(
        ui.column(6, ui.output_image("npo_logo", width="auto", height="auto"),
                  ui.h1("Explore new artists", class_='mb-5'),
                  class_= "column-title"),
        class_="title-bar"),

    # row 2 (description)
    ui.row(
        ui.column(8,
                  ui.h2("Select your favorite tracks based on energy and valence ", class_='description-text'),
                  class_ = "description"),
        ui.column(4,
                  ui.h2("Your recommendations:", class_="description-text"),
                  class_ = "description")),

    # row 3 (filter + plot + recommendations)
    ui.row(
        ui.column(3,

                  # explanation plot
                  ui.row(
                      ui.p('You can select certain genre clusters, specific genres, or no filter at all.\n', class_='text-explanation'),
                      ui.p("• Valence: the musical positiveness conveyed by a track", class_='text-explanation'),
                      ui.p("• Energy: a perceptual measure of intensity and activity", class_='text-explanation'),
                      class_= "row-explanation",
                  ),

                  ui.row(
                      # select genre group
                      ui.input_selectize("genre_cluster_filter", "Select Genre cluster:",
                                     choices=["All"] + sorted(tracks_data["genre_cluster"].unique().tolist()),
                                     multiple=True),

                      # select genre
                      ui.input_selectize("genre_filter", "Select Genre:",
                                     choices=["All"] + sorted(tracks_data["track_genre"].unique().tolist()),
                                     multiple=True),

                      # slider diversity
                      #ui.p('Select level of diversity:', class_='text-explanation'),
                      ui.input_slider("slider_diversity", "Select level of diversity:",
                                  min=1, max=3, step=1, value=2),
                      ui.row(
                          ui.p('• Level 1: low diversity (based on listening history)', class_='slider-explanation'),
                          ui.p('• Level 2: moderate diversity (combination)', class_='slider-explanation'),
                          ui.p('• Level 3: high diversity (diverse)',class_='slider-explanation'),
                          class_= "row-slider-explanation",
                      ),


                      class_= "row-sliders"),

                  # column class
                  class_="select-menu"),

            ui.column(5,

                      # plot widget
                      output_widget("plot"),

                      # clickable plot
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

                      # output list tracks
                      ui.output_ui("recommended_tracks_plot"),

                      # column class
                      class_="recommendations-column"
            ),

        # row class
        class_ = "filter-plot-recommendations mb-5"),

    # row 4 (buddy recommendations)
    ui.row(
        ui.column(4,
                  ui.h2("...or select a track from your buddy's playlist", class_='description-text'),

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

                  # checkbox buddy consent
                  ui.input_checkbox("buddy_consent", 'buddy consent', value=False),

                  # buddy modal
                  ui.output_ui("buddy_modal"),

                  # column class
                  class_="buddy-column"),

        ui.column(8,

                  # show buddy recommendations
                  ui.h2("Your buddy's recommendations:", class_='description-text'),
                  ui.output_ui("recommended_tracks_list_buddy"),

                  # column class
                  class_="buddy_recommendations-column"),

        # row class
        class_="buddy-recommendations mb-5"),

    # row 5 (track selection)
    ui.row(
        ui.column(4,

                  # select specific track
                  ui.h2('...or select a track from the catalog', class_='description-text'),
                  ui.input_selectize("track_selection", "Search for a Track:",
                                     choices=["Select a track"] + sorted(
                                         tracks_data.fillna("").apply(lambda row: f"{row['track_name']} - {row['artists']} ({row['album_name']})",
                                                                      axis=1).unique().tolist()
                                     ),
                                     multiple=False,
                                     options={"create": True} ),

                  # column class
                  class_="recommendations-column-2"),

        ui.column(8,

                  # show recommendations specific track
                  ui.h2("Your recommendations based on selected track:", class_='description-text'),
                  ui.output_ui("recommended_tracks_track"),

                  # column class
                  class_="track_recommendations-column"),

        # row class
        class_="track-recommendation mb-5"),

)

# server
def server(input, output, session):

    # render NPO logo
    @render.image
    def npo_logo():
        dir = Path(__file__).resolve().parent
        logo: ImgData = {"src": str(dir / "static/NPO_logo.png"), "width": "auto", "height": "auto"}
        return logo

    # filter data (based on dropdown menu selection)
    @reactive.Calc
    def filtered_data():

        # create copy of data + save selected genres
        data = tracks_data.copy()

        # filter based on selected genre (cluster)
        selected_clusters = input.genre_cluster_filter()
        if isinstance(selected_clusters, str):
            selected_clusters = [selected_clusters]
        if selected_clusters and "All" not in selected_clusters:
            data = data[data["genre_cluster"].isin(selected_clusters)]

        selected_genres = input.genre_filter()
        if isinstance(selected_genres, str):
            selected_genres = [selected_genres]
        if selected_genres and "All" not in selected_genres:
            data = data[data["track_genre"].isin(selected_genres)]

        return data

    # print selected categories to console
    @reactive.Effect
    def print_selected_categories():
        print(f"Selected genres: {input.genre_filter()}")
        print(f"Selected genre clusters: {input.genre_cluster_filter()}")

    # render plot (for clicking ability)
    @render_widget
    def plot():
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

        # convert plot to widget
        w = go.FigureWidget(scatterplot.data, scatterplot.layout)

        # capture clicks
        w.data[0].on_click(on_point_click)

        return w

    # react to click (selects nearest point)
    @reactive.Effect
    def update_nearest_point():
        data = input.plot_click_any()

        if data is not None:
            df = filtered_data()

            # compute distance from click to points
            distances = ((df["valence"] - data["x"])**2 + (df["energy"] - data["y"])**2)**0.5

            # find nearest point
            nearest_idx = distances.idxmin()
            nearest_valence = df.loc[nearest_idx, "valence"]
            nearest_energy = df.loc[nearest_idx, "energy"]
            valence_selected.set(nearest_valence)
            energy_selected.set(nearest_energy)
            print(f"Nearest point selected: Valence: {nearest_valence}, Energy: {nearest_energy}")

            # reset track selection
            session.send_input_message("track_selection", {"value": "Select a track"})

    # track selection
    @reactive.Calc
    def selected_track():

        # save and print selected track from input
        track_info = input.track_selection()
        print(f'Selected track: {track_info}')

        if not track_info or track_info == "Select a track":
            return None

        # extract track & artist
        track_name, artist_album = track_info.split(" - ", 1)
        artist, album = artist_album.rsplit(" (", 1)
        album = album.rstrip(")")

        # find track in dataframe
        track_row = tracks_data[
            (tracks_data["track_name"] == track_name) &
            (tracks_data["artists"] == artist) &
            (tracks_data["album_name"] == album)
            ]

        if track_row.empty:
            print(f"WARNING: Selected track '{track_info}' not found in dataset!", flush=True)
            return None

        return track_row.iloc[0]

    @reactive.Effect
    @reactive.event(input.track_selection)
    def execute_function_on_track_selection():

        # save selected track from input function
        track = selected_track()

        if track is None:
            print("ERROR: No valid track selected.", flush=True)
            return

        # save & print track id
        track_id = track["track_id"]
        print(f"catalogue track selected: {track_id}", flush=True)

        # get recommendations using FAISS
        track_sel_rec = recommend_similar_tracks_audio_ft(
            track_id, tracks_data, tracks_faiss[0], tracks_faiss[1], audio_features, num_recommendations=5
        )

        recc_tracks_track.set(track_sel_rec)

    # get diversity slider value
    @reactive.calc
    def get_diversity_level():
        return {1: 5, 2: 20, 3: 200}[input.slider_diversity()]

    # reactive function (get KNN recommendations based on valence and energy values)
    @reactive.Effect
    def recommended_tracks():

        # current_user_id = user_id.get()
        current_user_id = 1

        # get slider input
        top_n = get_diversity_level()

        # get valence and energy values from the nearest point
        valence = valence_selected.get()
        energy = energy_selected.get()

        # call KNN function with updated values
        nn = knn_module(filtered_data(), valence, energy)
        
        # apply filter function to remove Christmas songs
        nc = filter_christmas_songs(nn)

        # get similar tracks based on KNN results & diversity level
        sim = get_most_similar_tracks(nc, user_data, current_user_id, top_n=top_n)
        sim = sim.head(5)

        # check max length if top_n == 200 (if lower -> change max)
        if top_n == 200:
            max_tracks = len(sim)
            top_n = min(max_tracks, 200)

        # check if sim is empty before applying inverse_popularity
        if sim.empty:
            print("WARNING: No similar tracks found. No recommendations available.", flush=True)
            return

        # apply inverse popularity filter
        inv_pop = inverse_popularity(sim, top_n)

        # check if inv_pop is empty before updating recc_tracks
        if inv_pop.empty:
            print("WARNING: No recommendations after applying inverse popularity filter.", flush=True)
            return

        # update reactive value
        recc_tracks_plot.set(inv_pop)

        # print recommended tracks
        print(f"Updated recommendations using 'recommended_tracks' with {len(inv_pop)} tracks", flush=True)

    # plot function
    @render_widget
    def plot():
        scatterplot = px.scatter(

            # call filtered data
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

        # hover menu
        ]).update_traces(
            marker=dict(size=8, color="#DEDEDE", opacity=0.6),
            hovertemplate="<b>Valence:</b> %{customdata[0]}<br>"
                          "<b>Energy:</b> %{customdata[1]}<br>"
                          "<b>Song:</b> %{customdata[2]}<br>"
                          "<b>Artist:</b> %{customdata[3]}<br>"
                          "<b>Album:</b> %{customdata[4]}<br>"
                          "<b>Genre:</b> %{customdata[5]}<br>"

        # plot layout
        ).update_layout(
            title={
                "text": "Select your favorite tracks based on energy and valence to get recommendations!",
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

    @render.image
    def clickable_img():
        # return dictionary with at least src and one of width/height
        dir = Path(__file__).resolve().parent
        return {
            "src": str(dir / "static/buddy_3.png"),
            "width": "200px",
            "height": "auto",
            "alt": "buddy image",
            "id": "clickable_img",
            "style": "cursor: pointer;"
        }

    # execute the function when image is clicked
    @reactive.Effect
    @reactive.event(input.img_clicked)
    def execute_function_on_image_click():
        print(f'image clicked', flush=True)
        current_user_id = 1
        buddy_rec = buddy_recommendations(current_user_id, user_data, num_recommendations=5)
        recc_tracks_buddy.set(buddy_rec)
        session.send_input_message("track_selection", {"value": "Select a track"})

    # buddy consent
    @reactive.effect
    @reactive.event(input.buddy_consent)
    def buddy_modal():
        if input.buddy_consent():
            m = modal(
                "By providing buddy consent you are aware that this potentially decreases your privacy, as you might be identified via your buddy's internal/external data outside your control",
                title="Buddy consent",
                easy_close=True,
                fade=False,
                footer=modal_button("Close")
                )
            modal_show(m)

    # recommendations buddy
    @render.ui
    def recommended_tracks_list_buddy():
        if not input.buddy_consent():
            return tags.div(
                "❌ No recommendations available. Please provide buddy consent.",
                style="color: #ffffff; font-weight: bold; padding: 10px;"
            )

        tracks = recc_tracks_buddy.get()

        if tracks.empty:
            return tags.div(
                "⚠️ No recommendations found for your buddy.",
                style="color: #ffffff; font-weight: bold; padding: 10px;"
            )

        return generate_recommended_tracks_list(tracks)

    # generate recommended track lists
    @render.ui
    def recommended_tracks_plot():
        tracks = recc_tracks_plot.get()
        return generate_recommended_tracks_list(tracks)

    @render.ui
    def recommended_tracks_track():
        tracks = recc_tracks_track.get()
        return generate_recommended_tracks_list(tracks)

# run app
app = App(ui, server, static_assets=Path(__file__).parent/"static")

if __name__ == "__main__":
    run_app(app)

# TODO logic genre&cluster filter
# TODO tooltip explanation app

