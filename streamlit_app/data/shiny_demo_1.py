
import pandas as pd
from shiny import App, ui
from shiny import run_app
from shinywidgets import output_widget, render_widget
import plotly.express as px
from shiny import reactive, render_text

# data
#df = pd.read_csv("spotify_tracks_clean.csv")
data = {
    "name": ["cowboy like me", "Caught in the Middle", "imgonnagetyouback", "Told You So", "I Did Something Bad"],
    "artist": ["Taylor Swift", "Paramore", "Taylor Swift", "Paramore", "Taylor Swift"],
    "album": ["evermore", "After Laughter", "The Tortured Poets Department", "After Laughter", "reputation"],
    "danceability": [0.6, 0.8, 0.75, 0.65, 0.7],
    "energy": [0.5, 0.8, 0.4, 0.85, 0.65]
}
df = pd.DataFrame(data)

# UI
ui = ui.page_fluid(
    output_widget("plot"),
)

# SERVER
def server(input, output, session):

    # plot
    @render_widget
    def plot():
        scatterplot = px.scatter(
            data_frame = df,
            x = "danceability",
            y = "energy",
            hover_data = ["name", "artist", "album"]
        ).update_traces(
            hovertemplate = "<b>Song:</b> %{customdata[0]}<br><b>Artist:</b> %{customdata[1]}<br><b>Album:</b> %{customdata[2]}<extra></extra>"
        ).update_layout(
            title = {"text": "Danceability vs. Energy"},
            yaxis_title = "Energy",
            xaxis_title = "Danceability",
        )

        return scatterplot

# run app
app = App(ui, server)

if __name__ == "__main__":
    run_app(app)