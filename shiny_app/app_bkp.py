from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load dataset and randomly sample 100 rows
tracks_data = pd.read_csv("data/spotify_tracks_clean_clusters.csv")
tracks_data = tracks_data.sample(1000)

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

# UI definition with a custom JS hook
ui = ui.page_fluid(
    ui.h2("Spotify Track Analysis"),
    output_widget("plot"),
    ui.h2("Selected Track Features"),
    ui.output_text("selected_valence"),
    ui.output_text("selected_energy"),
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
                    var total_width = 600;
                    var total_height = 400;
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
    """)
)

# Server logic
def server(input, output, session):

    @reactive.Calc
    def filtered_data():
        return tracks_data

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
            width=600,
            height=400,
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

    @render.text
    def selected_valence():
        return f"Selected Valence: {valence_selected()}"

    @render.text
    def selected_energy():
        return f"Selected Energy: {energy_selected()}"

    output.plot = plot
    output.selected_valence = selected_valence
    output.selected_energy = selected_energy

app = App(ui, server)

if __name__ == "__main__":
    app.run()
