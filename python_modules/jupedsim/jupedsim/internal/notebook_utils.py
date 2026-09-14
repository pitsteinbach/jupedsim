## SPDX-License-Identifier: LGPL-3.0-or-later

"""This code is used in examples on jupedsim.org.

We make no promises about the functions from this file w.r.t. API stability. We
reservere us the right to change the code here w.o. warning. Do not use the
code here. Use it at your own peril.
"""

import pathlib
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pedpy
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# Fix for plotly 6.x wrt animations in jupyter notebooks
pio.renderers.default = "sphinx_gallery"

DUMMY_SPEED = -1000


def read_sqlite_file(
    trajectory_file: str,
) -> tuple[pedpy.TrajectoryData, pedpy.WalkableArea]:
    """ """
    with sqlite3.connect(trajectory_file) as con:
        data = pd.read_sql_query(
            "select frame, id, pos_x as x, pos_y as y from trajectory_data",
            con,
        )
        fps = float(
            con.cursor()
            .execute("select value from metadata where key = 'fps'")
            .fetchone()[0]
        )
        walkable_area = (
            con.cursor().execute("select wkt from geometry").fetchone()[0]
        )
        return (
            pedpy.TrajectoryData(data=data, frame_rate=fps),
            pedpy.WalkableArea(walkable_area),
        )


def _speed_to_color(speed, min_speed, max_speed):
    """Map a speed value to a color using a colormap."""
    normalized_speed = (speed - min_speed) / (max_speed - min_speed)
    r, g, b = plt.cm.jet_r(normalized_speed)[:3]
    return f"rgba({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f}, 0.5)"


# Categorical palette, used when agents are colored by a category (e.g. the
# operational model they use) instead of by speed. Slots are assigned in this
# fixed order and never cycled.
CATEGORY_PALETTE = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
UNKNOWN_CATEGORY_COLOR = "#969696"


def _hex_to_rgba(hex_color, alpha=0.9):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _build_category_colors(categories):
    """Map each category to a palette slot, in the order given."""
    if len(categories) > len(CATEGORY_PALETTE):
        raise ValueError(
            f"can color at most {len(CATEGORY_PALETTE)} categories, "
            f"got {len(categories)}"
        )
    return {
        category: color for category, color in zip(categories, CATEGORY_PALETTE)
    }


def _get_legend_traces(category_colors):
    """Off-canvas markers that carry the category legend."""
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=_hex_to_rgba(color)),
            name=str(category),
            showlegend=True,
            hoverinfo="none",
        )
        for category, color in category_colors.items()
    ]


def _get_line_color(disk_color):
    r, g, b, _ = [int(float(val)) for val in disk_color[5:-2].split(",")]
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "black" if brightness > 127 else "white"


def _contrast_color(color):
    """Black or white, whichever is readable on top of ``color``."""
    r, g, b = to_rgb(color)
    return "black" if (r * 299 + g * 587 + b * 114) / 1000 > 0.5 else "white"


def _create_orientation_line(row, line_length=0.2, color="black"):
    # Derive the orientation from the pedpy-computed velocity (v_x, v_y).
    vx = row["v_x"]
    vy = row["v_y"]
    norm = np.hypot(vx, vy)
    if norm > 0:
        vx, vy = vx / norm, vy / norm
    else:
        vx = vy = 0.0
    end_x = row["x"] + line_length * vx
    end_y = row["y"] + line_length * vy

    orientation_line = go.layout.Shape(
        type="line",
        x0=row["x"],
        y0=row["y"],
        x1=end_x,
        y1=end_y,
        line=dict(color=color, width=3),
    )
    return orientation_line


def _get_geometry_traces(area):
    geometry_traces = []
    x, y = area.exterior.xy
    geometry_traces.append(
        go.Scatter(
            x=np.array(x),
            y=np.array(y),
            mode="lines",
            line={"color": "grey"},
            showlegend=False,
            name="Exterior",
            hoverinfo="name",
        )
    )
    for inner in area.interiors:
        xi, yi = zip(*inner.coords[:])
        geometry_traces.append(
            go.Scatter(
                x=np.array(xi),
                y=np.array(yi),
                mode="lines",
                line={"color": "grey"},
                showlegend=False,
                name="Obstacle",
                hoverinfo="name",
            )
        )
    return geometry_traces


def _get_colormap(frame_data, max_speed):
    """Utilize scatter plots with varying colors for each agent instead of individual shapes.

    This trace is only to incorporate a colorbar in the plot.
    """
    scatter_trace = go.Scatter(
        x=frame_data["x"],
        y=frame_data["y"],
        mode="markers",
        marker=dict(
            size=frame_data["radius"] * 2,
            color=frame_data["speed"],
            colorscale="Jet_r",
            colorbar=dict(title="Speed [m/s]"),
            cmin=0,
            cmax=max_speed,
        ),
        text=frame_data["speed"],
        showlegend=False,
        hoverinfo="none",
    )

    return [scatter_trace]


def _get_shapes_for_frame(
    frame_data, min_speed, max_speed, category_colors=None
):
    def create_shape(row):
        hover_trace = go.Scatter(
            x=[row["x"]],
            y=[row["y"]],
            text=[f"ID: {row['id']}, Pos({row['x']:.2f},{row['y']:.2f})"],
            mode="markers",
            marker=dict(size=1, opacity=1),
            hoverinfo="text",
            showlegend=False,
        )
        if row["speed"] == DUMMY_SPEED:
            dummy_trace = go.Scatter(
                x=[row["x"]],
                y=[row["y"]],
                mode="markers",
                marker=dict(size=1, opacity=0),
                hoverinfo="none",
                showlegend=False,
            )
            return (
                go.layout.Shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=row["x"] - row["radius"],
                    y0=row["y"] - row["radius"],
                    x1=row["x"] + row["radius"],
                    y1=row["y"] + row["radius"],
                    line=dict(width=0),
                    fillcolor="rgba(255,255,255,0)",  # Transparent fill
                ),
                dummy_trace,
                _create_orientation_line(row, color="rgba(255,255,255,0)"),
            )
        if category_colors is None:
            color = _speed_to_color(row["speed"], min_speed, max_speed)
        else:
            color = _hex_to_rgba(
                category_colors.get(row["category"], UNKNOWN_CATEGORY_COLOR)
            )
        return (
            go.layout.Shape(
                type="circle",
                xref="x",
                yref="y",
                x0=row["x"] - row["radius"],
                y0=row["y"] - row["radius"],
                x1=row["x"] + row["radius"],
                y1=row["y"] + row["radius"],
                line_color=color,
                fillcolor=color,
            ),
            hover_trace,
            _create_orientation_line(row, color=_get_line_color(color)),
        )

    results = frame_data.apply(create_shape, axis=1).tolist()
    shapes = [res[0] for res in results]
    hover_traces = [res[1] for res in results]
    arrows = [res[2] for res in results]
    return shapes, hover_traces, arrows


def _create_fig(
    initial_agent_count,
    initial_shapes,
    initial_arrows,
    initial_hover_trace,
    initial_scatter_trace,
    geometry_traces,
    frames,
    steps,
    area_bounds,
    width=800,
    height=800,
    title_note: str = "",
    legend_title: str = "",
):
    """Creates a Plotly figure with animation capabilities.

    Returns:
        go.Figure: A Plotly figure with animation capabilities.
    """

    minx, miny, maxx, maxy = area_bounds
    title = f"<b>{title_note + '  |  ' if title_note else ''}Number of Agents: {initial_agent_count}</b>"
    fig = go.Figure(
        data=geometry_traces
        + initial_scatter_trace
        # + hover_traces
        + initial_hover_trace,
        frames=frames,
        layout=go.Layout(
            shapes=initial_shapes + initial_arrows, title=title, title_x=0.5
        ),
    )
    fig.update_layout(
        legend=dict(title=legend_title, itemsizing="constant"),
        updatemenus=[_get_animation_controls()],
        sliders=[_get_slider_controls(steps)],
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(range=[minx - 0.5, maxx + 0.5]),
        yaxis=dict(
            scaleanchor="x", scaleratio=1, range=[miny - 0.5, maxy + 0.5]
        ),
    )

    return fig


def _get_animation_controls():
    """Returns the animation control buttons for the figure."""
    return {
        "buttons": [
            {
                "args": [
                    None,
                    {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                    },
                ],
                "label": "Play",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }


def _get_slider_controls(steps):
    """Returns the slider controls for the figure."""
    return {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Frame:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 100, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": steps,
    }


def _get_processed_frame_data(data_df, frame_num, max_agents):
    """Process frame data and ensure it matches the maximum agent count."""
    frame_data = data_df[data_df["frame"] == frame_num]
    agent_count = len(frame_data)
    dummy_agent_data = {"x": 0, "y": 0, "radius": 0, "speed": DUMMY_SPEED}
    while len(frame_data) < max_agents:
        dummy_df = pd.DataFrame([dummy_agent_data])
        frame_data = pd.concat([frame_data, dummy_df], ignore_index=True)
    return frame_data, agent_count


def _prepare_data(data, radius, agent_categories):
    """Speed, velocity, radius and (optionally) category per agent and frame."""
    data_df = pedpy.compute_individual_speed(
        traj_data=data,
        frame_step=5,
        compute_velocity=True,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    data_df = data_df.merge(data.data, on=["id", "frame"], how="left")
    data_df["radius"] = radius
    category_colors = None
    if agent_categories is not None:
        data_df["category"] = data_df["id"].map(agent_categories)
        category_colors = _build_category_colors(
            list(dict.fromkeys(agent_categories.values()))
        )
    return data_df, category_colors


def animate(
    data: pedpy.TrajectoryData,
    area: pedpy.WalkableArea,
    *,
    every_nth_frame: int = 50,
    width: int = 800,
    height: int = 800,
    radius: float = 0.2,
    title_note: str = "",
    agent_categories: dict[int, str] | None = None,
    legend_title: str = "",
):
    """Animate the trajectories.

    Arguments:
        agent_categories: Optional mapping of agent id to a category name,
            e.g. the operational model the agent was simulated with. If given,
            agents are colored by category (legend) instead of by speed
            (colorbar). Agent ids not in the mapping are drawn grey.
        legend_title: Title of the category legend.
    """
    data_df, category_colors = _prepare_data(data, radius, agent_categories)
    min_speed = data_df["speed"].min()
    max_speed = data_df["speed"].max()
    max_agents = data_df.groupby("frame").size().max()
    frames = []
    steps = []
    unique_frames = data_df["frame"].unique()
    selected_frames = unique_frames[::every_nth_frame]
    geometry_traces = _get_geometry_traces(area.polygon)
    initial_frame_data = data_df[data_df["frame"] == data_df["frame"].min()]
    initial_agent_count = len(initial_frame_data)
    (
        initial_shapes,
        initial_hover_trace,
        initial_arrows,
    ) = _get_shapes_for_frame(
        initial_frame_data, min_speed, max_speed, category_colors
    )
    if category_colors is None:
        legend_traces = _get_colormap(initial_frame_data, max_speed)
    else:
        legend_traces = _get_legend_traces(category_colors)
    for frame_num in selected_frames:
        frame_data, agent_count = _get_processed_frame_data(
            data_df, frame_num, max_agents
        )
        shapes, hover_traces, arrows = _get_shapes_for_frame(
            frame_data, min_speed, max_speed, category_colors
        )
        title = f"<b>{title_note + '  |  ' if title_note else ''}Number of Agents: {agent_count}</b>"
        frame_name = str(int(frame_num))
        frame = go.Frame(
            data=geometry_traces + legend_traces + hover_traces,
            name=frame_name,
            layout=go.Layout(
                shapes=shapes + arrows,
                title=title,
                title_x=0.5,
            ),
        )
        frames.append(frame)

        step = {
            "args": [
                [frame_name],
                {
                    "frame": {"duration": 100, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 500},
                },
            ],
            "label": frame_name,
            "method": "animate",
        }
        steps.append(step)

    return _create_fig(
        initial_agent_count,
        initial_shapes,
        initial_arrows,
        initial_hover_trace,
        legend_traces,
        geometry_traces,
        frames,
        steps,
        area.bounds,
        width=width,
        height=height,
        title_note=title_note,
        legend_title=legend_title,
    )


def animate_to_gif(
    data: pedpy.TrajectoryData,
    area: pedpy.WalkableArea,
    output_file: str | pathlib.Path,
    *,
    every_nth_frame: int = 5,
    fps: int = 10,
    radius: float = 0.2,
    width: float = 8.0,
    height: float = 8.0,
    dpi: int = 100,
    title_note: str = "",
    agent_categories: dict[int, str] | None = None,
    legend_title: str = "",
) -> pathlib.Path:
    """Write the animation to an animated GIF, e.g. for use in slides.

    Renders with matplotlib -- same colors as :func:`animate`, but a plain
    image sequence instead of an interactive plotly figure.

    Arguments:
        output_file: Path of the GIF to write.
        every_nth_frame: Only every n-th simulation frame becomes a GIF frame.
        fps: Frames per second of the GIF.
        agent_categories: Optional mapping of agent id to a category name. If
            given, agents are colored by category (legend) instead of by speed
            (colorbar).
        legend_title: Title of the category legend.

    Returns:
        The path of the written GIF.
    """
    data_df, category_colors = _prepare_data(data, radius, agent_categories)
    min_speed = data_df["speed"].min()
    max_speed = data_df["speed"].max()
    max_agents = data_df.groupby("frame").size().max()
    selected_frames = data_df["frame"].unique()[::every_nth_frame]

    fig, axes = plt.subplots(figsize=(width, height), dpi=dpi)
    pedpy.plot_walkable_area(walkable_area=area, axes=axes)
    minx, miny, maxx, maxy = area.bounds
    axes.set_xlim(minx - 0.5, maxx + 0.5)
    axes.set_ylim(miny - 0.5, maxy + 0.5)
    axes.set_xlabel("x/m")
    axes.set_ylabel("y/m")
    axes.set_aspect("equal")

    if category_colors is None:
        norm = plt.Normalize(vmin=min_speed, vmax=max_speed)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet_r),
            ax=axes,
            label="Speed [m/s]",
            fraction=0.046,
            pad=0.04,
        )
    else:
        axes.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markersize=10,
                    color=color,
                    label=str(category),
                )
                for category, color in category_colors.items()
            ],
            title=legend_title or None,
            loc="upper left",
        )

    # One reusable disk and orientation line per agent, hidden while unused.
    disks = []
    orientations = []
    for _ in range(max_agents):
        disk = Circle((0, 0), radius, alpha=0.9, visible=False)
        axes.add_patch(disk)
        disks.append(disk)
        orientations.append(axes.plot([], [], linewidth=1.5, visible=False)[0])

    def draw_frame(frame_num):
        frame_data = data_df[data_df["frame"] == frame_num]
        for disk, orientation, (_, row) in zip(
            disks, orientations, frame_data.iterrows()
        ):
            if category_colors is None:
                color = plt.cm.jet_r(
                    (row["speed"] - min_speed) / (max_speed - min_speed)
                )
            else:
                color = category_colors.get(
                    row["category"], UNKNOWN_CATEGORY_COLOR
                )
            disk.set(center=(row["x"], row["y"]), color=color, visible=True)
            norm_v = np.hypot(row["v_x"], row["v_y"])
            if norm_v > 0:
                orientation.set_data(
                    [row["x"], row["x"] + radius * row["v_x"] / norm_v],
                    [row["y"], row["y"] + radius * row["v_y"] / norm_v],
                )
            else:
                orientation.set_data([row["x"]], [row["y"]])
            orientation.set(color=_contrast_color(color), visible=True)
        for disk, orientation in zip(
            disks[len(frame_data) :], orientations[len(frame_data) :]
        ):
            disk.set_visible(False)
            orientation.set_visible(False)
        axes.set_title(
            f"{title_note + '  |  ' if title_note else ''}"
            f"Number of Agents: {len(frame_data)}"
        )
        return [*disks, *orientations]

    animation = FuncAnimation(
        fig, draw_frame, frames=selected_frames, blit=False
    )
    output_file = pathlib.Path(output_file)
    animation.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_file
