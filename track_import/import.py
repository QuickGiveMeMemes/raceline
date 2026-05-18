import argparse
import yaml
from track_import.track_fitting import fit_track
from track_import.gpx_parsing import read_gpx_splines
import plotly.graph_objects as go

"""
Interface for running track fitting
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g", 
        "--gpx", 
        required=True, 
        type=str, 
        help="Source path to track gpx file.",
    )
    parser.add_argument(
        '-s',
        "--savefile",
        type=str,
        help="Destination path of fitted track.",
    )
    parser.add_argument(
        '-c',
        "--config", 
        default="config/default.yaml", 
        type=str, 
        help="Path to config file.",
    )
    parser.add_argument(
        "-p",
        "--plot", 
        default=False, 
        action="store_true", 
        help="Toggles on plotting.",
    )
    parser.add_argument(
        "-r",
        "--refine", 
        default=False, 
        action="store_true", 
        help="Toggles mesh refinement.",
    )
    parser.add_argument(
        "--solver", 
        default="mumps", 
        type=str, 
        help="Solver to use (mumps, ma57, ma86, ma97, etc.).",
    )

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config_data = yaml.safe_load(file)

    config_data["track_fitting"]["ipopt"]["ipopt.linear_solver"] = args.solver

    (
        original_track,
        (
            max_dist,
            spline_l,
            spline_r,
            spline_c,
            _,
            _,
            _,
        ),
        ccw,
    ) = read_gpx_splines(args.gpx)

    track = fit_track(spline_c, spline_l, spline_r, max_dist, config_data["track_fitting"], ccw, args.refine)

    if args.savefile:
        track.save(args.track_destination)

    if args.plot:
        plots = []

        plots.append(
            go.Scatter3d(
                x=original_track[2][0],
                y=original_track[2][1],
                z=original_track[2][2],
                name="original center",
                mode="lines",
            )
        )

        plots.append(
            go.Scatter3d(
                x=original_track[0][0],
                y=original_track[0][1],
                z=original_track[0][2],
                name="original left",
                mode="lines",
            )
        )
        plots.append(
            go.Scatter3d(
                x=original_track[1][0],
                y=original_track[1][1],
                z=original_track[1][2],
                name="original right",
                mode="lines",
            )
        )

        fig = go.Figure()

        fine_plot, q_fine = track.plot_uniform(1)
        ribbon = track.plot_ribbon()
        collocation_plot, q_collocation = track.plot_collocation()

        for i in (*fine_plot, ribbon, *collocation_plot, *plots):
            fig.add_trace(i)

        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
            ),
            legend=dict(
                orientation="h",
            ),
        )

        fig.show()

        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
            ),
            legend=dict(
                orientation="h",
            ),
        )
        fig.show()

        # Plot theta, mu, phi
        # TODO maybe add a for fun config for this
        # q_fig = go.Figure(data=[q_collocation, q_fine])
        # q_fig.show()
