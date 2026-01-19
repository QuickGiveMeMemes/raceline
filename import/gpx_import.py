from dataclasses import dataclass
import casadi as ca
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gpxpy
import gpxpy.gpx
from pyproj import Transformer
from scipy.interpolate import splev, splprep
from scipy.spatial import KDTree
from casadi import *
import plotly.express as px
import plotly.graph_objects as go


matplotlib.use("Qt5Agg")

FILE = "Track.gpx"
RESOLUTION = 15.0  # meters

def parameterized_interpolation(track: list[np.ndarray]) -> np.ndarray:

    l_init = track[0]
    r_init = track[1]
    c_init = track[2]
    
    # The arrays encode the path distances and spline point values for the center and both 
    # bounds at an evenly spaced interval around the track. 
    # Indexing is [l, r, c]
    spl_dist, spl_pts, c_spline = interpolate(c_init, RESOLUTION, True)

    # Generates high-resolution interpolations for the left and right boundaries, this will
    # result in high accuracy for nearest-neighbor search and pairing.
    _, l_fine_pts, _ = interpolate(l_init, spacing=0.5)
    _, r_fine_pts, _ = interpolate(r_init, spacing=0.5)
    l_fine_pts = np.asarray(l_fine_pts)
    r_fine_pts = np.asarray(r_fine_pts)
    
    # Uses nearest-neighbor search to pair/unify the center points with boundary points for
    # ribbon construction. The nearest neighbor on either boundary of a center point is assumed
    # to be approximately orthogonal to the centerline.
    l_nn = KDTree(np.transpose(l_fine_pts[:2]))
    r_nn = KDTree(np.transpose(r_fine_pts[:2]))
    _, l_nearest = l_nn.query(np.transpose(spl_pts[:2]))
    _, r_nearest = r_nn.query(np.transpose(spl_pts[:2]))


    l_spline, _ = splprep(l_fine_pts[:, l_nearest], u=spl_dist, s=0)
    r_spline, _ = splprep(r_fine_pts[:, r_nearest], u=spl_dist, s=0)

    # TODO remove most of these as they are debug/plot
    return spl_dist[-1], l_spline, r_spline, c_spline, l_fine_pts[:, l_nearest], r_fine_pts[:, r_nearest], spl_pts


def interpolate(sample: np.ndarray, spacing: float = 1.0, accurate_param=False) -> np.ndarray:
    """
    Fits polynomial splines to the given sample and returns a new array with
    evenly spaced points sampled from the splines. We approximate the total distance
    of the calculated splines and map the desired distances of each point to the
    default parameterization produce by scipy.

    Args:
        sample (np.ndarray):    A 2-dimensional array in the format [[point],[point], ... ].
                                For convenience, the method parameterizes based on arc length
                                of the first three values in each point assumed (x,y,z)
        spacing (float, optional):  The distance spacing between each of the points in the array
                                    that is returned. Defaults to 0.1.
        accurate_param (bool):  Denotes whether or not the method will regenerate the spline based
                                on accurate arc length parameterization

    Returns:
        np.ndarray: A set of evenly-spaced points along the interpolated spline
    """

    # Creates an initial spline approximation for a fine sampling of points along the spline.
    # This way, we have an accurate arc length approximation to evenly space the points
    spline, u = splprep(sample, s=0, k=3, per=True)
    u_fine = np.linspace(u.min(), u.max(), 250_000)
    fine_sample = splev(u_fine, spline)
    dist = np.cumsum(
        np.sqrt(
            np.diff(fine_sample[0]) ** 2
            + np.diff(fine_sample[1]) ** 2
            + np.diff(fine_sample[2]) ** 2
        )
    )
    dist = np.insert(dist, 0, 0)  # 0 inserted for first point

    # Calculates number of samples and generates distances to sample at with even spacing according
    # to arc length approximation, then maps arc length -> scipy parameterization
    samples = int(dist[-1] / spacing)
    target_dist = np.linspace(0, samples * spacing, samples + 1)
    u_spaced = np.interp(target_dist, dist, u_fine)

    # Evaluates spline
    sampled = splev(u_spaced, spline)

    # Reparameterizes the spline based on accurate arc length if the option is enabled
    if accurate_param:
        spline, _ = splprep(sampled, u=target_dist, s=0)

    return target_dist, sampled, spline



# Reads gpx file
with open(FILE) as file:
    gpx = gpxpy.parse(file)


trm = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)

# 0 - Outside/Left, 1 - Inside/Right, 2 - Center
# x, y, z:
#   1st dim - outside/inside
#   2nd dim - [[x], [y], [z]]
track: list[list | np.ndarray] = [[[], [], []], [[], [], []], [[], [], []]]

# Origin
zero_pt = gpx.tracks[0].segments[0].points[0]  # lat long
glob_zero_x, glob_zero_y = trm.transform(
    zero_pt.longitude, zero_pt.latitude
)  # cartesian

# Fills in track, generates x, y, z coordinates from (lat, long, elev)
for t in gpx.tracks:
    if t.name == "Outside":
        i = 0
    elif t.name == "Inside":
        i = 1
    else:
        raise ValueError(f"{FILE} must contain only tracks 'Outside' and 'Inside'")

    for s in t.segments:
        for p in s.points:
            x, y = trm.transform(p.longitude, p.latitude)

            # Shifts coordinates according to global zero
            track[i][0].append(x - glob_zero_x)
            track[i][1].append(y - glob_zero_y)
            track[i][2].append(p.elevation)
    track[i] = np.asarray(track[i])


# Set minimum elevation to 0
z_min = min(np.min(track[0][2]), np.min(track[1][2]))
track[0][2] -= z_min
track[1][2] -= z_min

# Query for nearest point on the right for each left point to generate the centerline
center_nn = KDTree(np.transpose(track[1]))
_, c_nearest = center_nn.query(np.transpose(track[0]))
for i, c in enumerate(c_nearest):  # type: ignore

    # Loop over each axis
    for j in range(3):
        track[2][j].append((track[0][j][i] + track[1][j][c]) / 2.0)  # type: ignore

track[2] = np.asarray(track[2])

# interpolated_track = []
# dists = []
# splines = []

# for i, t in enumerate(track):
#     dist, sampled, spline = interpolate(t, 0.1 if i < 2 else RESOLUTION)  # type: ignore

#     interpolated_track.append(sampled)
#     interpolated_track[i] = np.asarray(interpolated_track[i])
#     dists.append(dist)
#     splines.append(spline)


# out_nn = KDTree(np.transpose(interpolated_track[0][:2]))
# in_nn = KDTree(np.transpose(interpolated_track[1][:2]))


# _, o_nearest = out_nn.query(np.transpose(interpolated_track[2][:2]))
# _, i_nearest = in_nn.query(np.transpose(interpolated_track[2][:2]))

# # [out/1, in/r, center]
# s_track = [
#     interpolated_track[0][:, o_nearest],
#     interpolated_track[1][:, i_nearest],
#     interpolated_track[2],
# ]

# print(len(interpolated_track[0][:, o_nearest]), len(dists[2]))
# print(s_track[0].shape, dists[2].shape)
# spline_l, _ = splprep(s_track[0], u=dists[2], s=0)
# spline_r, _ = splprep(s_track[1], u=dists[2], s=0)
# spline_c, _ = splprep(s_track[2], u=dists[2], s=0)

s_track = [0,0,0]
max_dist, spline_l, spline_r, spline_c, s_track[0], s_track[1], s_track[2] = parameterized_interpolation(track)


plots = []
# for t in interpolated_track:
#     ax.plot(*t)

# for t in track:
#     ax.scatter(*t)

for t in s_track:
    plots.append(go.Scatter3d(x=t[0], y=t[1], z=t[2], name="original"))

for s in (spline_l, spline_r, spline_c):
    print(len(s))
    x, y, z = splev(np.linspace(0, max_dist, (int)(max_dist // 5)), s)
    plots.append(go.Scatter3d(x=x, y=y, z=z, name="param splines"))


fig = go.Figure(data=plots)
fig.update_layout(scene=dict(aspectmode="data"))

fig.show()
