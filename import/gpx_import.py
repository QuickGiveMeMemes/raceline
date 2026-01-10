from dataclasses import dataclass
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gpxpy
import gpxpy.gpx
from pyproj import Transformer


matplotlib.use("Qt5Agg")
    
FILE = "Track.gpx" 

with open(FILE) as file:
    gpx = gpxpy.parse(file)

trm = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)

# 0 - Outside/Left, 1 - Inside/Right
# x, y, z:
#   lst dim - outside/inside
#   2nd dim - [[x], [y], [z]]
track = \
    [
        [[], [], []],
        [[], [], []]
    ]

# Origin
zero_pt = gpx.tracks[0].segments[0].points[0] # lat long
glob_zero_x, glob_zero_y = trm.transform(zero_pt.longitude, zero_pt.latitude) # cartesian

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

    # Convert to numpy arrays
    for j in range(len(track[i])):
        track[i][j] = np.asarray(track[i][j])

# Set minimum elevation to 0
z_min = min(np.min(track[0][2]), np.min(track[1][2])) 
track[0][2] -= z_min
track[1][2] -= z_min

   



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(track[0][0], track[0][1], track[0][2])
ax.scatter(track[1][0], track[1][1], track[1][2]) 
ax.set_aspect('equal', adjustable='box')

plt.show()
