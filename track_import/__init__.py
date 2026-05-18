"""
## Module for track generation.

**usage**: ```python -m track_import.import [-h] -g GPX [-s SAVEFILE] [-c CONFIG] [-p] [-r] [--solver SOLVER]```

**options**:

  ```-h```, ```--help```            show this help message and exit

  ```-g GPX_SOURCE```, ```--gpx GPX_SOURCE```
                        Source path to track gpx file.

  ```-s SAVEFILE```, ```--savefile SAVEFILE```
                        Destination path of fitted track.

  ```-c CONFIG````, ```--config CONFIG```       
                        Path to config file.
  ```-p```, ```--plot```
                        Toggles on plotting.
  ```-r```, ```--refine```
                        Toggles mesh refinement.
  ```--solver SOLVER```
                        Solver to use (mumps, ma57, ma86, ma97, etc.).
"""
