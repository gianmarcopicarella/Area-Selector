import pathlib

import pandas as pd
from numpy.random import default_rng

from convex_area_selector.convex_polygon_finder import analyse_detections

rng = default_rng()

# Settings
max_area = 2
verbose = False
record_screen = False
n_areas = 3
plot_end_result = True
n_jobs = 4
patch_size = 3
step_size = patch_size * 0.66
algorithm = "switch"  # exact, heuristic, switch
exact_max = 25
mpp = 0.23
buffer = True

settings_dict = {
    "max_area": max_area,
    "verbose": verbose,
    "record_screen": record_screen,
    "plot_end_result": plot_end_result,
    "n_jobs": n_jobs,
    "patch_size": patch_size,
    "step_size": step_size,
    "algorithm": algorithm,
    "exact_max": exact_max,
    "microns_per_pixel": mpp,
    "buffer": buffer,
}

# Getting the right paths
path = pathlib.PurePath(__file__)

# Load the example points from a CSV file
data = pd.read_csv(path.parents[1] / "example_data" / "example_detections.csv")

save_path = path.parents[1] / "plots" / "whole_slide_area_selector.png"


analyse_detections(data, settings_dict, save_path=save_path)
