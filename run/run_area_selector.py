import pathlib
import json
import os

import numpy as np

import src.constants as constants
import pandas as pd
from numpy.random import default_rng

from src.convex_area_selector.convex_polygon_finder import analyse_detections

rng = default_rng()

# Settings
max_area = 2
verbose = False
record_screen = False
n_areas = 3
plot_end_result = False
n_jobs = 4
patch_size = 3
step_size = patch_size * 0.66
algorithm = "switch"  # exact, heuristic, switch
exact_max = 25
mpp = 0.23
buffer = False
should_scale = False

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
    "should_scale": should_scale
}

max_diameter = 1.414 * patch_size
path_to_real = os.path.join(constants.PATH_TO_EXPERIMENTS, "real")

report = {"count": 0, "results": []}

for i in range(constants.REAL_BENCHMARKS_COUNT):
    path_to_data = os.path.join(path_to_real, str(i), "points_0.json")
    json_data = json.load(fp=open(path_to_data, "r"))
    df = pd.DataFrame()
    df["x_absolute"] = [np.float64(p["x"]) for p in json_data["points"]]
    df["y_absolute"] = [np.float64(p["y"]) for p in json_data["points"]]
    results = analyse_detections(df, settings_dict)
    entry = {
        "name": f"AreaSelector/Real/{i}/0/iterations:0",  # test_idx/diameter_idx/
        "max_area": max_area,
        "max_diameter": max_diameter,
        "max_count": 18446744073709551615
    }
    if len(results) > 0:
        convex_area = {
            "area": results[0]["area"],
            "count": results[0]["mitosis"],
            "hull_indices": [],
            "diameter_indices": []
        }

        # find diameter indices and collect hull indices
        for p in results[0]["polygon"]:
            for i in range(df.shape[0]):
                row = df.iloc[i]
                if row["x_absolute"] == p[0] and row["y_absolute"] == p[1]:
                    convex_area["hull_indices"].append(i)
                    break

        diameter_norm = -1
        for i in convex_area["hull_indices"]:
            for j in convex_area["hull_indices"]:
                if i == j: continue
                dist = np.linalg.norm(np.array([json_data["points"][i]['x'], json_data["points"][i]['y']]) -
                                      np.array([json_data["points"][j]['x'], json_data["points"][j]['y']]))
                if dist > diameter_norm:
                    diameter_norm = dist
                    convex_area["diameter_indices"] = [i, j]

        entry["convex_area"] = convex_area

    report["results"].append([entry])
    report["count"] += 1

json.dump(report, fp=open(constants.PATH_TO_AREA_SELECTOR_RESULTS, "w"))
