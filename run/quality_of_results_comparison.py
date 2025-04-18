import json
import os
import shutil
import time

import pandas as pd
import numpy as np
import thesis

from src import constants
from src.convex_area_selector.convex_polygon_finder import analyse_detections, create_patch_dict


def prepare_path(path):
    if os.path.isdir(path): shutil.rmtree(path)
    os.makedirs(path)


def ann(points):
    distances = []
    for i, p in enumerate(points):
        smallest_dist = np.inf
        for j, q in enumerate(points):
            if i == j: continue
            smallest_dist = min(smallest_dist, np.linalg.norm(np.array(p) - np.array(q)))
        distances.append(smallest_dist)
    return np.mean(distances), np.std(distances)


default_settings = \
    {
        "max_area": 4,
        "verbose": False,
        "record_screen": False,
        "plot_end_result": False,
        "n_jobs": 4,
        "patch_size": 3,
        "step_size": 3 * 0.66,
        "algorithm": "switch",
        "exact_max": 25,
        "microns_per_pixel": 0.23,
        "buffer": False,
        "should_scale": False,
        "max_diameter": np.inf
    }

path_to_real = os.path.join(constants.PATH_TO_EXPERIMENTS, "real")
max_count = 18446744073709551615


def run_antipodal(points, settings):
    out = thesis.AntipodalOptimized(points, max_count, settings["max_area"], settings["max_diameter"], True, True)
    if out is not None:
        return out[0], out[1], out[2], np.linalg.norm(np.array(points[out[3][0]]) - np.array(points[out[3][1]]))


def run_area_selector(points, settings):
    df = pd.DataFrame()
    df["x_absolute"] = [p[0] for p in points]
    df["y_absolute"] = [p[1] for p in points]
    results = analyse_detections(df, settings)
    if len(results) > 0:
        hull_indices, diam_indices = [], []
        for p in results[0]["polygon"]:
            for i in range(df.shape[0]):
                row = df.iloc[i]
                if row["x_absolute"] == p[0] and row["y_absolute"] == p[1]:
                    hull_indices.append(i)
                    break
        diameter_norm = np.max([np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                                for i in hull_indices for j in hull_indices])
        return results[0]["area"], results[0]["mitosis"], hull_indices, diameter_norm
    return None


def run_eppstein_tiled(points, settings):
    df = pd.DataFrame()
    df["x_absolute"] = [p[0] for p in points]
    df["y_absolute"] = [p[1] for p in points]
    patch_dict, ordered_patch_list = create_patch_dict(df, settings)
    result = None
    for key in ordered_patch_list:
        patch_points = patch_dict[key]
        if result is not None and result[1] > len(patch_points):
            break
        out = thesis.Eppstein(patch_points, max_count, settings["max_area"], True, True)
        if out is not None and (result is None or result[1] < out[1]):
            real_hull_indices = []
            for i in out[2]:
                for j, p in enumerate(points):
                    if p == patch_points[i]:
                        real_hull_indices.append(j)
                        break
            assert (len(real_hull_indices) == len(out[2]))
            result = (out[0], out[1], real_hull_indices)

    if result is not None:
        diameter_norm = np.max([np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                                for i in result[2] for j in result[2]])
        return result[0], result[1], result[2], diameter_norm

    return None


def optimal(opt, value):
    if value == opt:
        return r"\textbf{" + str(value) + "}"
    return str(value)


# Generate a comparison table in LaTeX
table_header = \
    r"""
\begin{table}
    \small
    \centering
    \begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
        \hline
        \multicolumn{3}{|c|}{\textbf{Dataset}} & \multicolumn{3}{c|}{\textbf{AP}} & \multicolumn{3}{c|}{\textbf{AS ($s=1.98$)}} & \multicolumn{3}{c|}{\textbf{AS ($s=0.5$)}} & \multicolumn{3}{c|}{\textbf{ET ($s=0.5$)}} \\
        \cline{1-15}
        Index & C & ANN & C & A & D & C & A & D & C & A & D & C & A & D \\
        \hline
    """ + "\n"

table_footer = \
    r"""
    \hline
    \end{tabular}
    \caption{Comparison of the results obtained by applying the antipodal algorithm (AP), the area selector method (AS) and Eppstein et al.'s algorithm tiled (ET) to the 10 most populated point sets available in the real-world dataset after applying a probability threshold $t=0.86$. AS is run for step sizes $s=1.98, s=0.5$. ET is run for step size $s=0.5$. All the values for area (A) and diameter (D) are expressed in $\text{mm}^2$ and $\text{mm}$ respectively.}
    \label{tab:comparison}
\end{table}
"""


def fill_figure(figure_width, picture_width, picture_height, poly_points_str, color, path_to_dat):
    base_figure = \
        r"""
    \begin{subfigure}{<FIG_WIDTH>\textwidth}
        \centering
        \begin{tikzpicture}
            \begin{axis}[
                grid=major,
                width=<PIC_WIDTH>cm,
                height=<PIC_HEIGHT>cm,
                axis equal,
                xtick distance=5,
                ytick distance=5,
                xticklabels={},
                yticklabels={}
            ]
            \addplot[only marks, mark=*, color=<COLOR>, mark size=0.4pt]
            coordinates {<POLY_POINTS>};
            \addplot[thick, <COLOR>, fill=<COLOR>!20, opacity=1]
            coordinates {<POLY_POINTS>};

            \addplot[only marks, mark=*, color=black, mark size=0.1pt] table {<PATH>};
            \end{axis}
        \end{tikzpicture}
        \caption{}
    \end{subfigure}
    """
    figure = base_figure.replace("<FIG_WIDTH>", str(figure_width))
    figure = figure.replace("<PIC_WIDTH>", str(picture_width))
    figure = figure.replace("<PIC_HEIGHT>", str(picture_height))
    figure = figure.replace("<POLY_POINTS>", poly_points_str)
    figure = figure.replace("<COLOR>", color)
    return figure.replace("<PATH>", path_to_dat)


table_rows = ""
figures = ""

times = []

for index in range(constants.REAL_BENCHMARKS_COUNT):
    path_to_data = os.path.join(path_to_real, str(index), "points_0.json")
    json_data = json.load(fp=open(path_to_data, "r"))
    points = [(p["x"], p["y"]) for p in json_data["points"]]

    ann_avg, ann_stddev = ann(points)

    default_settings["step_size"] = 0.66 * 3
    default_settings["max_diameter"] = 4
    start_time = time.time()
    ap_result = run_antipodal(points, default_settings)
    times.append(time.time() - start_time)

    default_settings["max_diameter"] = np.inf
    as_result = run_area_selector(points, default_settings)

    default_settings["step_size"] = 0.5
    as_result_050 = run_area_selector(points, default_settings)
    et_result_050 = run_eppstein_tiled(points, default_settings)

    max_sol_count = max(ap_result[1], as_result[1], as_result_050[1], et_result_050[1])

    table_rows += f"{index} & {len(points)} & ${np.round(ann_avg, 2)}\\pm{np.round(ann_stddev, 2)}$ & "
    table_rows += f"{optimal(max_sol_count, ap_result[1])} & {np.round(ap_result[0], 2)} & {np.round(ap_result[3], 2)} & "
    table_rows += f"{optimal(max_sol_count, as_result[1])} & {np.round(as_result[0], 2)} & {np.round(as_result[3], 2)} & "
    table_rows += f"{optimal(max_sol_count, as_result_050[1])} & {np.round(as_result_050[0], 2)} & {np.round(as_result_050[3], 2)} & "
    table_rows += f"{optimal(max_sol_count, et_result_050[1])} & {np.round(et_result_050[0], 2)} & {np.round(et_result_050[3], 2)}" + r"\\" + "\n"

    dat = "\n".join(f"{p[0]} {p[1]}" for p in points)
    path_to_dat = os.path.join(constants.PATH_TO_LATEX, "dat", "real", str(index))
    prepare_path(path_to_dat)
    with open(os.path.join(path_to_dat, "points_0.dat"), "w") as out_file:
        out_file.write(dat)

    figures += r"\begin{figure}[h]" + "\n" + "\\centering\n" + "\\small\n"

    figures += fill_figure(0.24, 4, 4, " ".join(f"({points[j][0]}, {points[j][1]})" for j in ap_result[2] + [ap_result[2][0]]), "Red", path_to_dat)
    figures += r"\hfill"
    figures += fill_figure(0.24, 4, 4, " ".join(f"({points[j][0]}, {points[j][1]})" for j in as_result[2] + [as_result[2][0]]), "Cyan", path_to_dat)
    figures += r"\hfill"
    figures += fill_figure(0.24, 4, 4, " ".join(f"({points[j][0]}, {points[j][1]})" for j in as_result_050[2] + [as_result_050[2][0]]), "Green", path_to_dat)
    figures += r"\hfill"
    figures += fill_figure(0.24, 4, 4, " ".join(f"({points[j][0]}, {points[j][1]})" for j in et_result_050[2] + [et_result_050[2][0]]), "RoyalPurple", path_to_dat)
    figures += r"\caption{Convex areas found by AP (a), AS using $s=1.98$ (b), AS using $s=0.5$ (c) and ET using $s=0.5$ (d) for point set $" + str(index) + "$.}\n"
    figures += r"\end{figure}"

    print("File ", str(index), "done")


print(table_header + table_rows + table_footer)
print(figures)
print(times)
