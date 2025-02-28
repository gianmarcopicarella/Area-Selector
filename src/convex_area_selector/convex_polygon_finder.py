####
## Convex polygon area selector
## Author - Sjoerd de Vries
####

# Standard libraries
import itertools
import time

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull

# Local imports
from .exact.exact import exact_convex_hull
from .heuristic.analyse_patches import analyse_patches
from .utils.utils import create_buffer, find_top_areas, polygon_area


def analyse_detections(data, settings, save_path=None):
    start_time = time.time()

    # If detections are empty, return an empty result
    if len(data) < 1:
        return [{"mitosis": 0, "area": 0, "polygon": ""}]

    patch_dict, ordered_patch_list = create_patch_dict(data, settings)

    # Most mitosis found within max_area mm2 so far
    mitosis_count_dict = {}
    ch_dict = {}

    if settings["algorithm"] == "switch":
        heuristic_ordered_patch_list = [
            key
            for key in ordered_patch_list
            if len(patch_dict[key]) > settings["exact_max"]
        ]
        exact_ordered_patch_list = [
            key
            for key in ordered_patch_list
            if len(patch_dict[key]) <= settings["exact_max"]
        ]

        method = analyse_patches
        if settings["verbose"]:
            print(f"patches analysed heuristicaly: {len(heuristic_ordered_patch_list)}")

        result_list_heuristic = Parallel(n_jobs=settings["n_jobs"])(
            delayed(method)(
                patch_dict[key],
                settings["max_area"],
                verbose=settings["verbose"],
                key=key,
            )
            for key in heuristic_ordered_patch_list
        )
        if settings["verbose"]:
            print("--- %s seconds ---" % (time.time() - start_time))
            print()
            print(f"patches analysed exactly: {len(exact_ordered_patch_list)}")

        method = exact_convex_hull
        result_list_exact = Parallel(n_jobs=settings["n_jobs"])(
            delayed(method)(
                patch_dict[key],
                settings["max_area"],
                verbose=settings["verbose"],
                key=key,
            )
            for key in exact_ordered_patch_list
        )

        if settings["verbose"]:
            print("--- %s seconds ---" % (time.time() - start_time))
            print()
        result_list = result_list_heuristic + result_list_exact
    else:
        if settings["algorithm"] == "heuristic":
            method = analyse_patches
        elif settings["algorithm"] == "exact":
            method = exact_convex_hull
        else:
            raise Exception(
                f'The selected algorithm: "{settings["algorithm"]}" is not valid'
            )

        result_list = Parallel(n_jobs=settings["n_jobs"])(
            delayed(method)(
                patch_dict[key],
                settings["max_area"],
                verbose=settings["verbose"],
                key=key,
            )
            for key in ordered_patch_list
        )

    for result in result_list:
        if result[0] >= 3:
            mitosis_count_dict[result[2]] = result[0]
            ch_dict[result[2]] = result[1]

    top_areas = find_top_areas(mitosis_count_dict, ch_dict)

    best_result = []
    for key in top_areas.keys():
        convex_hull_scipy = ConvexHull(ch_dict[key])
        total_area = convex_hull_scipy.volume
        if settings["verbose"]:
            print(f"key: {key}, area: {total_area}")

        if settings["buffer"]:
            polygon = create_buffer(ch_dict[(key)], buffer_size=0.01)
        else:
            polygon = ch_dict[(key)]
        best_result.append(
            {"mitosis": top_areas[key], "area": total_area, "polygon": polygon}
        )

    if settings["verbose"]:
        print()
        print(top_areas)

        print()
        print("--- %s seconds ---" % (time.time() - start_time))
        print()

    if settings["plot_end_result"]:
        plot_result(data, top_areas, ch_dict, save_path)

    if not best_result:
        best_result.append({"mitosis": 0, "area": 0, "polygon": ""})

    return best_result


def create_patch_dict(data, settings):
    # Scaling the data to milimeter scale
    if settings["should_scale"]:
        data["x_absolute_scaled"] = data["x_absolute"] / (
            1000 / float(settings["microns_per_pixel"])
        )
        data["y_absolute_scaled"] = data["y_absolute"] / (
            1000 / float(settings["microns_per_pixel"])
        )
    else:
        data["x_absolute_scaled"] = data["x_absolute"]
        data["y_absolute_scaled"] = data["y_absolute"]

    # Inspecting the data
    xmin = data["x_absolute_scaled"].min()
    xmax = data["x_absolute_scaled"].max()
    ymin = data["y_absolute_scaled"].min()
    ymax = data["y_absolute_scaled"].max()

    if settings["verbose"]:
        print(f"x-min: {xmin}")
        print(f"x-max: {xmax}")
        print(f"y-min: {ymin}")
        print(f"y-max: {ymax}")

    # Split into patches for faster processing
    patch_dict = {}

    # Construct an array that has as many elements as there are
    # steps between xmin and xmax
    if (xmax - xmin) < settings["patch_size"]:
        x_range = [xmin - 0.01]
    else:
        x_range = np.arange(xmin - 0.01, xmax, settings["step_size"])

    if (ymax - ymin) < settings["patch_size"]:
        y_range = [ymin - 0.01]
    else:
        y_range = np.arange(ymin - 0.01, ymax, settings["step_size"])

    for x_y in itertools.product(x_range, y_range):
        patch_dict[x_y] = []

    # Remove duplicates
    originial_shape = data.shape
    data.drop_duplicates(inplace=True)
    new_shape = data.shape
    if settings["verbose"]:
        print()
        print(f"There were {originial_shape[0]-new_shape[0]} duplicated removed")
        print()

    patch_dict_copy = patch_dict.copy()

    for (x, y), _ in patch_dict_copy.items():
        # Every area is equal in size to the patch_size
        x_end = x + settings["patch_size"]
        y_end = y + settings["patch_size"]

        # Find the rows that contain the points that are in the area
        df_x = data[data["x_absolute_scaled"].between(x, x_end)]
        df_y = data[data["y_absolute_scaled"].between(y, y_end)]

        # Join the two dataframes if they have common rows (both x and y in area)
        rows = df_x.merge(df_y, how="inner")

        if not rows.empty:
            points_x = rows["x_absolute_scaled"].to_list()
            points_y = rows["y_absolute_scaled"].to_list()

            for point in zip(points_x, points_y):
                patch_dict[(x, y)].append(point)
        else:
            # Remove all areas that are empty
            patch_dict.pop((x, y), True)

    ordered_patch_list = sorted(
        patch_dict, key=lambda k: len(patch_dict[k]), reverse=True
    )

    return patch_dict, ordered_patch_list


def plot_result(data, top_areas, ch_dict, save_path=None):
    plt.close()

    figure, ax = plt.subplots(figsize=(10, 7.5))

    # Setting labels
    plt.title("Dynamic Plot of the Mitosis Area Selector", fontsize=18)
    plt.xlabel("X", fontsize=18)
    plt.ylabel("Y", fontsize=18)

    # Plotting scattered points
    ax.scatter(data["x_absolute_scaled"], data["y_absolute_scaled"], s=1)

    for top_area in top_areas.keys():
        ### Plotting exact buffer
        buffered_poly = create_buffer(ch_dict[(top_area)], buffer_size=0.01)

        x_convex_polygon, y_convex_polygon = [
            list(z) for z in zip(*buffered_poly)
        ]

        x_convex_polygon.append(x_convex_polygon[0])
        y_convex_polygon.append(y_convex_polygon[0])

        ax.plot(x_convex_polygon, y_convex_polygon, "r")

        # Exact buffer area
        buffer_area = polygon_area(buffered_poly)
        print(f"buffer area: {buffer_area}")

    plt.gca().invert_yaxis()
    # Save the plot if save_path is not None
    if save_path:
        plt.savefig(save_path)

    plt.show()
