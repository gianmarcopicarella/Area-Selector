# Method for parallel analysis of patches

# Third party imports
import numpy as np
from scipy.spatial import ConvexHull

# Local imports
from convex_area_selector.heuristic.heuristic import convex_hull_shrinkage
from convex_area_selector.utils import utils


def analyse_patches(points, max_area, verbose=False, key=None):
    if verbose and (key is not None):
        x_index = key[0]
        y_index = key[1]
        print(f"({x_index},{y_index})")

    if len(points) < 3:
        return 0, [], key

    # Remove duplicates and print if any are present
    points = utils.remove_duplicate_points(points)
    remaining_indices = list(range(len(points)))

    try:
        convex_hull_scipy = ConvexHull(points)
    except:
        return 0, [], key  # points collinear

    convex_hull_indices = utils.get_bottom_and_reverse(
        list(convex_hull_scipy.vertices), points
    )

    # In case area condition if fulfilled and while loop not entered
    convex_hull = [points[i] for i in convex_hull_indices]

    if verbose:
        ch_points = np.array(points)[convex_hull_indices]
        print("Convex hull covering all points:")
        print(ch_points)
        print()

    total_area = convex_hull_scipy.volume

    if verbose:
        print(f"Total initial area: {total_area}")

    drop_dictionary = {}

    while total_area > max_area and len(remaining_indices) > 3:
        # The drop dictionary is used to store the area lost by shrinking the convex
        # hull for the specified three neighbouring points on the CH
        (
            convex_hull_indices,
            remaining_indices,
            drop_dictionary,
            area_removed,
        ) = convex_hull_shrinkage(
            points,
            remaining_indices,
            convex_hull_indices,
            drop_dictionary,
            total_area,
            max_area,
        )

        if remaining_indices is None:
            # No area could be reduced, except to 0
            return 0, [], key

        total_area = total_area - area_removed
        convex_hull = [points[i] for i in convex_hull_indices]

        if verbose:
            remaining_points = [points[i] for i in remaining_indices]
            print()
            print("Points currently on the convex polygon:")
            print(convex_hull)
            print("Points remaining:")
            print(remaining_points)
            print(f"Total area: {total_area}")

    if total_area > max_area + 1e-9:
        if verbose:
            print("No solution found")
        return 0, [], key

    return len(remaining_indices), convex_hull, key
