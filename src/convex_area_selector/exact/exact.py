####
## An exact algorithm for calculating a convex polygon of maximum given area
## containing as many points as possible.
## Author - Sjoerd de Vries
####

# Standard libraries
import math

# Local imports
from src.convex_area_selector.utils import utils


def exact_convex_hull(points, max_area, verbose=False, key=None):
    if verbose and (key is not None):
        x_index = key[0]
        y_index = key[1]

    if len(points) < 3:
        return 0, [], key

    result_dict = {}

    # Remove duplicates and print if any are present
    points = utils.remove_duplicate_points(points)

    # Iterate over all points to find the convex hull containing the most points
    for start_point in points:
        path = [start_point]
        remaining_points = points.copy()
        remaining_points.remove(start_point)
        interior_points = []

        # Recursive procedure to construct the convex hull
        result_dict[start_point] = recursive_search(
            path, remaining_points, max_area, interior_points
        )

    mitosis = 0
    path = []

    for result in result_dict.items():
        if result[1] is not None:
            if result[1][0] > mitosis:
                mitosis, area, path = result[1]

    if verbose:
        print(f"({x_index},{y_index})")
        print(f"n points: {len(points)}")
        print(f"n mitosis: {mitosis}")
        print()

    return mitosis, path, key


# Takes an existing path, checks if points can be added (recursively).
# Each iteration the list of possible points is reduced further
def recursive_search(path, points, max_area, interior_points=[], tolerance=1e-09):
    """Recursive search for the convex hull containing the most points
    of at most given area.

    The Hull is constructed by adding points to the path, starting from the
    left side of the horizontal axis through the last point in path,
    checking points in counter-clockwise motion.

    Parameters
    ----------
    path : [(float,float)]
        list containing the path constructed so far
    points : [(float,float)]
        list containing the points that can still be added to the path
        or lie within the interior of the convex hull
    max_area : float
        the area constraint for the convex hull
    interior_points : [(float,float)], optional
        list containing the points that lie within the interior of the convex hull
        corresponding to the current path, by default []
    tolerance : float, optional
        small error correction used for the area calculation, by default 1e-09

    Returns
    -------
    best_result : (int, float, path)
        the number of mitosis, the area of the convex hull and the CH itself
    """

    # Set best_result to None, if no valid path is found, None is returned
    best_result = None

    # Initialize a potential path, i.e. back to the starting point
    potential_path = path + [path[0]]

    # Calculate the angle of the line between the last point and the first point
    # in path with respect to the the x-axis
    angle_to_start_point = utils.calculate_angle_wrt_x_axis(path[-1], path[0])
    angle_from_start_point = utils.angle_addition(angle_to_start_point, math.pi)

    # Check if a valid path already exists, so at least 3 points
    if len(path) > 2:
        potential_area = utils.polygon_area(potential_path)

        # And area not larger than max_area, with small error correction
        if potential_area > (max_area + tolerance):
            # Return None, as the potential polygon already exceeds the area constraint
            return best_result

    # Split on the number of points to be added
    if len(points) == 1:
        point_angle = utils.calculate_angle_wrt_x_axis(path[-1], points[0])
        remaining_points = []

        # Check if point in interior (works for len(path)=2 as well)
        if utils.angle_within_bounds(
            angle_from_start_point,
            angle_to_start_point,
            point_angle,
            colinear_included=True,
        ):
            # The point lies in interior
            interior_points.append(points[0])
        else:
            # The point lies on the convex hull
            new_path = path + [points[0]]

            best_result = recursive_search(
                new_path, remaining_points, max_area, interior_points.copy()
            )

    elif len(points) > 1:
        # Iterate over the remaining points, check if they can be added to the path

        # Bookkeeping
        start_point = path[-1]
        start_index = 1

        # Get the angles to all remaining points from the starting point,
        # w.r.t the x-axis. Sort for efficient iteration
        sorted_angles, sorted_points = utils.sort_by_angle(start_point, points)

        # Find the indices of duplicate values in a sorted list
        colinear_indices_dict = utils.find_duplicates(sorted_angles)

        for point_index, point in enumerate(sorted_points):
            # Iterate over points to potentially include

            temp_interior = interior_points.copy()

            colinear_indices = []
            # To deal with colinear points
            if point_index in colinear_indices_dict:
                for i in colinear_indices_dict[point_index]:
                    temp_interior.append(sorted_points[i])
                    colinear_indices.append(i)

            # Continue if angle already in interior
            if len(path) > 2:
                point_angle = sorted_angles[point_index]

                if utils.angle_within_bounds(
                    angle_from_start_point,
                    angle_to_start_point,
                    point_angle,
                    colinear_included=False,
                ):
                    remaining_points = sorted_points[point_index + 1 :]
                    interior_points.append(point)

                    continue

            # If not is_interior:
            # Find points that potentially lie in or on the convex hull
            # after inclusion of a point on the CH and add to remaining_indices
            remaining_indices = utils.eliminate_by_angle_index(
                point_index, start_index, sorted_angles, colinear_indices
            )

            if len(remaining_indices) == 0:
                # +1 as in the next iteration, point index is also +1
                start_index = point_index + 1
                remaining_points = []
            else:
                start_index = remaining_indices[-1]
                remaining_points = [sorted_points[i] for i in remaining_indices]

            # Remove points in interior from remaining points
            remaining_points = [
                point for point in remaining_points if point not in interior_points
            ]

            # Recursive search
            new_path = path + [point]
            result = recursive_search(
                new_path, remaining_points, max_area, temp_interior.copy()
            )

            if result is not None and result[1] > 0:  # Else all points on a line
                if best_result is None or result[0] > best_result[0]:
                    best_result = result

    # If no valid CH was found in the current method call,
    # the current path is the best result, given it contains at least 3 points.
    # The area was already checked earlier.
    if (best_result is None) and (len(path) > 2):
        potential_mitosis = len(path) + len(interior_points)
        best_result = potential_mitosis, potential_area, potential_path

    return best_result
