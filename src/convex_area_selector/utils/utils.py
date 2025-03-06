####
## A module containing utility functions
## Author - Sjoerd de Vries
####

# Standard libraries
import math

# Third party imports
import numpy as np
import shapely.geometry
from scipy.spatial import ConvexHull


# Calculate a buffer zone around a convex hull
def create_buffer(points, buffer_size=1):
    shapely_polygon = shapely.geometry.Polygon(points)
    buffer_zone = shapely_polygon.buffer(buffer_size)
    buffer_coordinates = list(buffer_zone.exterior.coords)

    return buffer_coordinates


def point_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_triangle_area(a, b, c):
    """Calculates the area of a triangle between points
    a, b and c.

    Parameters
    ----------
    a : (float, float)
        the first point
    b : (float, float)
        the second point
    c : (float, float)
        the third point

    Returns
    -------
    float
        area of the triangle between the points
    """
    area = abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) * 0.5
    return area


def get_bottom_and_reverse(indices, points):
    """Take the already sorted indices of a convex hull generated
    by the SciPy ConvexHull function. Finds the bottom point and
    reverses the order to make them suitable for the heuristic
    algorithm.

    Parameters
    ----------
    indices : [int]
        indices of the convex hull vertices
    points : [(float,float)]
        points available to the overall problem

    Returns
    -------
    [int]
        list of indices on the convex hull in the order needed
        for the heuristic algorithm
    """

    start_pt = points[indices[0]]
    start_id = indices[0]

    for x in range(1, len(indices)):
        temp_index = indices[x]
        if points[temp_index][1] < start_pt[1]:
            start_pt = points[temp_index]
            start_id = x
        elif (
            points[temp_index][1] == start_pt[1] and points[temp_index][0] > start_pt[0]
        ):
            start_pt = points[temp_index]
            start_id = x

    new_order = indices[start_id::-1] + indices[-1:start_id:-1]

    return new_order


def find_top_areas(mitosis_dict, ch_dict):
    top_candidates = {}
    top_areas = {}
    if len(mitosis_dict) < 2:
        result = mitosis_dict
    else:
        max_key = max(mitosis_dict, key=mitosis_dict.get)
        max_mitosis = max(mitosis_dict.values())

        for key, val in mitosis_dict.items():
            if val == max_mitosis:
                top_candidates[(key)] = val
                top_areas[(key)] = ch_dict[key]

        key_smallest_area = None
        smallest_area = np.finfo(np.float64).max
        for key, val in top_areas.items():
            convex_hull_scipy = ConvexHull(val)
            total_area = convex_hull_scipy.volume
            if total_area < smallest_area:
                smallest_area = total_area
                key_smallest_area = key

        """print(
            f"key: {key_smallest_area}, area smallest: {smallest_area}, nr of candidates: {len(top_areas)}"
        )"""
        result = {}
        result[(key_smallest_area)] = max_mitosis

    return result


# Calculates the angle the line between two points makes with the x-axis
# from -pi to pi, with -pi/pi on the left, 0 on the right of the x-axis and -1/2 pi
# on the bottom
def calculate_angle_wrt_x_axis(point_1, point_2):
    return math.atan2(
        (point_2[1] - point_1[1]),
        (point_2[0] - point_1[0]),
    )


# Check if two points are colinear by comparing their angles w.r.t. a third point
def colinear_by_angle(angle_1, angle_2):
    return abs(angle_1 - angle_2) < 1e-12


# Find the sets of indices of duplicate values in a sorted list
def find_duplicates(sorted_list):
    current_duplicates = []
    duplicate_dict = {}

    for i in range(len(sorted_list) - 1):
        if sorted_list[i] == sorted_list[i + 1]:
            current_duplicates.append(i)
        else:
            if len(current_duplicates) > 0:
                current_duplicates.append(i)

                for j in current_duplicates[1:]:
                    index_to_use = current_duplicates.index(j) - 1
                    duplicate_dict[j] = current_duplicates[index_to_use::-1]
                current_duplicates = []

        # In case the final element is a duplicate
        if (i == (len(sorted_list) - 2)) & (len(current_duplicates) > 0):
            current_duplicates.append(i + 1)

            for j in current_duplicates[1:]:
                index_to_use = current_duplicates.index(j) - 1
                duplicate_dict[j] = current_duplicates[index_to_use::-1]
            current_duplicates = []

    return duplicate_dict


# Check if new_angle lies between angle_1 and angle_2
# Second angle is defined in clockwise direction
def angle_within_bounds(angle_1, angle_2, new_angle, colinear_included=True):
    within_bounds = False

    # # Checking if point is on line between start and end point
    if colinear_by_angle(angle_2, new_angle) or colinear_by_angle(angle_1, new_angle):
        # Just checks if angels are nearly the same to prevent floating point errors
        if colinear_included:
            return True
        else:
            return False

    if angle_1 <= 0 and angle_2 > 0:
        if angle_1 >= new_angle or new_angle >= angle_2:
            within_bounds = True

    elif angle_1 >= new_angle and new_angle >= angle_2:
        within_bounds = True

    return within_bounds


# calculate the area of a polygon
def polygon_area(points):
    x, y = [list(z) for z in zip(*points)]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# adds two angels safely
def angle_addition(angle1, angle2):
    new_angle = angle1 + angle2

    if new_angle > math.pi:
        new_angle = new_angle - (2 * math.pi)

    return new_angle


# Takes a starting point and a list of points
# and sorts these points w.r.t. the x-axis in counterclockwise
# fashion (as defined by calculate_angle_wrt_x_axis)
# if two points have the same angle, the one closest to the starting point
# is placed first
def sort_by_angle(start_point, points):
    other_points = []

    for other_point in points:
        if other_point != start_point:
            angle = calculate_angle_wrt_x_axis(start_point, other_point)
            # angles.append(angle)
            # calculate the euclidean distance between the points
            distance = point_distance(start_point, other_point)
            # distances.append(distance)
            other_points.append((angle, distance, other_point))
    angle_distance_sorted = sorted(other_points, key=lambda e: (e[0], e[1]))

    sorted_angles = [i[0] for i in angle_distance_sorted]
    sorted_distances = [i[1] for i in angle_distance_sorted]
    sorted_points = [i[2] for i in angle_distance_sorted]

    return sorted_angles, sorted_points


# Takes a starting point and a list of points
# and sorts these points w.r.t. the x-axis in clockwise
# fashion (as defined by calculate_angle_wrt_x_axis)
# if two points have the same angle, the one closest to the starting point
# is placed first
def sort_by_angle_index(start_point, points, point_indices, start_angle):
    other_points = []

    for point_index in point_indices:
        other_point = points[point_index]
        angle = calculate_angle_wrt_x_axis(start_point, other_point)
        # - 1e-9 such that points on the start_angle line are properly handled
        adjusted_angle = angle_addition(angle - 1e-9, (math.pi - start_angle))
        # calculate the euclidean distance between the points
        distance = point_distance(start_point, other_point)
        other_points.append((adjusted_angle, distance, other_point, point_index, angle))
    angle_distance_sorted = sorted(
        other_points, key=lambda e: (e[0], -e[1]), reverse=True
    )

    sorted_adjusted_angles = [i[0] for i in angle_distance_sorted]
    sorted_distances = [i[1] for i in angle_distance_sorted]
    sorted_points = [i[2] for i in angle_distance_sorted]
    sorted_indices = [i[3] for i in angle_distance_sorted]
    sorted_angles = [i[4] for i in angle_distance_sorted]

    return (
        sorted_adjusted_angles,
        sorted_points,
        sorted_distances,
        sorted_indices,
        sorted_angles,
    )


# Function to efficiently iterate through angles multiple times in a row
def eliminate_by_angle_index(
    current_index, start_index, sorted_angles, interior_indices=[]
):
    """Iterate through a sorted list of angles and return the indices of
    those points that can still potentially be included.

    All points between current_index and start_index are included by default,
    as these points have been found suitable in previous iterations.

    Parameters
    ----------
    current_index : int
        index for which to start
    start_index : int
        index up until points were included in the previous iteration,
        and from which we have to start checking points in the current iteration
    sorted_angles : [float]
        a list of angles sorted in counterclockwise fashion

    Returns
    -------
    included_indices : [int]
        the indices of the points that can still be included
    """

    # if no points were included in the last iteration, increase start_index
    if current_index == start_index:
        start_index = (start_index + 1) % len(sorted_angles)

    # get the indices of all points
    all_indices = list(range(len(sorted_angles)))

    # from current to start always included. To check, from start to current.
    if start_index > current_index:
        ordered_indices_to_check = (
            all_indices[start_index:] + all_indices[:current_index]
        )
        included_indices = all_indices[current_index + 1 : start_index]
    else:
        ordered_indices_to_check = all_indices[start_index:current_index]
        included_indices = all_indices[current_index + 1 :] + all_indices[:start_index]

    # remove points that are already included
    ordered_indices_to_check = [
        i for i in ordered_indices_to_check if i not in interior_indices
    ]

    angle_limit = angle_addition(sorted_angles[current_index], math.pi)

    for index in ordered_indices_to_check:
        if angle_within_bounds(
            angle_limit, sorted_angles[current_index], sorted_angles[index]
        ):
            included_indices.append(index)
        else:
            break

    return included_indices


def elimiminate_by_angle(elimination_angle, sorted_angles):
    remaining_indices = []
    angle_limit = angle_addition(elimination_angle, math.pi)

    for index in range(len(sorted_angles)):
        if angle_within_bounds(angle_limit, elimination_angle, sorted_angles[index]):
            remaining_indices.append[index]

    return remaining_indices


def remove_duplicate_points(points):
    # remove duplicates & print if any are present
    original_length = len(points)
    unique_points = [
        item for index, item in enumerate(points) if item not in points[:index]
    ]

    if len(unique_points) < original_length:
        print("Duplicate points found and removed")

    return unique_points
