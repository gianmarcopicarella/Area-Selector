####
## A heuristic algorithm for calculating a convex polygon of maximum given area
## containing as many points as possible, greedy algorithm, not optimal
## Author - Sjoerd de Vries
####

# Local imports
from src.convex_area_selector.utils.utils import (
    angle_within_bounds,
    calculate_angle_wrt_x_axis,
    calculate_triangle_area,
    sort_by_angle_index,
)


# New and improved function
def shrink_triangle_new(
    points,
    start_point_index,
    middle_point_index,
    end_point_index,
    eligible_point_indices,
    triangle_area,
):
    start_point = points[start_point_index]
    end_point = points[end_point_index]

    # Eligible points to replace the middle point with.
    eligible_point_indices.append(end_point_index)

    start_angle = calculate_angle_wrt_x_axis(start_point, points[middle_point_index])

    (
        sorted_adjusted_angles,
        sorted_points,
        sorted_distances,
        sorted_indices,
        sotred_angles,
    ) = sort_by_angle_index(start_point, points, eligible_point_indices, start_angle)

    best_area_lost_per_point = -1

    for enumeration_index, potential_ch_point_index in enumerate(sorted_indices):
        potential_ch_point = points[potential_ch_point_index]
        remaining_area = calculate_triangle_area(
            start_point, potential_ch_point, end_point
        )
        area_lost = triangle_area - remaining_area

        # Calculate how many points remain if current point is used as new boundary
        # of the convex hull
        indices_dropped = sorted_indices[:enumeration_index]

        start_point_angle = calculate_angle_wrt_x_axis(potential_ch_point, start_point)
        end_point_angle = calculate_angle_wrt_x_axis(potential_ch_point, end_point)

        # start checking from the next point, but exclude the point on the CH
        for point_to_check_index in range(
            enumeration_index + 1, len(sorted_points) - 1
        ):
            original_point_index = sorted_indices[point_to_check_index]

            point_to_check_angle = calculate_angle_wrt_x_axis(
                potential_ch_point, points[original_point_index]
            )

            if not angle_within_bounds(
                end_point_angle, start_point_angle, point_to_check_angle
            ):
                indices_dropped.append(original_point_index)

        indices_dropped.append(middle_point_index)
        area_lost_per_point = area_lost / len(indices_dropped)

        if area_lost_per_point > best_area_lost_per_point:
            best_addition_index = potential_ch_point_index
            best_area_lost = area_lost
            best_area_lost_per_point = area_lost_per_point
            best_indices_dropped = indices_dropped

    return (
        best_addition_index,
        best_area_lost,
        best_area_lost_per_point,
        best_indices_dropped,
    )


def convex_hull_shrinkage(
    points,
    remaining_indices,
    convex_hull_indices,
    drop_dictionary,
    starting_area,
    max_area,
):
    """A single iteration of the heurstic algorithm, resulting in a shrunk convex hull.
    The algorithm loops over the ordered points on the CH, looking at two points at
    a time with a third in between and calculating for all points within that triangle
    which point would result in the most area lost / point dropped if the CH was moved
    to that point instead of the current point inbetween.

    Parameters
    ----------
    points : [(float,float)]
        points available to the overall problem
    remaining_indices : [int]
        indices of the points that remain on the interior
        or boundary of the convex hull
    convex_hull_indices : [int]
        indices of the points on the convex hull
    drop_dictionary : {}
        _description_
    starting_area : float
        area of the current convex hull

    Returns
    -------
    convex_hull_indices, remaining_indices, drop_dictionary, total_area_lost
    """
    # Iterates over all points on the CH and calculates the optimal point to drop.
    for ch_index in range(len(convex_hull_indices)):
        start_point_index = convex_hull_indices[ch_index]
        middle_point_index = convex_hull_indices[
            (ch_index + 1) % len(convex_hull_indices)
        ]
        end_point_index = convex_hull_indices[(ch_index + 2) % len(convex_hull_indices)]

        # The area that can be gained by dropping the middle point
        # has already been calculated
        ch_key = (start_point_index, middle_point_index, end_point_index)
        if ch_key not in drop_dictionary:
            start_point = points[start_point_index]
            middle_point = points[middle_point_index]
            end_point = points[end_point_index]

            triangle_area = calculate_triangle_area(
                start_point, middle_point, end_point
            )

            # Instantiate angles between three consecutive points on the CH
            first_angle = calculate_angle_wrt_x_axis(start_point, middle_point)
            second_angle = calculate_angle_wrt_x_axis(start_point, end_point)

            eligible_point_indices = []

            # Iterate over all points to find those in the area which can be dropped
            for point_index in remaining_indices:
                # To prevent points in the convex hull to be seen as points in the area
                if point_index not in convex_hull_indices:
                    point = points[point_index]
                    point_angle = calculate_angle_wrt_x_axis(start_point, point)

                    if angle_within_bounds(first_angle, second_angle, point_angle):
                        eligible_point_indices.append(point_index)

            args = (
                points,
                start_point_index,
                middle_point_index,
                end_point_index,
                eligible_point_indices,
                triangle_area,
            )

            # For each three consecutive points on the CH, calculate the best point to
            # change the middle point to, the total area lost and area per point lost
            # by dropping that point, and which points are dropped
            result = shrink_triangle_new(*args)

            drop_dictionary[ch_key] = result

    # Iterate over dictionary to find point to drop
    key_to_change = ()
    max_area_lost_per_point = 0

    # Iterate over all possible areas to drop points from
    # To find the best points to drop
    for key in drop_dictionary:
        temp_area_lost_per_point = drop_dictionary[key][2]
        if temp_area_lost_per_point > max_area_lost_per_point:
            # To avoid ending up with 0 area
            if starting_area - drop_dictionary[key][1] > 1e-9:
                key_to_change = key
                max_area_lost_per_point = temp_area_lost_per_point

    if key_to_change == ():
        # No area can be lost without reducing area to 0
        return None, None, None, None

    to_change = drop_dictionary[key_to_change]

    # Check to see if not area is reduced beyond the max_area
    if (starting_area - to_change[1]) < max_area:
        # Check to see if there is not an area we can drop that loses
        # fewer points while still becoming within max_area
        current_n_points_lost = len(to_change[3])

        for key in drop_dictionary:
            temp_area_lost = drop_dictionary[key][1]
            temp_points_lost = drop_dictionary[key][3]
            temp_n_points_lost = len(temp_points_lost)

            temp_remaining_area = starting_area - temp_area_lost

            if temp_remaining_area < max_area and temp_remaining_area > 1e-9:
                if temp_n_points_lost < current_n_points_lost:
                    key_to_change = key
                    to_change = drop_dictionary[key_to_change]
                    current_n_points_lost = temp_n_points_lost

    # Changing the convex hull
    end_point_index = key_to_change[2]
    middle_point_index = key_to_change[1]
    new_ch_index = to_change[0]

    if new_ch_index == end_point_index:
        convex_hull_indices.remove(middle_point_index)
    else:
        index_to_replace = convex_hull_indices.index(middle_point_index)
        convex_hull_indices[index_to_replace] = new_ch_index

    # Removing all newly excluded points
    for point in to_change[3]:
        remaining_indices.remove(point)

    # Cleaning up the drop_dictionary
    keys_to_drop = []

    for key in drop_dictionary:
        if key_to_change[1] in [key[0], key[1], key[2]]:
            keys_to_drop.append(key)

    for key in keys_to_drop:
        drop_dictionary.pop(key)

    total_area_lost = to_change[1]

    return convex_hull_indices, remaining_indices, drop_dictionary, total_area_lost
