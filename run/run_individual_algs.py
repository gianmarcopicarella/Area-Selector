import pathlib
import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull

import src.convex_area_selector.exact.exact as exact
import src.convex_area_selector.heuristic.analyse_patches as analyse_patches
from src.convex_area_selector.utils.utils import create_buffer

path = pathlib.PurePath(__file__)

# Load the example points from a CSV file
data = pd.read_csv(path.parents[1] / "example_data" / "example_points.csv")
points = list(data.itertuples(index=False, name=None))

print()
print("Starting comparison of exact and heuristic BAME convex hull algorithms")
print()
start_time = time.time()

result_exact = exact.exact_convex_hull(points, 2.0)

current_time = time.time()
print(f"Exact algorithm elapsed time: {current_time - start_time}")
start_time = current_time

result_heuristic = analyse_patches.analyse_patches(points, 2.0)

current_time = time.time()
print(f"heuristic algorithm elapsed time: {current_time - start_time}")
start_time = current_time

print()
print("Exact result")
print(f"{result_exact[0]} -- {result_exact[1]}")

print()
print("Heuristic result")
print(f"{result_heuristic[0]} -- {result_heuristic[1]}")
print()
figure, ax = plt.subplots(figsize=(10, 7.5))

x_coords = [x[0] for x in points]
y_coords = [x[1] for x in points]

# Plotting scattered points
scat_background = ax.scatter(x_coords, y_coords, s=2, c="black")

### Plotting heuristic result
buffered_poly = create_buffer(result_heuristic[1], buffer_size=0.015)

x_convex_polygon, y_convex_polygon = [list(z) for z in zip(*buffered_poly, strict=True)]

x_convex_polygon.append(x_convex_polygon[0])
y_convex_polygon.append(y_convex_polygon[0])

ax.plot(x_convex_polygon, y_convex_polygon, "b", label="Heuristic")


# Heuristic area
convex_hull_scipy = ConvexHull(result_heuristic[1])
total_area = convex_hull_scipy.volume
print(f"Heuristic area: {total_area}")

### Plotting exact buffer
buffered_poly = create_buffer(result_exact[1], buffer_size=0.015)

x_convex_polygon, y_convex_polygon = [list(z) for z in zip(*buffered_poly, strict=True)]

x_convex_polygon.append(x_convex_polygon[0])
y_convex_polygon.append(y_convex_polygon[0])

ax.plot(x_convex_polygon, y_convex_polygon, "r", label="Exact")

# Exact area
convex_hull_scipy = ConvexHull(result_exact[1])
total_area = convex_hull_scipy.volume
print(f"Exact area: {total_area}")

plt.title("Comparison of Heuristic and Exact BAME Convex Hull Algorithms", fontsize=18)
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.legend()

plt.savefig(path.parents[1] / "plots" / "BAME_CH_comparison.png")
plt.show()
plt.show()
