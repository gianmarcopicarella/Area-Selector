# Bounded Area Maximum Enclosing Convex Hull

This repository contains a number of algorithms for solving the problem of finding a convex polygon with bounded area, containing as many points as possible, also known as the Bounded Area Maximum Enclosing (BAME) Convex Hull problem.

The algorithms are implemented in Python and are used in the paper: "Breast cancer survival prediction using an automated mitosis detection pipeline" in *Journal of Pathology - Clinical Research*. The algorithms are used to find the convex polygon of at most a specified area that contains the most mitotic figures in a whole slide image.

## Installation

The code in this repository can be installed locally as a package by installing the
requirements in the `requirements.txt` file.

## Repository structure

The exact algorithm can be found in the `exact` folder, the heuristic algorithm in the `heuristic` folder.

The `convex_polygon_finder.py` file can be used to apply these algorithms to a whole slide image, optionally switching between their use depending on the number of mitotic figures. Furthermore, sliding windows are used to reduce the computational complexity. 

In the `run` folder, two scripts are provided that showcase uses of these algorithms. `run_area_selector.py` can be used to run the area selector on whole slide images using a patch based approach, while `run_individual_algs.py` can be used to run the individual algorithms on a set of points.

The `example_data` folder contains an example of whole slide image mitotic figure detections and an example of a set of points that can be used to test the algorithms.

The `plots` folder contains plots of the results of the algorithms on the example data.

