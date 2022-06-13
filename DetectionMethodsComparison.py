import random

import numpy as np
from details.Point import Point
from details.HexGenerator import HexGridGenerator
from details.Detector import MLD, ThrassosDetector
from details.Detector1 import Detector1
from details.Detector2 import Detector2
from details.Detector3 import Detector3
from time import time_ns

from Polygon import Polygon
from scipy.spatial import Voronoi
from math import sqrt
from matplotlib import pyplot as plt

##################################
# misc. functions related to     #
# visualization & preparing data #
##################################


def draw_point(point, plt: plt, col="black"):
    plt.scatter(point.x, point.y, color=col, s=0.1)


def draw_polygon(polygon: Polygon, plt: plt):
    # draw polygon
    plt.fill([polygon[0][i][0] for i in range(polygon.nPoints())],
             [polygon[0][i][1] for i in range(polygon.nPoints())])


def draw_polygon_outline(polygon: Polygon, plt: plt):
    coordinates = polygon[0]
    coordinates.append(coordinates[0])
    xs, ys = zip(*coordinates)
    plt.plot(xs, ys, color="black", linewidth=0.5)


def get_voronoi_polygons(points):
    voronoi_polygons = list()

    ext = 2*max(points).dist_from_origin

    voronoi = Voronoi([[point.x, point.y] for point in points] +
                      [[ext, 0], [0, -ext], [0, ext], [-ext, 0]])

    for region in voronoi.regions:
        if -1 not in region and len(region) > 0:
            pol = Polygon([voronoi.vertices[i] for i in region])
            voronoi_polygons.append(pol)

    return voronoi_polygons


def add_gaussian_noise(point, n0):
    return Point(point.x + np.random.normal(loc=0.0, scale=sqrt(n0/2)),
                 point.y + np.random.normal(loc=0.0, scale=sqrt(n0/2)))


##################
# implementation #
##################


d_min = 1
SNR = 20
n_test_points = 1000


m_values = [16*(2**i) for i in range(2, 9)]
results = list()
for m in m_values:
    gen = HexGridGenerator(d_min)
    points = gen.generate(m)

    voronoi_polygons = get_voronoi_polygons(points)

    detector1 = Detector1(d_min, points)
    detector2 = Detector2(d_min, points, voronoi_polygons, 64, 6*d_min)
    detector3 = Detector3(points, d_min)
    thrassos_detector = ThrassosDetector(points, d_min)
    mld_detector = MLD(points)

    detectors = [detector1, detector2, detector3,
                 thrassos_detector, mld_detector]

    points_list = list(points)

    n0 = gen.compute_av_energy(points) / (10**(SNR / 10))
    test_points = [add_gaussian_noise(random.choice(
        points_list), n0) for _ in range(n_test_points)]

    time_avg = [None for _ in detectors]

    for i, detector in enumerate(detectors):
        if detector == mld_detector and len(points) > 512:
            time_avg[i] = 10**9
            # skips MLD for very large constellations
            continue

        time_sum = 0
        t0 = time_ns()
        for point in test_points:
            s = detector.detect(point)
            if (detector == detector1) and s[0] == "ERR":
                detector2.detect(point)
                t1 = time_ns()
        t1 = time_ns()
        if t1-t0 == 0:
            print("Time error")
        dt = t1-t0
        time_avg[i] = (t1-t0)/len(test_points)

    results.append(time_avg)


plt.plot(
    m_values, [item[0]/1000 for item in results], label="method 1")
plt.plot(
    m_values, [item[1]/1000 for item in results], label="method 2")
plt.plot(
    m_values, [item[2]/1000 for item in results], label="method 3")
plt.plot(
    m_values, [item[3]/1000 for item in results], label="thrassos")
plt.plot(
    m_values, [item[4]/1000 for item in results], label="MLD")


plt.title("Method Comparison, SNR = " + str(SNR))
plt.xlabel("Constellation size")
plt.ylabel("Average symbol detection time (μs)")
plt.ylim([0, 100])
# plt.xscale("log")
plt.legend()
plt.savefig("Method_Comparison_SNR_" + str(SNR) + ".png", dpi=300)
plt.show()
