import random
from Point import Point
from HexGenerator import HexGridGenerator
from Detector import MLD, ThrassosDetector
from Detector1 import Detector1
from Detector2 import Detector2
from Detector3 import Detector3
from time import time

from Polygon import Polygon
from scipy.spatial import Voronoi
from math import sqrt

from matplotlib.colors import rgb2hex
from matplotlib import pyplot as plt

##################################
# misc. functions related to     #
# visualization & preparing data #
##################################


def draw_point(point, plt: plt, col="black"):
    plt.scatter(point.x, point.y, color=col, s=1)


def draw_polygon(polygon: Polygon, plt: plt):
    # draw polygon
    plt.fill([polygon[0][i][0] for i in range(polygon.nPoints())],
             [polygon[0][i][1] for i in range(polygon.nPoints())])


def draw_lines_from_indices(indices, d_min, plt):
    # Draws three lines that correspond
    i0, i1, i2 = indices[0], indices[1], indices[2]
    b = d_min/2
    a = 2*b/sqrt(3)
    offset1 = 3*a/2
    offset2 = a/2
    k0, k1, k2 = b, a, a

    plt.axline((i0*k0, 0), (i0*k0, 1))
    plt.axline(((i1*k1 - offset1)*sqrt(3), 0), (0, k1*i1 - offset1))
    plt.axline(((k2*i2 - offset2)*sqrt(3), 0),
               (0, (k2*i2 - offset2)))


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

    def add_gaussian_noise(self, point, n0):
        return Point(point.x + np.random.normal(loc=0.0, scale=sqrt(n0/2)), point.y + np.random.normal(loc=0.0, scale=sqrt(n0/2)))

##################
# implementation #
##################


d_min = 0.7

m_values = [16*(2**i) for i in range(1, 8)]
results = list()
for m in m_values:
    gen = HexGridGenerator(d_min)
    points = gen.generate(m)

    voronoi_polygons = get_voronoi_polygons(points)

    detector1 = Detector1(d_min, points)
    detector2 = Detector2(d_min, points, voronoi_polygons,
                          max(4, int(sqrt(m)/4)))
    thrassos_detector = ThrassosDetector(points, gen)
    mld_detector = MLD(points)

    detectors = [detector1, detector2, thrassos_detector]

    avg_times = [None for _ in detectors]

    points_list = list(points)
    test_points = [random.choice(points_list) for _ in range(100000)]

    for i, detector in enumerate(detectors):

        time_sum = 0
        for point in test_points:
            t0 = time()
            detector.detect(point)
            t1 = time()
            time_sum += (t1-t0)

        avg_times[i] = time_sum/len(test_points)

    results.append(avg_times)


plt.plot(m_values, [item[0]*1000 for item in results], label="method 1")
plt.plot(m_values, [item[1]*1000 for item in results], label="method 2")
plt.plot(m_values, [item[2]*1000 for item in results], label="thrassos")
# plt.plot(m_values, [item[3] for item in results])
plt.xlabel("Constellation size")
plt.ylabel("Average symbol detection time (ms)")
# plt.xscale("log")

plt.legend()


plt.show()
print(results)

# plt.show()
