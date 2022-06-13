from math import sqrt, isclose
from abc import ABCMeta, abstractmethod
from details.Point import Point
from matplotlib import pyplot as plt
from Polygon import Polygon
from bisect import bisect_left, bisect_right
from scipy.spatial import Voronoi


class DetectionMethod(metaclass=ABCMeta):
    def __init__(self, points, name):
        self.points = points
        self.name = name

    @abstractmethod
    def detect(self, received_point: Point) -> Point:
        pass

    @abstractmethod
    def bool_detect(self, transmitted_point: Point, received_point: Point) -> bool:
        pass

    @staticmethod
    def get_voronoi_polygons(points):
        voronoi_polygons = dict()
        ext = 10 * \
            max(points, key=lambda point: point.dist_from_origin).dist_from_origin
        voronoi = Voronoi([[point.x, point.y] for point in points] +
                          [[ext, 0], [0, -ext], [0, ext], [-ext, 0]])

        points = list(points)
        for i in range(len(points)):
            region = voronoi.regions[voronoi.point_region[i]]
            if -1 not in region and len(region) > 0:
                pol = Polygon([voronoi.vertices[j] for j in region])
                voronoi_polygons[points[i]] = pol
        return voronoi_polygons

    @staticmethod
    def draw_polygon(polygon: Polygon, plt: plt):
        plt.fill([polygon[0][i][0] for i in range(polygon.nPoints())],
                 [polygon[0][i][1] for i in range(polygon.nPoints())])

    @staticmethod
    def get_quadrants(point: Point):
        quadrants = list()
        if (point.x >= 0 or isclose(point.x, 0, abs_tol=1e-08)) and (point.y >= 0 or isclose(point.y, 0, abs_tol=1e-08)):
            quadrants.append(0)
        if (point.x <= 0 or isclose(point.x, 0, abs_tol=1e-08)) and (point.y >= 0 or isclose(point.y, 0, abs_tol=1e-08)):
            quadrants.append(1)
        if (point.x <= 0 or isclose(point.x, 0, abs_tol=1e-08)) and (point.y <= 0 or isclose(point.y, 0, abs_tol=1e-08)):
            quadrants.append(2)
        if (point.x >= 0 or isclose(point.x, 0, abs_tol=1e-08)) and (point.y <= 0 or isclose(point.y, 0, abs_tol=1e-08)):
            quadrants.append(3)
        return quadrants


class MLD(DetectionMethod):
    def __init__(self, points, *args, name="MLD"):
        super().__init__(points, name)

    def detect(self, received_point: Point) -> Point:
        min_dist = float("inf")
        for point in self.points:
            dist = (point.x - received_point.x)**2 + \
                (point.y - received_point.y)**2
            if dist < min_dist:
                min_dist = dist
                nearest_symbol = point
        return nearest_symbol

    def bool_detect(self, transmitted_point: Point, received_point: Point) -> bool:
        detected_point = self.detect(received_point)
        return bool(detected_point == transmitted_point)


class ThrassosDetector(DetectionMethod):
    def __init__(self, points, d_min, name="Thrassos' method"):
        super().__init__(points, name)
        if points:
            self.d_min = d_min
            self.Sx = self.create_Sx()
            self.A = self.create_A()
            self.Q = self.initialize_Q()

    def create_Sx(self):
        res = set()
        for point in self.points:
            res.add(point.x)
        res = list(res)
        res.sort()
        return res

    def create_A(self):
        A = dict()
        for point in self.points:
            if point.x in A.keys():
                if len(A[point.x]) < sqrt(len(self.points)):
                    A[point.x].append((point, point.y))
            else:
                A[point.x] = list()
                A[point.x].append((point, point.y))

        for Ai in A.values():
            Ai.sort(key=lambda p: p[1])
        return A

    def initialize_Q(self):
        Q = [set() for _ in range(4)]

        polygons = self.get_voronoi_polygons(self.points)

        quadrants = list()
        ext = 10*max(self.points,
                     key=lambda point: point.dist_from_origin).dist_from_origin
        quadrants.append(
            Polygon([[ext/5, 0], [0, ext/5], [0, ext], [ext, ext], [ext, 0]]))
        quadrants.append(
            Polygon([[-ext/5, 0], [0, ext/5], [0, ext], [-ext, ext], [-ext, 0]]))
        quadrants.append(
            Polygon([[-ext/5, 0], [0, -ext/5], [0, -ext], [-ext, -ext], [-ext, 0]]))
        quadrants.append(
            Polygon([[ext/5, 0], [0, -ext/5], [0, -ext], [ext, -ext], [ext, 0]]))

        for point, polygon in polygons.items():
            for j in range(len(quadrants)):
                if polygon.overlaps(quadrants[j]):
                    Q[j].add(point)

        return Q

    @staticmethod
    def binary_search(sorted_list, lower_bound, upper_bound):
        low = bisect_left(sorted_list, lower_bound)
        high = bisect_right(sorted_list, upper_bound)
        return range(low, high)

    def detect(self, received_point: Point):
        candidates = set()
        x = self.binary_search(list(self.Sx), received_point.x -
                               self.d_min, received_point.x + self.d_min)
        for i in x:
            y = self.binary_search([item[1] for item in self.A[self.Sx[i]]], received_point.y -
                                   self.d_min, received_point.y + self.d_min)
            for j in y:
                candidates.add(self.A[self.Sx[i]][j][0])
        if not candidates:
            quadrants = self.get_quadrants(received_point)
            for quadrant in quadrants:
                candidates.update(self.Q[quadrant])
        mld = MLD(candidates)
        nearest_symbol = mld.detect(received_point)
        return nearest_symbol

    def bool_detect(self, transmitted_point: Point, received_point: Point) -> bool:
        detected_point = self.detect(received_point)
        return bool(detected_point == transmitted_point)
