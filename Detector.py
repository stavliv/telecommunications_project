from cmath import inf
import numpy as np
from math import sqrt, isclose
from abc import ABCMeta, abstractmethod
from Point import Point
from HexGenerator import HexGridGenerator
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from Polygon import Polygon
from bisect import bisect_left, bisect_right
from time import time

class DetectionMethod(metaclass=ABCMeta):
    def __init__(self, points, name):
        self.points = points
        self.name = name
    
    @abstractmethod
    def detect(self, received_point: tuple, *args) -> Point:
        pass

    @abstractmethod
    def bool_detect(self, transmitted_point : Point, received_point: tuple, *args) -> bool:
        pass
    
    @staticmethod
    def get_voronoi_polygons(points):
        voronoi_polygons = dict()
        ext = 10*max(points, key=lambda point: point.dist_from_origin).dist_from_origin
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
    def get_quadrants(point: tuple):
        quadrants = list()
        if (point[0] >= 0 or isclose(point[0], 0, abs_tol=1e-08)) and (point[1] >= 0 or isclose(point[1], 0, abs_tol=1e-08)):
            quadrants.append(0)
        if (point[0] <= 0 or isclose(point[0], 0, abs_tol=1e-08)) and (point[1] >= 0 or isclose(point[1], 0, abs_tol=1e-08)):
            quadrants.append(1)
        if (point[0] <= 0 or isclose(point[0], 0, abs_tol=1e-08)) and (point[1] <= 0 or isclose(point[1], 0, abs_tol=1e-08)):
            quadrants.append(2)
        if (point[0] >= 0 or isclose(point[0], 0, abs_tol=1e-08)) and (point[1] <= 0 or isclose(point[1], 0, abs_tol=1e-08)):
            quadrants.append(3)
        return quadrants


class MLD(DetectionMethod):
    def __init__(self, points, *args, name="MLD"):
        super().__init__(points, name)
    
    def detect(self, received_point: tuple) -> Point:
        min_dist = float("inf")
        for point in self.points:
            dist = sqrt((point.x - received_point[0])**2 + (point.y - received_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_symbol = point
        return nearest_symbol

    def bool_detect(self, transmitted_point : Point, received_point: tuple, hexGridGenerator) -> bool:
        # mld regions for inner points are already known (hexagonals)
        if transmitted_point in hexGridGenerator.inner_points:
            if abs(received_point[0] - transmitted_point.x) <= hexGridGenerator.d_min/2 and abs(received_point[1] - transmitted_point.y) <= ((hexGridGenerator.d_min - abs(received_point[0] - transmitted_point.x) / sqrt(3))):
                return True
            return False
        # for outer points we just look if there is a symbol with smaller distance from the received point than the transmitted point
        elif transmitted_point in hexGridGenerator.outer_points:
            dist = sqrt((transmitted_point.x - received_point[0]) ** 2 + (transmitted_point.y - received_point[1]) ** 2)
            for point in reversed(list(self.points)): #traverse points in reverse order - more likely to mistake an outer symbol for another outer(ish) symbol
                cur_dist = sqrt((point.x - received_point[0]) ** 2 + (point.y - received_point[1]) ** 2)
                if cur_dist < dist:
                    return False
            return True


class ThrassosDetector(DetectionMethod):
    def __init__(self, points, hexGridGenerator, name="Thrassos' method"):
        super().__init__(points, name)
        if points:
            self.hexGridGenerator = hexGridGenerator
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
        ext = 10*max(self.points, key=lambda point: point.dist_from_origin).dist_from_origin
        quadrants.append(Polygon([[ext/5, 0], [0, ext/5], [0, ext], [ext, ext], [ext, 0]]))
        quadrants.append(Polygon([[-ext/5, 0], [0, ext/5], [0, ext], [-ext, ext], [-ext, 0]]))
        quadrants.append(Polygon([[-ext/5, 0], [0, -ext/5], [0, -ext], [-ext, -ext], [-ext, 0]]))
        quadrants.append(Polygon([[ext/5, 0], [0, -ext/5], [0, -ext], [ext, -ext], [ext, 0]]))
        
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

    def detect(self, received_point: tuple):
        candidates = set()
        x = self.binary_search(list(self.Sx), received_point[0] - self.hexGridGenerator.d_min, received_point[0] + self.hexGridGenerator.d_min)
        for i in x:
            y = self.binary_search([item[1] for item in self.A[self.Sx[i]]], received_point[1] - self.hexGridGenerator.d_min, received_point[1] + self.hexGridGenerator.d_min)
            for j in y:
                candidates.add(self.A[self.Sx[i]][j][0])
        if not candidates:
            quadrants = self.get_quadrants(received_point)
            for quadrant in quadrants:
                candidates.update(self.Q[quadrant])
        mld = MLD(candidates)
        nearest_symbol = mld.detect(received_point)
        return nearest_symbol  

    def bool_detect(self, transmitted_point: Point, received_point: tuple, *args) -> bool:
        detected_point = self.detect(received_point)
        return bool(detected_point == transmitted_point)



if __name__ == '__main__':
    gen = HexGridGenerator(1)
    points = gen.generate(31)
    #gen.plot(points)

    ext = 10*max(points, key=lambda point: point.dist_from_origin).dist_from_origin
    vor = Voronoi([[point.x, point.y] for point in points] 
                        )
    voronoi_plot_2d(vor)
    plt.savefig("plot.png")
    plt.show()

    detection = ThrassosDetector(points, gen)
    received_point = (3, 0)
    detected_point = detection.detect(received_point)
    print(f"received point: {received_point[0], received_point[1]}, detected point: {(detected_point.x, detected_point.y)}")


    """
    debug infinite regions
    for i in range(4):
        print(f"points with infinite area in quadrant {i + 1}:\n{[(point.x, point.y) for point in detection.Q[i]]}")
    
    plt.savefig("plot.png")
    plt.show()
    """

    
    