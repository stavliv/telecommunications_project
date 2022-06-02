from Point import Point
from math import sqrt
from Polygon import Polygon


class Detector2:

    @staticmethod
    def get_bounds(points):
        x_bounds = [float('inf'), float('-inf')]
        y_bounds = [float('inf'), float('-inf')]

        for point in points:
            x_bounds = [min(point.x, x_bounds[0]), max(point.x, x_bounds[1])]
            y_bounds = [min(point.y, y_bounds[0]), max(point.y, y_bounds[1])]

        return x_bounds, y_bounds

    def __init__(self, d_min, points, voronoi_polygons, subdiv=8, padding=0):
        self.points = list(points)

        voronoi_to_point_index = Detector2._get_voronoi_to_point_index_map(
            voronoi_polygons, self.points)

        x_bounds, y_bounds = Detector2.get_bounds(self.points)

        self.x_subdiv = subdiv
        x_padding = padding

        self.x_start = x_bounds[0] - d_min/2 - x_padding
        self.x_end = x_bounds[1] + d_min/2 + x_padding
        self.x_step = (self.x_end - self.x_start)/self.x_subdiv

        self.y_subdiv = subdiv
        y_padding = padding

        self.y_start = y_bounds[0] - d_min/sqrt(3) - y_padding
        self.y_end = y_bounds[1] + d_min/sqrt(3) + y_padding
        self.y_step = (self.y_end - self.y_start)/self.y_subdiv

        self.rect_inidices_to_point_indices = [
            [[] for x in range(self.x_subdiv)] for y in range(self.y_subdiv)]

        for y_i in range(self.y_subdiv):
            for x_i in range(self.x_subdiv):
                x1 = self.x_start + x_i * self.x_step
                x2 = x1 + self.x_step
                y1 = self.y_start + y_i * self.y_step
                y2 = y1 + self.y_step

                rect = Polygon([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])

                for polygon in voronoi_polygons:
                    if(rect.overlaps(polygon)):
                        point_index = voronoi_to_point_index[polygon]
                        self.rect_inidices_to_point_indices[x_i][y_i].append(
                            point_index)

    @staticmethod
    def _get_voronoi_to_point_index_map(voronoi_polygons, points):
        polygon_point_index_map = dict()
        avail_polygons = voronoi_polygons.copy()
        for id, point in enumerate(points):
            for polygon in avail_polygons:
                if polygon.isInside(point.x, point.y):
                    polygon_point_index_map[polygon] = id
                    break
            avail_polygons.remove(polygon)
        return polygon_point_index_map

    @staticmethod
    def sq_distance(p1: Point, p2: Point):
        return (p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y)

    def detect(self, received_point: Point):
        x_i = int((received_point.x - self.x_start)//self.x_step)
        y_i = int((received_point.y - self.y_start)//self.y_step)
        if x_i < 0 or x_i >= self.x_subdiv or y_i < 0 or y_i >= self.y_subdiv:
            # print("ERROR: Out of Bounds")
            return -1

        min_distance = float('inf')
        min_distance_point_id = -1

        for point_id in self.rect_inidices_to_point_indices[x_i][y_i]:
            point = self.points[point_id]
            d = Detector2.sq_distance(point, received_point)
            if d < min_distance:
                min_distance = d
                min_distance_point_id = point_id

        return min_distance_point_id
