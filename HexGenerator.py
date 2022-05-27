from math import sqrt
from Point import Point


class HexGridGenerator:

    def __init__(self, d_min, offset=Point(0.0, 0.0)):
        self.d_min = d_min
        self.inner_points = set()
        # set of all inner points (with no available neighbours)
        self.outer_points = set()
        # set of all outer points (with available neighbours)

        self.available_neighbours = dict()

        self.v0 = Point(self.d_min/2.0, 0)
        self.v1 = Point(0, self.d_min * sqrt(3)/2.0)
        # keep xmin/max and ymin/max for bounding box

        # key: any point
        # value:set of available neighbours

        point0 = Point(d_min/2, 0) + offset
        self.outer_points.add(point0)
        self.available_neighbours[point0] = self.calculateNeighbours(point0)

    @staticmethod
    def findClosestPoint(points: set):

        min_point = Point(float("inf"), float("inf"))
        for p in points:
            if (p.dist_from_origin < min_point.dist_from_origin):
                min_point = p

        return min_point

    def calculateNeighbours(self, point):
        v0 = self.v0
        v1 = self.v1
        return {point + 2*v0, point - 2*v0,
                point + v0 + v1, point - v0 - v1,
                point + v0 - v1, point - v0 + v1}

    def generate(self, n_points: int):

        for _ in range(n_points):
            closest_point = self.findClosestPoint(self.outer_points)

            closest_point_neighbours = self.available_neighbours[closest_point]
            point_to_add = self.findClosestPoint(closest_point_neighbours)

            # add point
            self.outer_points.add(point_to_add)

            self.available_neighbours[point_to_add] = self.calculateNeighbours(
                point_to_add)

            # remove availability of added point
            for neighbour in self.available_neighbours[point_to_add].copy():
                if (neighbour in self.outer_points):
                    self.available_neighbours[neighbour].remove(point_to_add)
                    self.available_neighbours[point_to_add].remove(neighbour)

                    if len(self.available_neighbours[neighbour]) == 0:
                        self.available_neighbours.pop(neighbour)
                        self.outer_points.remove(neighbour)
                        self.inner_points.add(neighbour)

        return self.outer_points.union(self.inner_points)
    
    def compute_av_energy(self, points):
        res = 0
        for point in points:
            res += point.dist_from_origin ** 2
        res = float(res) / len(points)
        return res

    def plot(self, points):
        for point in points:
            plt.scatter(point.x, point.y, color="black", s=1)

        plt.axis('scaled')
        #plt.show() 
