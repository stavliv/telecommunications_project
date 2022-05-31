from math import floor, log2, sqrt
from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH

from matplotlib import pyplot as plt
from Point import Point


class RegHQAMGenerator:

    def __init__(self, d_min):
        self.d_min = d_min
        self.points = set()

        self.available_neighbours = dict()

        self.v0 = Point(self.d_min/2.0, 0)
        self.v1 = Point(0, self.d_min * sqrt(3)/2.0)
        # keep xmin/max and ymin/max for bounding box

        # key: any point
        # value:set of available neighbours

    def generate(self, n_points: int):
        b = self.d_min/2
        a = self.d_min/sqrt(3)

        x_step = 2*b
        y_step = a + b/2
        offset = b

        point0 = Point(b/2, y_step/2)
        """
        self.points.add(point0)
        self.points.add(point0 - Point(-x_step, 0))
        self.points.add(point0 - Point(-x_step - offset, -y_step))
        self.points.add(point0 - Point(- offset, -y_step)) 
        """

        exponent = floor(log2(n_points))
        even_exponent = exponent - (exponent % 2)
        regular_reps = int(sqrt(2**even_exponent))

        for x in range(-regular_reps//2, regular_reps//2):
            for y in range(-regular_reps//2, regular_reps//2):
                point = point0 + Point(x*x_step + (y % 2)*offset, y*y_step)
                self.points.add(point)

        if exponent % 2 == 1:
            r = regular_reps//4
            # r -> how many rows to add at each side of the square
            #  to make cross-shape
            range1 = range(regular_reps//2, regular_reps//2 + r)
            range2 = range(-regular_reps//2 - r, regular_reps//2 - 1)
            for x in set(range1) | set(range2):
                for y in range(-regular_reps//2, regular_reps//2):
                    point = point0 + Point(x*x_step + (y % 2)*offset, y*y_step)
                    self.points.add(point)
            for y in set(range1) | set(range2):
                for x in range(-regular_reps//2, regular_reps//2):
                    point = point0 + Point(x*x_step + (y % 2)*offset, y*y_step)
                    self.points.add(point)

        return self.points

    def compute_av_energy(self, points):
        res = 0
        for point in points:
            res += point.dist_from_origin ** 2
        res = float(res) / len(points)
        return res


gen = RegHQAMGenerator(0.4)
points = gen.generate(1028)

for point in points:
    plt.scatter(point.x, point.y, color="black", s=0.1)

plt.axis('scaled')
plt.show()
