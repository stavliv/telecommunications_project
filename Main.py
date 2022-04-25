from HexGenerator import HexGridGenerator
from matplotlib import pyplot as plt


m = 512  # number of points in constellation
d_min = 0.00002

gen = HexGridGenerator(d_min)
points = gen.generate(m)

for point in points:
    plt.scatter(point.x, point.y, color="black", s=1)

plt.axis('scaled')
plt.show()
