from details.HexGenerator import HexGridGenerator
from details.RegHQAMGenerator import RegHQAMGenerator
from scipy.spatial import Voronoi
from Polygon import Polygon
from matplotlib import pyplot as plt


###################
# misc. functions #
###################


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


def get_max_coord_of_furthest_point(points):
    return max([max(point.x, point.y) for point in points])


##################
# implementation #
##################

m = 128  # number of points in constellation
d_min = 0.7


plt1 = plt.subplot(121)
plt2 = plt.subplot(122)
plt.tight_layout()

gen1 = RegHQAMGenerator(d_min)
points1 = gen1.generate(m)
voronoi_polygons1 = get_voronoi_polygons(points1)

for polygon in voronoi_polygons1:
    draw_polygon(polygon, plt1)
for point in points1:
    draw_point(point, plt1)


gen2 = HexGridGenerator(d_min)
points2 = gen2.generate(m)
voronoi_polygons2 = get_voronoi_polygons(points2)

for polygon in voronoi_polygons2:
    draw_polygon(polygon, plt2)
for point in points2:
    draw_point(point, plt2)

max_dist = 1.1 * max(get_max_coord_of_furthest_point(points1),
                     get_max_coord_of_furthest_point(points2))

plt1.axis('scaled')
plt2.axis('scaled')

plt1.set_xlim([-max_dist, max_dist])
plt2.set_xlim([-max_dist, max_dist])
plt1.set_ylim([-max_dist, max_dist])
plt2.set_ylim([-max_dist, max_dist])
plt1.set_title("Regular HQAM")
plt2.set_title("Custom HQAM")

plt.savefig("Constellations_M=" + str(m) + ".png", dpi=300)
plt.show()
