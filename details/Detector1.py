from details.Point import Point


class Detector1:
    def __init__(self, d_min, points):
        self.d_min = d_min
        self.sqrt3 = 1.73205080757

        self.hex_to_sym = dict()

        for point in points:
            point = Point(point.x - d_min/100000, point.y - d_min/100000)
            k, l = self._point_to_hex_coordinates(point, d_min)
            symbol = "symbol " + str(k) + " " + str(l)
            self.hex_to_sym[(k, l)] = symbol

        self.l_bounds = dict()
        min_k = min([key[0] for key in self.hex_to_sym.keys()])
        max_k = max([key[0] for key in self.hex_to_sym.keys()])
        self.k_bounds = (min_k, max_k)

        for key in self.hex_to_sym.keys():
            k, l = key[0], key[1]
            self.l_bounds[k] = (l, l)
        for key in self.hex_to_sym.keys():
            k, l = key[0], key[1]
            l_min_old, l_max_old = self.l_bounds[k]
            self.l_bounds[k] = (min(l_min_old, l), max(l_max_old, l))

    @staticmethod
    def _point_to_grid_coordinates(point: Point, d_min):
        sqrt3 = 1.73205080757
        b = d_min/2
        a = 2*b/sqrt3
        offset1 = 3*a/2
        offset2 = a/2
        k0, k1, k2 = b, a, a
        grid_0_index = point.x // k0
        grid_1_index = (point.y - point.x/sqrt3 + offset1) // k1
        grid_2_index = (point.y + point.x/sqrt3 + offset2) // k2

        return [grid_0_index, grid_1_index, grid_2_index]

    @staticmethod
    def _point_to_hex_coordinates(point: Point, d_min):
        c1, c2, c3 = Detector1._point_to_grid_coordinates(point, d_min)

        mat = [(c1, c2, c3),
               (c1-1, c2, c3),
               (c1, c2-1, c3),
               (c1, c2-1, c3-1),
               (c1-1, c2-1, c3-1),
               (c1-1, c2, c3-1)]

        for row in mat:
            c1, c2, c3 = row[0], row[1], row[2]
            k = -(c2-c1)//3
            l = c1 - 2*k
            condition1 = (c1 == 2*k+l)
            condition2 = (c2 == -k+l)
            condition3 = (c3 == k+2*l)
            is_valid_coordinate = (condition1 and condition2 and condition3)
            if is_valid_coordinate:
                return k, l
        # Below is a fallback case, might occur when the received point is
        # exactly a point of a constellation
        c1, c2, c3 = mat[0][0], mat[0][1], mat[0][2]
        k = -(c2-c1)//3
        l = c1 - 2*k
        return k, l

    def detect(self, point: Point):
        k, l = self._point_to_hex_coordinates(point, self.d_min)
        # if l < self.k_bounds[k][0] or l > self.k_bounds[k][1]:
        if (k, l) not in self.hex_to_sym:
            if k < self.k_bounds[0]:
                symbol = "ERR"
            elif k > self.k_bounds[1]:
                symbol = "ERR"
            elif l < self.l_bounds[k][0]:
                symbol = "ERR"
            elif l > self.l_bounds[k][1]:
                symbol = "ERR"
        else:
            symbol = self.hex_to_sym[(k, l)]

        return (symbol, k, l)
