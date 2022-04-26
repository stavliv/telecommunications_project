from math import sqrt, isclose,  floor, log10


class Point:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
        self.dist_from_origin = sqrt(x**2 + y**2).real
        self._hash = None

    def __eq__(self, other: object) -> bool:
        return (isclose(self.x, other.x) and isclose(self.y, other.y))

    def __hash__(self):
        if(self._hash is None):
            self._hash = self._gen_hash()
        return self._hash

    def _gen_hash(self):

        sig = 8

        if (self.x != 0.0):
            x_scale = int(floor(log10(abs(self.x))))
        else:
            x_scale = 0

        if (self.y != 0.0):
            y_scale = int(floor(log10(abs(self.y))))
        else:
            y_scale = 0

        # How much to scale x and y to get 'sig' non-decimal digits

        x_rounded = round(self.x, sig - x_scale - 1)
        y_rounded = round(self.y, sig - y_scale - 1)

        # round x and y to 8 significant digits
        # (eg 1234.56789 will be rounded to 1234.5678
        # 12345678942 will be rounded to 12345679000 etx)
        return hash((x_rounded, y_rounded))

    def __add__(self, other: object):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: object):
        return self + (-other)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __mul__(self, a: int):
        return Point(a*self.x, a*self.y)

    def __rmul__(self, a: int):
        return self.__mul__(a)


Point(123456789, 0.123456789).__hash__()
