from Point import Point
from math import floor, sqrt
import numpy as np
from Detector import MLD

class Detector3:
    def __init__(self, points, d_min):
        self.d_min = d_min       
        self.points = points

        self.x_div = self.d_min / 2.0
        self.y_div = self.d_min * sqrt(3) / 2.0

        bounds = self.get_h_w_bounds()
        self.h_bounds = bounds[0]
        self.w_bounds = bounds[1]
        self.x_bounds = bounds[2]
   
    def get_h_w_bounds(self):
        h_bounds = {-1 : float('-inf'), 1 : float('-inf')}
        x_bounds = {-1 : dict(), 1 : dict()}
        w_bounds = {-1 : dict(), 1 : dict()}

        for point in self.points:
            h = floor(abs(point.y) / self.y_div)

            sign_x = np.sign(point.x) if np.sign(point.x) != 0 else 1.0
            sign_y = np.sign(point.y) if np.sign(point.y) != 0 else 1.0
            
            if h > h_bounds[sign_y] :
                h_bounds[sign_y] = h

            if h not in x_bounds[sign_y].keys():
                x_bounds[sign_y][h] = {-1 : float('-inf'), 1 : float('-inf')}

            if abs(point.x) > x_bounds[sign_y][h][sign_x]:
                x_bounds[sign_y][h][sign_x] = abs(point.x)

        h_bounds[-1] -= 1
        h_bounds[1] -= 1

        x_bounds[-1][0] = x_bounds[1][0]

        for signy in x_bounds.keys():
            for height in range(h_bounds[signy] + 1):
                w_bounds[signy][height] = dict()
                for signx in x_bounds[signy][height].keys():
                    w_bounds[signy][height][signx] = min((x_bounds[signy][height][signx] / self.x_div), (x_bounds[signy][height + 1][signx] / self.x_div))                   
        return (h_bounds, w_bounds, x_bounds)   

    def detect(self, point: Point):
        w = floor(abs(point.x) / self.x_div)
        h = floor(abs(point.y) / self.y_div)

        sign_x = np.sign(point.x) if np.sign(point.x) != 0 else 1.0
        sign_y = np.sign(point.y) if np.sign(point.y) != 0 else 1.0
      
        if h <= self.h_bounds[sign_y] and w <= self.w_bounds[sign_y][h][sign_x]:
            if abs(w - h) % 2 == 0:
                cand_1 = Point(sign_x * (w + 1) * self.x_div, sign_y * h * self.y_div)
                cand_2 = Point(sign_x * w * self.x_div, sign_y * (h + 1) * self.y_div)
            else:
                cand_1 = Point(sign_x * w * self.x_div, sign_y * h * self.y_div)
                cand_2 = Point(sign_x * (w + 1)* self.x_div, sign_y * (h + 1) * self.y_div)
            mld = MLD((cand_1, cand_2))
            nearest_symbol = mld.detect(point)
        elif (h == self.h_bounds[sign_y] + 1 and w <= (self.x_bounds[sign_y][h][sign_x] / self.x_div)) or \
         (h <= self.h_bounds[sign_y] and w <= max((self.x_bounds[sign_y][h][sign_x] / self.x_div), (self.x_bounds[sign_y][h + 1][sign_x] / self.x_div))):
            if abs(w - h) % 2 == 0:
                cand_1 = Point(sign_x * (w + 1) * self.x_div, sign_y * h * self.y_div)
                cand_2 = Point(sign_x * w * self.x_div, sign_y * (h + 1) * self.y_div)
            else:
                cand_1 = Point(sign_x * w * self.x_div, sign_y * h * self.y_div)
                cand_2 = Point(sign_x * (w + 1)* self.x_div, sign_y * (h + 1) * self.y_div)
            mld = MLD((cand_1, cand_2))
            nearest_symbol = mld.detect(point)
            if nearest_symbol not in self.points:
                nearest_symbol = Point(100000, 1000)
        else:
            nearest_symbol = Point(100000, 1000)      
        return nearest_symbol
