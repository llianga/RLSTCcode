import math
import numpy as np
class Point_xy(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_point(self):
        return self.x, self.y

    def __add__(self, other):
        if not isinstance(other, Point_xy):
            raise TypeError("The other type is not 'Point' type.")
        _add_x = self.x + other.x
        _add_y = self.y + other.y
        return Point_xy(_add_x, _add_y)

    def __sub__(self, other):
        if not isinstance(other, Point_xy):
            raise TypeError("The other type is not 'Point' type.")
        _sub_x = self.x - other.x
        _sub_y = self.y - other.y
        return Point_xy(_sub_x, _sub_y)

    def __mul__(self, x):
        if isinstance(x, float):
            return Point_xy(self.x * x, self.y * x)
        else:
            raise TypeError("The other object must 'float' type.")

    def __truediv__(self, x):
        if isinstance(x, float):
            return Point_xy(self.x / x, self.y / x)
        else:
            raise TypeError("The other object must 'float' type.")

    def distance(self, other):
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def as_array(self):
        return np.array((self.x, self.y))

def _point2line_distance(point, start, end):
    if np.all(np.equal(start, end)): 
        return np.linalg.norm(point - start) 
    return np.divide(np.abs(np.linalg.norm(np.cross(end - start, start - point))),
                     np.linalg.norm(end - start)) 