import math

class Point(object):

    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def distance(self, other):
        return math.sqrt(math.pow(self.x-other.x, 2) + math.pow(self.y-other.y, 2))

    def equal(self, other):
        if self.x == other.x and self.y == other.y and self.t == other.t:
            return True
        
