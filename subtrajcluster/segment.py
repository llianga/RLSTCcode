import math
from point_xy import _point2line_distance

class Segment(object):
    eps = 1e-12
    def __init__(self, start_point, end_point,traj_id=None):
        self.start = start_point
        self.end = end_point
        self.traj_id = traj_id

    @property
    def length(self):
        return self.end.distance(self.start)

    def perpendicular_distance(self, other):
        l1 = other.start.distance(self._projection_point(other, typed="start"))
        l2 = other.end.distance(self._projection_point(other, typed="end"))
        if l1 < self.eps and l2 < self.eps:
            return 0
        else:
            return (math.pow(l1, 2) + math.pow(l2, 2)) / (l1 + l2)

    def parallel_distance(self, other):
        l1 = min(self.start.distance(self._projection_point(other, typed='start')),self.end.distance(self._projection_point(other, typed='start')))
        l2 = min(self.end.distance(self._projection_point(other, typed='end')),self.start.distance(self._projection_point(other, typed='end')))
        return min(l1, l2)

    def angle_distance(self, other):
        self_vector = self.end - self.start
        other_vector = other.end - other.start
        self_dist, other_dist = self.end.distance(self.start), other.end.distance(other.start)

        if self_dist == 0:
            return _point2line_distance(self.start.as_array(), other.start.as_array(), other.end.as_array())
        elif other_dist == 0:
            return _point2line_distance(other.start.as_array(), self.start.as_array(), self.end.as_array())

        if self.start == other.start and self.end == other.end:
            return 0
        cos_theta = self_vector.dot(other_vector) / (self_dist * other_dist)
        if cos_theta > self.eps:
            if cos_theta >= 1:
                cos_theta = 1.0
            return other.length * math.sqrt(1 - math.pow(cos_theta, 2))
        else:
            return other.length

    def _projection_point(self, other, typed="e"):
        if typed == 's' or typed == 'start':
            tmp = other.start - self.start
        else:
            tmp = other.end - self.start
        if math.pow(self.end.distance(self.start), 2) == 0: #start=end
            return self.start
        u = tmp.dot(self.end-self.start) / math.pow(self.end.distance(self.start), 2)
        return self.start + (self.end-self.start) * u

    def get_all_distance(self, seg):  
        res = self.angle_distance(seg) + self.parallel_distance(seg) + self.perpendicular_distance(seg)
        return res

def compare(segment_a, segment_b):
    return (segment_a, segment_b) if segment_a.length > segment_b.length else (segment_b, segment_a)

