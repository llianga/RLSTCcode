import math
from segment import Segment, compare
from point import Point
from traj import Traj
import numpy as np
from point_xy import Point_xy

eps = 1e-12   # defined the segment length theta, if length < eps then l_h=0

def makemid(x1,t1,x2,t2,t):
    return x1+(t-t1)/(t2-t1)*(x2-x1)


def traj_mdl_comp(points, start_index, curr_index, typed):
    seg = Segment(points[start_index], points[curr_index])
    h = 0
    lh = 0
    if typed == 'simp':
        if seg.length > eps:
            h = 0.5*math.log2(seg.length) + 0.5*(abs(points[start_index].t - points[curr_index].t))
    t1, t2, x1, x2, y1, y2 = points[start_index].t, points[curr_index].t, points[start_index].x, points[curr_index].x, points[start_index].y, points[curr_index].y
    
    for i in range(start_index, curr_index, 1):
        if typed == 'simp':
            t = points[i].t
            new_x = makemid(x1, t1, x2, t2, t)
            new_y = makemid(y1, t1, y2, t2, t)
            new_p = Point(new_x, new_y, t)
            lh += points[i].distance(new_p)
        elif typed == 'orign':
            d = 0.5*(points[i].distance(points[i+1]))+0.5*(abs(points[i].t-points[i+1].t))
            if d > eps:
                h += math.log2(d)
    if typed == 'simp':
        if lh > eps:
            h += math.log2(lh)
        return h
    else:
        return h


def timedTraj(points, ts, te): #get trajectories whose time intervals between ts and te
    if ts == te:
        return
    if ts >points[-1].t or te < points[0].t:
        return
    s_i = 0
    e_i = len(points)-1
    new_points = []
    while points[s_i].t < ts:
        s_i += 1
    while points[e_i].t > te:
        e_i -= 1
    if s_i != 0 and points[s_i].t != ts:
        x = makemid(points[s_i - 1].x, points[s_i - 1].t, points[s_i].x, points[s_i].t, ts)
        y = makemid(points[s_i - 1].y, points[s_i - 1].t, points[s_i].y, points[s_i].t, ts)
        new_p = Point(x,y,ts)
        new_points.append(new_p)
    for i in range(s_i, e_i+1):
        new_points.append(points[i])
    if e_i != len(points)-1 and points[e_i].t != te:
        x = makemid(points[e_i].x, points[e_i].t, points[e_i+1].x, points[e_i+1].t, te)
        y = makemid(points[e_i].y, points[e_i].t, points[e_i+1].y, points[e_i+1].t, te)
        new_p = Point(x, y, te)
        new_points.append(new_p)
    new_ts = new_points[0].t
    new_te = new_points[-1].t
    new_size = len(new_points)
    new_traj = Traj(new_points,new_size,new_ts, new_te)
    return new_traj
 
def line2lineIDE(p1s,p1e,p2s,p2e):
    d1 = p1s.distance(p2s)
    d2 = p1e.distance(p2e)
    d = 0.5*(d1 + d2)*(p1e.t - p1s.t)
    return d 

def getstaticIED(points, x, y, t1, t2): #t1<t2
    s_t = max(points[0].t, t1)
    e_t = min(points[-1].t, t2)
    sum = 0
    if s_t >= e_t:
        return 1e10
    ps, pe = Point(x, y, 0), Point(x, y, 0)
    timedpoints = timedTraj(points, s_t, e_t)
    for i in range(timedpoints.size - 1):
        ps.t = timedpoints.points[i].t
        pe.t = timedpoints.points[i+1].t
        pd = line2lineIDE(timedpoints.points[i], timedpoints.points[i+1], ps, pe)
        sum += pd
    return sum
 
def traj2trajIED(traj_points1, traj_points2):
    t1s, t1e, t2s, t2e = traj_points1[0].t, traj_points1[-1].t, traj_points2[0].t, traj_points2[-1].t
    if t1s >= t2e or t1e <= t2s:
        return 1e10
    sum = 0
    timedtraj = timedTraj(traj_points2, t1s, t1e) 
    cut1 = timedtraj.ts
    cut2 = timedtraj.te
    
    commontraj = timedTraj(traj_points1, cut1, cut2) 
    if t1s < cut1:
        pd = getstaticIED(traj_points1, timedtraj.points[0].x, timedtraj.points[0].y, t1s, cut1)
        sum += pd
    if t2s < t1s:
        pd = getstaticIED(traj_points2, traj_points1[0].x, traj_points1[0].y, t2s, t1s)
        sum += pd
    if t1e > cut2:
        pd = getstaticIED(traj_points1, timedtraj.points[-1].x, timedtraj.points[-1].y, cut2, t1e)
        sum += pd
    if t1e < t2e:
        pd = getstaticIED(traj_points2, traj_points1[-1].x, traj_points1[-1].y, t1e, t2e)
        sum += pd
    
    if commontraj is not None and commontraj.size != 0:
        newtime, lasttime = commontraj.ts, commontraj.ts
        iter1, iter2 = 0, 0
        lastp1, lastp2 = commontraj.points[0], timedtraj.points[0]

        while lasttime != timedtraj.te:
            if timedtraj.points[iter2+1].t == commontraj.points[iter1+1].t:
                newtime = timedtraj.points[iter2+1].t
                newp1 = commontraj.points[iter1+1]
                newp2 = timedtraj.points[iter2+1]
                iter1 += 1
                iter2 += 1
            elif timedtraj.points[iter2+1].t < commontraj.points[iter1+1].t:
                t = timedtraj.points[iter2+1].t
                x = makemid(commontraj.points[iter1].x, commontraj.points[iter1].t, commontraj.points[iter1+1].x, commontraj.points[iter1+1].t, t)
                y = makemid(commontraj.points[iter1].y, commontraj.points[iter1].t, commontraj.points[iter1+1].y, commontraj.points[iter1+1].t, t)
                newp1 = Point(x, y, t)
                newp2 = timedtraj.points[iter2+1]
                iter2 += 1
            else:
                t = commontraj.points[iter1+1].t
                x = makemid(timedtraj.points[iter2].x, timedtraj.points[iter2].t, timedtraj.points[iter2 + 1].x,
                            timedtraj.points[iter2 + 1].t, t)
                y = makemid(timedtraj.points[iter2].y, timedtraj.points[iter2].t, timedtraj.points[iter2 + 1].y,
                            timedtraj.points[iter2 + 1].t, t)
                newp2 = Point(x, y, t)
                newp1 = commontraj.points[iter1+1]
                iter1 += 1
            lasttime = newtime
            pd = line2lineIDE(lastp1, newp1, lastp2, newp2)
            sum += pd
            lastp1 = newp1
            lastp2 = newp2
    return sum

class Distance:
    def __init__(self, N, M):  # N = length of C, M = length of Q
        self.D0 = np.zeros((N + 1, M + 1))
        self.flag = np.zeros((N, M))
        self.D0[0, 1:] = np.inf
        self.D0[1:, 0] = np.inf
        self.D = self.D0[1:, 1:]  

    def FRECHET(self, traj_C, traj_Q, skip=[]):
        n = len(traj_C)
        m = len(traj_Q)
        for i in range(n):
            for j in range(m):
                if self.flag[i, j] == 0:
                    cost = traj_C[i].distance(traj_Q[j])
                    self.D[i, j] = max(cost, min(self.D0[i, j], self.D0[i, j + 1], self.D0[i + 1, j]))
                    self.flag[i, j] = 1
        return self.D[n - 1, m - 1]
    

class Dtwdistance:
    def __init__(self, N, M): 
        self.D0 = np.zeros((N + 1, M + 1))
        self.flag = np.zeros((N, M))
        self.D0[0,1:] = np.inf
        self.D0[1:,0] = np.inf
        self.D = self.D0[1:,1:] 
        
    def DTW(self, traj_C, traj_Q, skip=[]):
        n = len(traj_C)
        m = len(traj_Q)
        for i in range(n):
            for j in range(m):
                if self.flag[i,j] == 0:
                    temp_res = []
                    temp_x = traj_C[i].x - traj_Q[j].x
                    temp_y = traj_C[i].y - traj_Q[j].y
                    temp_res.append(temp_x)
                    temp_res.append(temp_y)
                    cost = np.linalg.norm(temp_res)
                    self.D[i,j] = cost + min(self.D0[i,j],self.D0[i,j+1],self.D0[i+1,j])
                    self.flag[i,j] = 1
        return self.D[n-1, m-1]
    
def wd_dist(t1, t2):
    # sp1, ep1, sp2, ep2 = t1[0], t1[-1], t2[0], t2[-1]
    sp1, ep1, sp2, ep2 = Point_xy(t1[0].x, t1[0].y), Point_xy(t1[-1].x, t1[-1].y), Point_xy(t2[0].x, t2[0].y), Point_xy(t2[-1].x, t2[-1].y)
    seg1 = Segment(sp1, ep1)
    seg2 = Segment(sp2, ep2)
    seg_long, seg_short = compare(seg1, seg2)
    dist = seg_long.get_all_distance(seg_short)
    return dist
