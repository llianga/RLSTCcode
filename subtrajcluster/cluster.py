import numpy as np
from point import Point
from segment import Segment
from traj import Traj
from trajdistance import traj2trajIED, getstaticIED, makemid, line2lineIDE

def incremental_sp(traj1,traj2, k_dict, k):
    t1s, t1e, t2s, t2e = traj1[0].t, traj1[-1].t, traj2[0].t, traj2[-1].t
    if t1s >= t2e or t1e <= t2s:
        k_dict[k]['mid_dist'] = 1e10
        k_dict[k]['real_dist'] = 1e10
        k_dict[k]['lastp'] = traj2[0]
        k_dict[k]['j'] = 0
        return k_dict
    e_i = len(traj2) - 1
    if t1e >= t2e:
        d = traj2trajIED(traj1, traj2)
        k_dict[k]['mid_dist'] = d
        k_dict[k]['real_dist'] = d
        k_dict[k]['lastp'] = Point(traj2[-1].x, traj2[-1].y, t1e)
        k_dict[k]['j'] = len(traj2) - 1
        return k_dict
    else:
        while traj2[e_i].t > t1e:
            e_i -= 1
        if traj2[e_i].t == t1e:
            lastp = traj2[e_i]
            front_traj2 = traj2[:e_i + 1]
            mid_dist = traj2trajIED(traj1, front_traj2)
            back_traj2 = traj2[e_i:]
            back_dist = getstaticIED(back_traj2, traj1[-1].x,traj1[-1].y, t1e, t2e)
            k_dict[k]['mid_dist'] = mid_dist
            k_dict[k]['real_dist'] = mid_dist + back_dist
            k_dict[k]['lastp'] = lastp
            k_dict[k]['j'] = e_i
        if traj2[e_i].t < t1e:
            front_traj2 = traj2[:e_i + 1]
            x = makemid(traj2[e_i].x, traj2[e_i].t, traj2[e_i + 1].x, traj2[e_i + 1].t, t1e)
            y = makemid(traj2[e_i].y, traj2[e_i].t, traj2[e_i + 1].y, traj2[e_i + 1].t, t1e)
            lastp = Point(x, y, t1e)
            front_traj2.append(lastp)
            back_traj2 = traj2[e_i + 1:]
            back_traj2.insert(0, lastp)
            mid_dist = traj2trajIED(traj1, front_traj2)
            back_dist = getstaticIED(back_traj2, traj1[-1].x, traj1[-1].y, t1e, t2e)
            k_dict[k]['mid_dist'] = mid_dist
            k_dict[k]['real_dist'] = mid_dist + back_dist
            k_dict[k]['lastp'] = lastp
            k_dict[k]['j'] = e_i
        return k_dict
    
def incremental_nsp(traj1, traj2, k_dict, k, i):
    t1s, t1e, t2s, t2e = traj1[0].t, traj1[-1].t, traj2[0].t, traj2[-1].t
    if t1s >= t2e or t1e <= t2s:
        k_dict[k]['mid_dist'] = 1e10
        k_dict[k]['real_dist'] = 1e10
        k_dict[k]['lastp'] = traj2[0]
        k_dict[k]['j'] = 0
        return k_dict
    if k_dict[k]['mid_dist'] == 1e10:
        k_dict = incremental_sp(traj1, traj2, k_dict, k)
        return k_dict
    if t2e == t1e:
        temptraj1 = []
        temptraj1.append(traj1[i-1])
        temptraj1.append(traj1[i])
        temptraj2 = traj2[k_dict[k]['j']:]
        if traj2[k_dict[k]['j']].t<=k_dict[k]['lastp'].t:
            temptraj2[0]= k_dict[k]['lastp']
        else:
            temptraj2.insert(0,k_dict[k]['lastp'])

        d = k_dict[k]['mid_dist'] + traj2trajIED(temptraj1,temptraj2)
        k_dict[k]['mid_dist'] = d
        k_dict[k]['real_dist'] = k_dict[k]['mid_dist']
        k_dict[k]['lastp'] = traj2[-1]
        k_dict[k]['j'] = len(traj2)-1
        return k_dict
    if t2e < t1e and t2e > traj1[i-1].t:
        temptraj1 = []
        temptraj1.append(traj1[i-1])
        temptraj1.append(traj1[i])
        temptraj2 = traj2[k_dict[k]['j']:]
        if traj2[k_dict[k]['j']].t<=k_dict[k]['lastp'].t:
            temptraj2[0]= k_dict[k]['lastp']
        else:
            temptraj2.insert(0,k_dict[k]['lastp'])
        d = traj2trajIED(temptraj1, temptraj2)
        lastp = Point(traj2[-1].x, traj2[-1].y, t1e)
        k_dict[k]['mid_dist'] = k_dict[k]['mid_dist'] + d
        k_dict[k]['real_dist'] = k_dict[k]['mid_dist']
        k_dict[k]['lastp'] = lastp
        k_dict[k]['j'] = len(traj2) -1
        return k_dict
    if t2e < t1e and t2e <= traj1[i-1].t:
        newp = Point(traj2[-1].x, traj2[-1].y, t1e)
        d = line2lineIDE(traj1[i-1], traj1[i], k_dict[k]['lastp'], newp)
        k_dict[k]['mid_dist'] = k_dict[k]['mid_dist'] + d
        k_dict[k]['real_dist'] = k_dict[k]['mid_dist']
        k_dict[k]['lastp'] = newp
        k_dict[k]['j'] = len(traj2) - 1
        return k_dict
    if t1e < t2e:
        e_i = len(traj2)-1
        while traj2[e_i].t > t1e:
            e_i -= 1
        if traj2[e_i].t == t1e:
            lastp = traj2[e_i]
            front_traj2 = traj2[k_dict[k]['j']:e_i+1]
            if k_dict[k]['lastp'].t >= front_traj2[0].t:
                front_traj2[0]=k_dict[k]['lastp']
            else:
                front_traj2.insert(0, k_dict[k]['lastp'])
            midtraj1 = []
            midtraj1.append(traj1[i-1])
            midtraj1.append(traj1[i])
            mid_dist = traj2trajIED(midtraj1, front_traj2)
            back_traj2 = traj2[e_i:]
            back_dist = getstaticIED(back_traj2, traj1[-1].x, traj1[-1].y, t1e, t2e)

            k_dict[k]['mid_dist'] = k_dict[k]['mid_dist']+ mid_dist
            k_dict[k]['real_dist'] = k_dict[k]['mid_dist'] + back_dist
            k_dict[k]['lastp'] = lastp
            k_dict[k]['j'] = e_i
        if traj2[e_i].t < t1e:
            front_traj2 = traj2[k_dict[k]['j']:e_i + 1]
            if k_dict[k]['lastp'].t >= front_traj2[0].t:
                front_traj2[0]=k_dict[k]['lastp']
            else:
                front_traj2.insert(0, k_dict[k]['lastp'])

            x = makemid(traj2[e_i].x, traj2[e_i].t, traj2[e_i + 1].x, traj2[e_i + 1].t, t1e)
            y = makemid(traj2[e_i].y, traj2[e_i].t, traj2[e_i + 1].y, traj2[e_i + 1].t, t1e)
            lastp = Point(x, y, t1e)
            front_traj2.append(lastp)
            back_traj2 = traj2[e_i + 1:]
            back_traj2.insert(0, lastp)
            temptraj1 = []
            temptraj1.append(traj1[i-1])
            temptraj1.append(traj1[i])
            mid_dist = traj2trajIED(temptraj1, front_traj2)
            back_dist = getstaticIED(back_traj2, traj1[-1].x, traj1[-1].y, t1e, t2e)
            k_dict[k]['mid_dist'] = k_dict[k]['mid_dist'] + mid_dist
            k_dict[k]['real_dist'] = k_dict[k]['mid_dist'] + back_dist
            k_dict[k]['lastp'] = lastp
            k_dict[k]['j'] = e_i
        return k_dict

def incremental_IED(traj1, traj2, k_dict, k, i, sp_i):
    if i == sp_i + 1 : 
        k_dict = incremental_sp(traj1, traj2, k_dict, k)
    else:
        k_dict = incremental_nsp(traj1, traj2, k_dict, k, i)
    return k_dict

def incremental_mindist(trajectory, start_index, current_index, k_dict, cluster_dict, episode): 
    min_dist = 1e10
    cluster_id = -1

    count = 0
    traj_points = trajectory.points[start_index:current_index+1]
    s_i = 0
    c_i = current_index - start_index
    
    for i in cluster_dict.keys(): 
        center = cluster_dict[i][2]
        if len(center) == 0:
            continue
        dicts = incremental_IED(traj_points,center,k_dict,i,c_i,s_i)
        
        if count == 0: 
            min_dist = dicts[i]['real_dist']
            cluster_id = i
        else:
            if dicts[i]['real_dist'] < min_dist: 
                min_dist = dicts[i]['real_dist']
                cluster_id = i
        count += 1
    return min_dist, cluster_id

def add2clusdict(points, clus_dict,k):
    for key_t in clus_dict[k][3].keys():
        if key_t >= points[0].t and key_t <= points[-1].t:
            clus_dict[k][3][key_t][1] += 1
    for i in range(len(points)): 
        curr_t = points[i].t
        if curr_t not in clus_dict[k][3]:
            clus_dict[k][3][curr_t]=[[points[i]],1,points[i].x,points[i].y]
            for j in range(len(clus_dict[k][1]) - 1):
                traj = clus_dict[k][1][j]
                if points[i].t >= traj.ts and points[i].t <= traj.te:
                    clus_dict[k][3][points[i].t][1] += 1
        else:
            clus_dict[k][3][curr_t][0].append(points[i])
            clus_dict[k][3][curr_t][2] += points[i].x
            clus_dict[k][3][curr_t][3] += points[i].y

def computecenter(clus_dict, k, threshold_num, threshold_t):
    keys = sorted(clus_dict[k][3].keys())
    sortkeys, threshold_nums = [], []
    for key in keys:
        threshold_nums.append(clus_dict[k][3][key][1])
        if clus_dict[k][3][key][1] >= threshold_num:
            sortkeys.append(key)
    if len(sortkeys) == 0:
        mean_num = np.mean(threshold_nums)
        for key in keys:
            if clus_dict[k][3][key][1] >= mean_num:
                sortkeys.append(key)
    start_i = 0
    i = start_i + 1
    count = len(clus_dict[k][3][sortkeys[start_i]][0])
    sum_x, sum_y, sum_t = clus_dict[k][3][sortkeys[start_i]][2], clus_dict[k][3][sortkeys[start_i]][3], sortkeys[start_i]
    center = []
    while i < len(sortkeys):
        if sortkeys[i] - sortkeys[start_i] <= threshold_t:
            count += len(clus_dict[k][3][sortkeys[i]][0])
            sum_x += clus_dict[k][3][sortkeys[i]][2]
            sum_y += clus_dict[k][3][sortkeys[i]][3]
            sum_t += sortkeys[i]
            if i == len(sortkeys)-1:
                count_t = i - start_i + 1
                aver_x, aver_y, aver_t  = sum_x / count, sum_y / count, sum_t / count_t
                point = Point(aver_x, aver_y, aver_t)
                center.append(point)
            i += 1
        else:
            count_t = i - start_i
            aver_x, aver_y, aver_t = sum_x/count, sum_y/count, sum_t/count_t
            point = Point(aver_x, aver_y, aver_t)
            center.append(point)
            start_i = i
            i = start_i + 1
            count = len(clus_dict[k][3][sortkeys[start_i]][0])
            sum_x, sum_y, sum_t = clus_dict[k][3][sortkeys[start_i]][2], clus_dict[k][3][sortkeys[start_i]][3], sortkeys[start_i]
    return center

def compute_overdist(cluster_dict):
    count = 0
    sumval = 0
    for i in cluster_dict.keys():
        if len(cluster_dict[i][0])!=0:
            count += len(cluster_dict[i][0])
            sumval += sum(cluster_dict[i][0])
    overdist = sumval/count
    return overdist

def update_centers(cluster_dict, threshold_num, threshold_t):
    for i in cluster_dict.keys():
        if len(cluster_dict[i][0])!=0:
            center = computecenter(cluster_dict,i,threshold_num, threshold_t)
            if len(center) != 0:
                cluster_dict[i][2] = center
    overdist = compute_overdist(cluster_dict)
    return overdist, cluster_dict
    
