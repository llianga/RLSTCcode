import sys
import pickle
import numpy as np
from segment import Segment, compare
from point import Point
from point_xy import Point_xy, _point2line_distance
from traj import Traj
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from time import time
from trajdistance import traj2trajIED, makemid
import argparse


def agglomerative_clusteing_with_dist(distance_matrix, split_traj, cluster_num):
    cluster_segment = defaultdict(list)
    cluster = AgglomerativeClustering(n_clusters=cluster_num, affinity='precomputed', linkage='average').fit(distance_matrix)
    cluster_lables = cluster.labels_
    for i in range(len(cluster_lables)):
        cluster_segment[cluster_lables[i]].append(split_traj[i])
    return cluster_segment

def agglomerative_clusteing_without_dist(split_traj, cluster_num):
    cluster_segment = defaultdict(list)
    distance_matrix = sim_affinity(split_traj)
    cluster = AgglomerativeClustering(n_clusters=cluster_num, affinity='precomputed', linkage='average').fit(distance_matrix)
    cluster_lables = cluster.labels_
    for i in range(len(cluster_lables)):
        cluster_segment[cluster_lables[i]].append(split_traj[i])
    return cluster_segment

def dbscan_with_dist(distance_matrix, split_traj,ep, sample):
    cluster_segment = defaultdict(list)
    remove_cluster = dict()
    count = 0
    for i in range(len(distance_matrix[0])):
        count += distance_matrix[0][i]
    ep = float(ep)
    sample = float(sample)
    cluster = DBSCAN(eps=ep, min_samples=sample, metric='precomputed').fit(distance_matrix)
    cluster_lables = cluster.labels_
    for i in range(len(cluster_lables)):
        cluster_segment[cluster_lables[i]].append(split_traj[i])
    return cluster_segment

def dbscan_without_dist(split_traj,ep, sample):
    cluster_segment = defaultdict(list)
    remove_cluster = dict()
    startcal = time()
    distance_matrix = sim_affinity(split_traj)
    endcal = time()
    # print('cal dist matrix time: ', endcal - startcal, 'seconds')
    ep = float(ep)
    sample = float(sample)
    cluster = DBSCAN(eps=ep, min_samples=sample, metric='precomputed').fit(distance_matrix)
    cluster_lables = cluster.labels_
    for i in range(len(cluster_lables)):
        cluster_segment[cluster_lables[i]].append(split_traj[i])
    return cluster_segment

def kMeans_without_dist(cluster_dict, split_traj):   
    trajsize = len(split_traj)
    clusterAssment = np.mat(np.zeros((trajsize,2)))
    clusterChanged = True   
    count = 0
    while clusterChanged:
        cluster_segment = defaultdict(list)    
        count += 1
        clusterChanged = False
        for i in range(trajsize):  
            minDist = float("inf") 
            minIndex = -1
            for j in cluster_dict.keys():
                represent_traj = cluster_dict[j][1]      
                dist = traj2trajIED(represent_traj.points, split_traj[i].points)
                if dist < minDist:
                    minDist = dist 
                    minIndex = j 
            cluster_segment[minIndex].append(split_traj[i])
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True      
            clusterAssment[i,:] = minIndex,minDist**2  
        cluster_dict = compute_statistic(cluster_segment)
        if count == 30:
            break
    return cluster_dict

def sim_affinity(split_traj):
    length = len(split_traj)
    dist_matrix = np.zeros(shape=(length, length), dtype='float32')
    for i in range(length):
        for j in range(i + 1, length):
            temp_dist = traj2trajIED(split_traj[i].points,split_traj[j].points)
            dist_matrix[i][j] = temp_dist
            dist_matrix[j][i] = dist_matrix[i][j]
    return dist_matrix

def compute_center(cluster_segments, threshold, min_dist): 
    segment_points, timesets, center = [], [], []
    for i in range(len(cluster_segments)):
        segment_points.append(cluster_segments[i].points)
        timesets.append(cluster_segments[i].ts)
        timesets.append(cluster_segments[i].te)
    timesets = sorted(timesets) 
    for t in timesets:
        intersect, sum_x, sum_y = 0, 0, 0
        for i in range(len(segment_points)):
            if segment_points[i][0].t > t or segment_points[i][-1].t < t:
                continue
            else:
                intersect += 1
                s_i = 0
                while segment_points[i][s_i].t < t:
                    s_i += 1
                if s_i != 0 and segment_points[i][s_i].t != t:
                    x = makemid(segment_points[i][s_i - 1].x, segment_points[i][s_i - 1].t, segment_points[i][s_i].x, segment_points[i][s_i].t, t)
                    y = makemid(segment_points[i][s_i - 1].y, segment_points[i][s_i - 1].t, segment_points[i][s_i].y, segment_points[i][s_i].t, t)
                    sum_x += x
                    sum_y += y
                if s_i == 0 or segment_points[i][s_i].t == t:
                    sum_x += segment_points[i][s_i].x
                    sum_y += segment_points[i][s_i].y
        if intersect >= threshold:
            new_x, new_y = sum_x/intersect, sum_y/intersect
            newpoint = Point(new_x, new_y, t)
            _size = len(center) - 1
            if _size < 0 or (_size >= 0 and newpoint.distance(center[_size]) > min_dist):
                center.append(newpoint)
    return center

def compute_statistic(cluster_segment, min_lines=20, min_dist=0.005):
    cluster_dict = defaultdict(list)
    representive_point = defaultdict(list)
    for i in cluster_segment.keys():
        temp_subtrajs = []
        temp = []
        temp_dists = []
        center = compute_center(cluster_segment[i], min_lines, min_dist)
        if len(center) == 0:
            centertraj = cluster_segment[i][0]
            center = cluster_segment[i][0].points
            
        centertraj=Traj(center,len(center),center[0].t, center[-1].t) 
        """compute distance between subtrajs and center"""
        cluster_size = len(cluster_segment.get(i))
        for j in range(cluster_size):
            dist = traj2trajIED(cluster_segment[i][j].points, center)
            if dist!=1e10:
                temp_dists.append(dist)
                temp_subtrajs.append(cluster_segment[i][j]) 

            temp.append(dist)
        aver_dist = np.mean(temp)
        cluster_dict[i].append(aver_dist)  
        cluster_dict[i].append(centertraj)  
        cluster_dict[i].append(temp)  
        cluster_dict[i].append(cluster_segment[i]) 
        cluster_dict[i].append(temp_dists) 
        cluster_dict[i].append(temp_subtrajs) 

    return cluster_dict

def init_cluster(split_traj, cluster_dict_ori, clustermethod, ep, sample):
    traj_num, count_sim, less_sim, less_traj = 0, 0, 0, 0
    if clustermethod == 'AHC':
        cluster_segment = agglomerative_clusteing_without_dist(split_traj, cluster_num=10) 
        cluster_dict = compute_statistic(cluster_segment)
    if clustermethod == 'dbscan':
        cluster_segment = dbscan_without_dist(split_traj, ep, sample) 
        cluster_dict = compute_statistic(cluster_segment)
    if clustermethod == 'kmeans':
        cluster_dict = kMeans_without_dist(cluster_dict_ori,split_traj)

    for i in cluster_dict.keys():    
        count_sim += np.sum(cluster_dict[i][2])
        traj_num += len(cluster_dict[i][3])    
        less_sim += np.sum(cluster_dict[i][4])
        less_traj += len(cluster_dict[i][5])
    if traj_num == 0:
        overall_sim = 1e10
    else:
        overall_sim = count_sim / traj_num
    if less_traj == 0:
        over_sim = 1e10
    else:
        over_sim = less_sim / less_traj
    return cluster_dict, overall_sim, traj_num, over_sim, less_traj

if __name__ == "__main__":
    res = []
    parser = argparse.ArgumentParser(description="splitmethod")
    parser.add_argument("-splittrajfile", default='../data/ied_subtrajs_100', help="subtraj file")
    parser.add_argument("-clustermethod", default='dbscan', help="subtraj file")
    parser.add_argument("-baseclusterfile", default='../data/tdrive_clustercenter', help="subtraj file")
    parser.add_argument("-ep", type=float, default=0.005, help="ep")
    parser.add_argument("-sample", type=int, default=70, help="sample")
    
    args = parser.parse_args()
    centers = pickle.load(open(args.baseclusterfile, 'rb'))
    cluster_dict_ori = centers[0][2]
    split_traj = pickle.load(open(args.splittrajfile, 'rb'))
    cluster_dict, overall_sim, traj_num, over_sim, less_traj = init_cluster(split_traj, cluster_dict_ori, args.clustermethod, args.ep, args.sample)
    
    res.append((overall_sim, over_sim, cluster_dict))
    print('-----over_sim-----', over_sim)