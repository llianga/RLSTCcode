import pickle
import sys
import numpy as np
import random
from point import Point
from segment import Segment
from traj import Traj
from collections import defaultdict
from trajdistance import traj2trajIED
import argparse

def initialize_centers(data, K):
    centers = [random.choice(data)]
    while len(centers) < K:
        distances = [min([traj2trajIED(center.points, traj.points) for center in centers]) for traj in data]
        new_center = data[distances.index(max(distances))]
        centers.append(new_center)
    return centers

def getbaseclus(trajs, k, subtrajs):
    centers = initialize_centers(trajs, k)
    cluster_dict = defaultdict(list)
    cluster_segments = defaultdict(list)
    dists_dict = defaultdict(list)
    for i in range(len(subtrajs)):
        mindist = float("inf")
        minidx = 0
        for j in range(k): 
            dist = traj2trajIED(centers[j].points, subtrajs[i].points)
            if dist == 1e10:
                continue
            if dist < mindist:
                mindist = dist
                minidx = j
        if mindist == float("inf"):
            continue
        else:
            cluster_segments[minidx].append(subtrajs[i])
            dists_dict[minidx].append(mindist)
    
    for i in range(k):
        if len(cluster_segments[i]) == 0:
            cluster_segments[minidx].append(centers[i])
            dists_dict[i].append(0)
        
    for i in cluster_segments.keys():
        center = centers[i]
        temp_dist = dists_dict[i]
        aver_dist = np.mean(temp_dist)
        cluster_dict[i].append(aver_dist)
        cluster_dict[i].append(center)
        cluster_dict[i].append(temp_dist)
        cluster_dict[i].append(cluster_segments[i])
    
    return cluster_dict

def saveclus(k, subtrajs, trajs, amount):
    trajs = trajs[:amount]
    cluster_dict = getbaseclus(trajs, k, subtrajs)
    count_sim, traj_num = 0, 0
    for i in cluster_dict.keys():
        count_sim += np.sum(cluster_dict[i][2])
        traj_num += len(cluster_dict[i][3])
    if traj_num == 0:
        overall_sim = 1e10
    else:
        overall_sim = count_sim / traj_num
    res = []
    res.append((overall_sim, overall_sim, cluster_dict))
    return res
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="crossvalidate")
    parser.add_argument("-subtrajsfile", default='../data/traclus_subtrajs', help="baseclusTfile")
    parser.add_argument("-trajsfile", default='../data/Tdrive_norm_traj', help="baseclusEfile")
    parser.add_argument("-k", type=int, default=10, help="k")
    parser.add_argument("-amount", type=int, default=1000, help="k")
    parser.add_argument("-centerfile", default='../data/tdrive_clustercenter', help="baseclusEfile")
    
    # subtrajsfile = sys.argv[1]
    # trajsfile = sys.argv[2]
    # k = int(sys.argv[3])
    # amount = int(sys.argv[4])
    # centerfile = sys.argv[5]
    args = parser.parse_args()
    subtrajs = pickle.load(open(args.subtrajsfile, 'rb'))
    trajs = pickle.load(open(args.trajsfile, 'rb'))
    
    res = saveclus(args.k, subtrajs, trajs, args.amount)
    pickle.dump(res, open(args.centerfile, 'wb'), protocol=2)
    