import sys
import os
import pickle
import random
import numpy as np
from MDP import TrajRLclus
from rl_nn import DeepQNetwork
from time import time
from collections import defaultdict
from cluster import compute_overdist
from trajdistance import traj2trajIED
import argparse

def ksplitdataset(trajs,k):
    index_list = list(range(len(trajs)))
    random.shuffle(index_list)
    shuffled_trajs = [trajs[i] for i in index_list]
    chunk_size = len(shuffled_trajs) // k
    chunks = [shuffled_trajs[i:i+chunk_size] for i in range(0, len(shuffled_trajs), chunk_size)]
    for j in range(k):
        test_set = chunks[j]
        train_set = [item for i, sublist in enumerate(chunks) if i != j for item in sublist]
        testfilename = '../data/'+'tdrive_testset'+str(j)
        trainfilename = '../data/'+'tdrive_trainset'+str(j)
        pickle.dump(test_set, open(testfilename, 'wb'), protocol=2)
        pickle.dump(train_set, open(trainfilename, 'wb'), protocol=2)

def run_effective_rl(elist, calsse=0, theta=0.01): #1
    count = 0  
    ori_overdist = env.basesim_E 
    while True: 
        count += 1
        for e in elist: 
            observation, steps = env.reset(e, 'E')
            for index in range(1, steps):
                action = RL.fast_online_act(observation)
                observation_, _ = env.step(e, action, index, 'E')
                observation = observation_
    
        ori_centers, subtrajectory = [], []
        trajnum = 0

        for i in env.clusters_E.keys():
            for traj in env.clusters_E[i][1]:
                subtrajectory.append(traj)

            trajnum += len(env.clusters_E[i][0])
            ori_centers.append(env.clusters_E[i][2])

        cluster_dict = env.clusters_E  
        sse, sse_ori, sse_count, num = 0, 0, 0, 10
        if calsse == 1:
            for i in env.clusters_E.keys():
                cluster_size = len(env.clusters_E[i][1])
                subtrajs = cluster_dict[i][1]
                dist_sum, dist_square, dist_sum_ori, dist_square_ori = 0, 0, 0, 0
                for j in range(cluster_size): 
                    for k in range(j+1,cluster_size):
                        dist = traj2trajIED(subtrajs[j].points, subtrajs[k].points)
                        dist_square_ori = dist*dist
                        dist_sum_ori += dist_square_ori
                        if dist == 1e10:
                            sse_count += 1
                        else:
                            dist_square = dist*dist
                        dist_sum += dist_square
                if cluster_size != 0:
                    dist_sum_clus = dist_sum/(2*num*cluster_size)
                    dist_sum_clus_ori = dist_sum_ori/(2*num*cluster_size)
                    sse += dist_sum_clus
                    sse_ori += dist_sum_clus_ori  
                     
        temp_dist = []
        env.update_cluster('E') 
        overdist = env.basesim_E
        for i in env.clusters_E.keys():
            if len(env.clusters_E[i][2]) == 0:
                continue
            d = traj2trajIED(ori_centers[i], env.clusters_E[i][2])
            temp_dist.append(d)
        if max(temp_dist)<theta or count == 2:
            break
    return overdist, sse_ori

def estimate(testsetfile, model, calsse, theta):
    RL.load(model)
    testset = pickle.load(open(testsetfile, 'rb'))
    elist = [i for i in range(len(testset))]
    od, sse = run_effective_rl(elist,calsse,theta)
    return od, sse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="crossvalidate")
    parser.add_argument("-baseclusT", default='../data/tdrive_clustercenter', help="baseclusTfile")
    parser.add_argument("-baseclusE", default='../data/tdrive_clustercenter', help="baseclusEfile")
    parser.add_argument("-saveclus", default='../savemodels/kfoldmodels', help="saveclusfile")
    parser.add_argument("-k", type=int, default=5, help="k")
    parser.add_argument("-calsse", type=int, default=0, help="calsse")
    parser.add_argument("-dataset", default='tdrive', help="dataset")
    parser.add_argument("-theta", type=float, default=0.8, help="theta")
    
    res1, res2 = [], []
    args = parser.parse_args()
    for i in range(args.k):
        savecluspath = args.saveclus+str(i)
        modelnames = os.listdir(savecluspath)
        model = savecluspath + '/' + modelnames[0]
        testfilename = '../data/'+args.dataset+'_testset'+str(i)
        env = TrajRLclus(testfilename, args.baseclusT, args.baseclusE)
        RL = DeepQNetwork(env.n_features, env.n_actions, 'None')
        OD, sse = estimate(testfilename, model,args.calsse, args.theta)
        res1.append(OD)
        res2.append(sse)
    if args.calsse == 1:
        print('-----average res-----', np.mean(res1), np.mean(res2))
    else:
        print('-----average res-----', np.mean(res1))
    