from MDP import TrajRLclus
from rl_nn import DeepQNetwork
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from time import time
import os
import sys
import pickle
from trajdistance import traj2trajIED
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def save_dist(split_traj):
    length = len(split_traj)
    dist_matrix = np.zeros(shape=(length, length), dtype='float32')
    for i in range(length):
        for j in range(i + 1, length):
            temp_dist = traj2trajIED(split_traj[i].points, split_traj[j].points)
            dist_matrix[i][j] = temp_dist
            dist_matrix[j][i] = dist_matrix[i][j]
    return dist_matrix

def effective_rl(elist, savesubtraj, theta):
    count = 0
    print(len(elist))
    ori_overdist = env.basesim_E 
    while True: 
        count += 1
        for e in elist:  
            observation, steps = env.reset(e, 'E')
            for index in range(1, steps):
                action = RL.fast_online_act(observation)
                observation_, _ = env.step(e, action, index, 'E')
                observation = observation_

        ori_centers = []
        trajnum = 0
        subtrajectory = []

        for i in env.clusters_E.keys():
            for traj in env.clusters_E[i][1]:
                subtrajectory.append(traj)

            trajnum += len(env.clusters_E[i][0])
            ori_centers.append(env.clusters_E[i][2])

        cluster_dict = env.clusters_E
           
        env.update_cluster('E')
        overdist = env.basesim_E
        temp_dist = []
        for i in env.clusters_E.keys():
            d = traj2trajIED(ori_centers[i], env.clusters_E[i][2])
            temp_dist.append(d)
        
        filtered_list = [x for x in temp_dist if x != 1e10]
    
        if filtered_list:
            max_value = max(filtered_list)  
        else:
            max_value = 1e10
        
        if (max_value<theta) or count==8:
            break   
    
    if savesubtraj == 1:
        pickle.dump(subtrajectory, open('../data/' + 'ied_subtrajs_'+str(len(elist)), 'wb'), protocol=2)
    return overdist
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="estimate")
    parser.add_argument("-amount", type=int, default=1000, help="data size")
    parser.add_argument("-modeldir", default='../savemodels/kfoldmodels2', help="model folder")
    parser.add_argument("-modelchoose", type=int, default=0, help="choose model")
    parser.add_argument("-testdata", default='../data/Tdrive_testdata', help="choose model")
    parser.add_argument("-base_cluster", default='../data/tdrive_clustercenter', help="base_cluster")
    parser.add_argument("-savesubtraj", type=int, default=0, help="whether save subtraj")
    parser.add_argument("-theta", type=float, default=0.8, help="threshold of iteration")
    parser.add_argument("-caltime", type=int, default=0, help="display time")
    
    args = parser.parse_args()
    
    env = TrajRLclus(args.testdata, args.base_cluster, args.base_cluster)
    RL = DeepQNetwork(env.n_features, env.n_actions)
    modelnames = os.listdir(args.modeldir)
    model = args.modeldir + '/' + modelnames[args.modelchoose]
    RL.load(model) 
    elist = [i for i in range(args.amount)]
    st = time()
    overdist = effective_rl(elist, args.savesubtraj, args.theta) 
    et = time()
    
    print('--------OD--------', overdist)
    if args.caltime == 1:
        print('--------estimate time--------', et - st, 'seconds')