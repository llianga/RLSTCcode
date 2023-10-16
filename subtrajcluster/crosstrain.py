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

def evaluate(elist):
    env.allsubtraj_E = []
    for e in elist:
        observation, steps = env.reset(e, 'E')
        for index in range(1, steps):
            action = RL.online_act(observation)
            observation_, _ = env.step(e, action, index, 'E')
            observation = observation_
    odist_e = compute_overdist(env.clusters_E)
    aver_cr = float(odist_e/env.basesim_E)
    return aver_cr

def train(amount, saveclus, sidx, eidx):
    batch_size = 32
    check = 999999
    TR_CR = []
    Round = 1 
    idxlist = [i for i in range(amount)]
    while Round != 0:
        random.shuffle(idxlist)
        Round = Round - 1
        REWARD = 0.0
        for episode in idxlist:
            observation, steps = env.reset(episode, 'T') 
            for index in range(1, steps): 
                if index == steps - 1:
                    done = True
                else:
                    done = False
                action = RL.act(observation)
                observation_, reward = env.step(episode, action, index, 'T')
                
                if reward != 0:
                    REWARD = REWARD + reward
                RL.remember(observation, action, reward, observation_, done)
                if done:
                    break
                if len(RL.memory) > batch_size:
                    RL.replay(episode, batch_size)
                    RL.soft_update(0.001) 
                observation = observation_
            if episode % 500 == 0 and episode != 0:
                aver_cr = evaluate([i for i in range(sidx, eidx)])  
               
                for i in env.clusters_E.keys():
                    env.clusters_E[i][0] = []
                    env.clusters_E[i][1] = []
                    env.clusters_E[i][3] = defaultdict(list)
                
                if aver_cr < check or episode % 500 == 0:
                    RL.save(saveclus + '/sub-RL-' + str(aver_cr) + '.h5')
                    # print('Save model at episode {} with competive ratio {}'.format(episode, aver_cr))
                if aver_cr < check:
                    check = aver_cr
                    print('maintain the current best', check)
            
            od = env.overall_sim
        env.update_cluster('T') 
        
def ksplitdataset(trajs,k, dataset):
    index_list = list(range(len(trajs)))
    random.shuffle(index_list)
    shuffled_trajs = [trajs[i] for i in index_list]
    
    chunk_size = len(shuffled_trajs) // k
 
    chunks = [shuffled_trajs[i:i+chunk_size] for i in range(0, len(shuffled_trajs), chunk_size)]
    for j in range(k):
        test_set = chunks[j]

        train_set = [item for i, sublist in enumerate(chunks) if i != j for item in sublist]
        testfilename = '../data/'+ dataset + '_testset_'+str(j)
        trainfilename = '../data/'+ dataset + '_trainset_'+str(j)
        pickle.dump(test_set, open(testfilename, 'wb'), protocol=2)
        pickle.dump(train_set, open(trainfilename, 'wb'), protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    
    parser.add_argument("-trajfile", default='../data/Tdrive_norm_traj', help="baseclusTfile")
    parser.add_argument("-baseclusT", default='../data/tdrive_clustercenter', help="baseclusTfile")
    parser.add_argument("-baseclusE", default='../data/tdrive_clustercenter', help="baseclusEfile")
    parser.add_argument("-saveclus", default='../models/kfoldmodels', help="saveclusfile")
    parser.add_argument("-k", type=int, default=5, help="k")
    parser.add_argument("-dataset", default='tdrive', help="dataset")
    args = parser.parse_args()
    
    trajs = pickle.load(open(args.trajfile, 'rb'))
    ksplitdataset(trajs,args.k, args.dataset)
    
    for i in range(args.k):
        trainfilename = '../data/'+args.dataset+'_trainset_'+str(i)
        savecluspath = args.saveclus+str(i)
        if not os.path.exists(savecluspath):
            os.makedirs(savecluspath)
        trainset = pickle.load(open(trainfilename, 'rb'))
        validation_percent = 0.1
        total_length = len(trainset)
        amount = int(total_length * (1-validation_percent))
        sidx = int(total_length * (1-validation_percent))
        eidx = total_length
        
        env = TrajRLclus(trainfilename, args.baseclusT, args.baseclusE)
        RL = DeepQNetwork(env.n_features, env.n_actions, 'None')
        train(amount, savecluspath, sidx, eidx)
    
    
    