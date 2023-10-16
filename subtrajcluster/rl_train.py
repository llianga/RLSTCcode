import sys
import numpy as np
import tensorflow as tf
# import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from time import time
from MDP import TrajRLclus
from rl_nn import DeepQNetwork
from cluster import compute_overdist
import random
import os
from collections import defaultdict
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

np.random.seed(1)
tf.set_random_seed(1)


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
    start = time()
    Round = 2
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
                    RL.soft_update(0.05)
                observation = observation_

            if episode % 500 == 0 and episode != 0:
                aver_cr = evaluate([i for i in range(sidx, eidx)])  #
                
                for i in env.clusters_E.keys():
                    env.clusters_E[i][0] = []
                    env.clusters_E[i][1] = []
                    env.clusters_E[i][3] = defaultdict(list)
            
                CR = compute_overdist(env.clusters_T)
                # print('Training CR: {}, Validation CR: {}'.format(CR, aver_cr))
                
                if aver_cr < check or episode % 500 == 0:
                    RL.save(saveclus + '/sub-RL-' + str(aver_cr) + '.h5')
                if aver_cr < check:
                    check = aver_cr
                    print('maintain the current best', check)
            
            cr = compute_overdist(env.clusters_T)
            train_acc = cr/env.basesim_T
            od = env.overall_sim  
        env.update_cluster('T')  
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("-amount", type=int, default=500, help="training trajectory")
    parser.add_argument("-traindata", default='../data/Tdrive_norm_traj', help="baseclusTfile")
    parser.add_argument("-baseclusT", default='../data/tdrive_clustercenter', help="baseclusTfile")
    parser.add_argument("-baseclusE", default='../data/tdrive_clustercenter', help="baseclusEfile")
    parser.add_argument("-saveclus", default='../models', help="saveclusfile")
    args = parser.parse_args()

    env = TrajRLclus(args.traindata, args.baseclusT, args.baseclusE)
    RL = DeepQNetwork(env.n_features, env.n_actions)
    validation_percent = 0.1
    
    sidx = int(args.amount * (1-validation_percent))
    eidx = args.amount
    train(args.amount, args.saveclus, sidx, eidx)
    
    




