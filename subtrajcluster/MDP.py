import pickle
import numpy as np
from collections import defaultdict
from point import Point
from segment import Segment
from traj import Traj
from trajdistance import traj2trajIED
from cluster import incremental_mindist, add2clusdict, update_centers


class TrajRLclus():
    def __init__(self, cand_train, base_centers_T, base_centers_E):
        self.n_actions = 2  
        self.n_features = 5 
        centers_T = pickle.load(open(base_centers_T, 'rb'), encoding='bytes')
        centers_data_T = centers_T[0][2]
        k = len(centers_data_T) 
        self.RW = 0.0
        self.clusters_T = defaultdict(list) 
        self.clusters_E = defaultdict(list)

        for i in range(k):
            self.clusters_T[i].append([])
            self.clusters_T[i].append([])
        for i in range(k):
            self.clusters_E[i].append([])
            self.clusters_E[i].append([])
        self.allsubtrajs_T = []
        self.allsubtrajs_E = []
        self.allsubindexes_E = []
        self._load(cand_train, base_centers_T, base_centers_E)
        
    def _load(self,cand_train, base_centers_T, base_centers_E):
        cand_train_data = pickle.load(open(cand_train, 'rb'), encoding='bytes')
        self.trajsdata = cand_train_data 
        centers_T = pickle.load(open(base_centers_T, 'rb'), encoding='bytes')
        self.basesim_T = centers_T[0][1]
        centers_data_T = centers_T[0][2] 
        centers_E = pickle.load(open(base_centers_E, 'rb'), encoding='bytes')
        self.basesim_E = centers_E[0][1]
        centers_data_E = centers_E[0][2]
        for i in range(len(centers_data_T)):
            self.clusters_T[i].append(centers_data_T[i][1].points)
            self.clusters_T[i].append(defaultdict(list))
        for i in range(len(centers_data_E)):
            self.clusters_E[i].append(centers_data_E[i][1].points)
            self.clusters_E[i].append(defaultdict(list))
    
    def reset(self, episode, label = 'T'):
        self.split_point = 0
        self.length = self.trajsdata[episode].size
        self.k_dict = dict() 
        k = len(self.clusters_T)
        for i in range(k):
            self.k_dict[i] = dict()
            self.k_dict[i]['mid_dist'] = 1e10
            self.k_dict[i]['real_dist'] = 1e10
            self.k_dict[i]['lastp'] = Point(0, 0, 0)
            self.k_dict[i]['j'] = 0
        self.traj_num = 1
        self.minsim = 0
        self.k = 0
        if label == 'T':
            self.minsim, self.k = incremental_mindist(self.trajsdata[episode], self.split_point, 1, self.k_dict, self.clusters_T, episode)
        else:
            self.minsim, self.k = incremental_mindist(self.trajsdata[episode], self.split_point, 1, self.k_dict, self.clusters_E, episode)
        self.next_minsim = self.minsim
        center_points = self.clusters_T[self.k][2]
        if label == 'E':
            center_points = self.clusters_E[self.k][2]
        self.overall_sim = traj2trajIED(self.trajsdata[episode].points, center_points) 
        self.split_overdist = self.minsim
        
        observation = np.array([self.overall_sim, self.minsim, self.overall_sim*10, 2 / self.length,
                                (self.length - 1) / self.length]).reshape(1, -1)
       
        self.split = []  
        self.subtrajindex = []  
        self.subtraj = []  
        return observation, self.length

    def step(self, episode, action, index, label='T'):
        if action == 0:
            if index + 1 != self.length: 
                if label == 'T':
                    self.next_minsim, self.k = incremental_mindist(self.trajsdata[episode],self.split_point, index+1, self.k_dict, self.clusters_T, episode)
                else:
                    self.next_minsim, self.k = incremental_mindist(self.trajsdata[episode], self.split_point, index + 1,
                                                              self.k_dict, self.clusters_E, episode)
                if self.split_point == 0:
                    self.split_overdist = self.next_minsim / self.traj_num
                else:
                    self.split_overdist = (self.overall_sim * self.traj_num + self.next_minsim) / (self.traj_num + 1)
            else: 
                subtraj_points = self.trajsdata[episode].points[self.split_point: index+1]
                size = len(subtraj_points)
                ts, te = subtraj_points[0].t, subtraj_points[-1].t
                subtraj = Traj(subtraj_points,size,ts,te,self.trajsdata[episode].traj_id)
                self.subtrajindex.append([self.split_point, index])
                self.split.append(index)
                self.traj_num += 1
                if label == 'T':
                    self.clusters_T[self.k][1].append(subtraj)
                    if self.next_minsim != 1e10:
                        self.clusters_T[self.k][0].append(self.next_minsim)
                        add2clusdict(subtraj.points, self.clusters_T, self.k)
                if label == 'E':
                    self.clusters_E[self.k][1].append(subtraj)
                    if self.next_minsim != 1e10:
                        self.clusters_E[self.k][0].append(self.next_minsim) 
                        add2clusdict(subtraj.points, self.clusters_E, self.k)
                self.overall_sim = self.split_overdist
           
            observation = np.array([self.overall_sim, self.split_overdist, self.overall_sim*10,
                                    (index - self.split_point + 2) / self.length,
                                    (self.length - (index + 1)) / self.length]).reshape(1, -1)
            reward = 0
            return observation, reward

        if action == 1:
            self.split.append(index)
            self.minsim = self.next_minsim
            subtraj_points = self.trajsdata[episode].points[self.split_point: index + 1]
            size = len(subtraj_points)
            ts = subtraj_points[0].t
            te = subtraj_points[-1].t
            subtraj = Traj(subtraj_points, size, ts, te, self.trajsdata[episode].traj_id)
            if label == 'T':
                self.clusters_T[self.k][1].append(subtraj)
                if self.next_minsim != 1e10:
                    self.clusters_T[self.k][0].append(self.next_minsim)  
                    add2clusdict(subtraj.points, self.clusters_T,self.k)
            if label == 'E':
                self.clusters_E[self.k][1].append(subtraj)
                if self.next_minsim !=  1e10:
                    self.clusters_E[self.k][0].append(self.next_minsim)    
                    add2clusdict(subtraj.points,self.clusters_E,self.k)
            last_overall_sim = self.overall_sim
            if self.split_point != 0:
                self.traj_num += 1
            self.overall_sim = self.split_overdist
            self.split_point = index
            if index + 1 != self.length:
                if label == 'T':
                    self.next_minsim, self.k = incremental_mindist(self.trajsdata[episode],self.split_point,index+1,self.k_dict,self.clusters_T, episode)
                else:
                    self.next_minsim, self.k = incremental_mindist(self.trajsdata[episode], self.split_point, index + 1,
                                                              self.k_dict, self.clusters_E, episode)
                self.split_overdist = (self.overall_sim * self.traj_num + self.next_minsim) / (self.traj_num + 1)
           
            observation = np.array([self.overall_sim, self.split_overdist, self.overall_sim*10,
                                    (index - self.split_point + 2) / self.length,
                                    (self.length - (index + 1)) / self.length]).reshape(1, -1)
    
            reward = last_overall_sim - self.overall_sim
            return observation, reward
    
    def output(self, label ='T'):
        if label == 'T':
            return [self.overall_sim, self.RW, self.clusters_T]
        if label == 'E':
            return [self.overall_sim, self.RW, self.clusters_E]
    
    def update_cluster(self, label='T'): 
        if label == 'T': 
            self.basesim_T, self.clusters_T = update_centers(self.clusters_T,3, 0.095)
            for i in self.clusters_T.keys():
                self.clusters_T[i][0] = []
                self.clusters_T[i][1] = []
                self.clusters_T[i][3] = defaultdict(list)
        if label == 'E':  
            self.basesim_E, self.clusters_E = update_centers(self.clusters_E, 3, 0.095)
            for i in self.clusters_E.keys():
                self.clusters_E[i][0] = []
                self.clusters_E[i][1] = []
                self.clusters_E[i][3] = defaultdict(list)
        