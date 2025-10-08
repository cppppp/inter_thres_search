from ntpath import join
import os
import numpy as np
import time
from scipy.fft import ifftn
import torch
from torch import optim
from data_loader1 import get_loader
torch.backends.cudnn.deterministic = True
from tqdm import tqdm
import torch.distributed as dist
import gc

from matplotlib import pyplot as plt

output=[0,0,0,0,0,0]

class Solver(object):
    def __init__(self, config, gpus):
        self.device = torch.device('cpu')
        self.batch_size = config.batch_size
        self.rdo_param = config.rdo_param

    def calculate_loss(self, images, rdtcost,last_thres, new_thres, rdo_gamma):
        soft = images['gt'][:,-5:]
        
        for j in range(images['gt'].size(0)):

            last_best_rd = 1000000000000000
            selected_last = [True, False, False, False, False, False]
            if soft[j][0]>last_thres[0] and images['gt'][j][1]>0.00001:
                selected_last[1] = True
            else:
                one_valid = False
                valid = [False, False, False, False]
                for k in range(4):
                    selected_last[k+2] = (soft[j][k+1]>last_thres[k+1] and images['gt'][j][k+2]>0.00001)
                    valid[k] = (images['gt'][j][k+2]>0.00001)
                    if selected_last[k+2]:
                        one_valid = True
                if not one_valid:
                    largest_pred = -1.
                    pos = 0
                    for k in range(4):
                        if valid[k] and soft[j][k+1]>largest_pred:
                            pos = k
                            largest_pred = soft[j][k+1]
                    selected_last[pos+2] = True

            for k in range(6):
                if selected_last[k] and images['gt'][j][k]>0.00001:
                    last_best_rd = min(last_best_rd, images['gt'][j][k])
            
            new_best_rd = 1000000000000000
            selected_new = [True, False, False, False, False, False]
            if soft[j][0]>new_thres[0] and images['gt'][j][1]>0.00001:
                selected_new[1] = True
            else:
                one_valid = False
                valid = [False, False, False, False]
                for k in range(4):
                    selected_new[k+2] = (soft[j][k+1]>new_thres[k+1] and images['gt'][j][k+2]>0.00001)
                    valid[k] = (images['gt'][j][k+2]>0.00001)
                    if selected_new[k+2]:
                        one_valid = True
                if not one_valid:
                    largest_pred = -1.
                    pos = 0
                    for k in range(4):
                        if valid[k] and soft[j][k+1]>largest_pred:
                            pos = k
                            largest_pred = soft[j][k+1]
                    selected_new[pos+2] = True

            for k in range(6):
                if selected_new[k] and images['gt'][j][k]>0.00001:
                    new_best_rd= min(new_best_rd, images['gt'][j][k])
            
            
            for k in range(5):
                if images['gt'][j][k+1]==0: 
                    continue
                if new_thres[k]>last_thres[k] and selected_new[k+1]==False and selected_last[k+1]==True and \
                   soft[j][k]>last_thres[k] and soft[j][k]<=new_thres[k]:
                    delta_rd = new_best_rd - images['gt'][j][k+1]
                    if delta_rd < 0:
                        delta_rd = 0
                    
                    delta_time = images['gt'][j][6+k+1]
                    if k==0:
                        for z in range(1,5):
                            if selected_new[k+1]==True:
                                delta_time -= images['gt'][j][6+z+1]
                        
                    rdtcost[k] += (rdo_gamma * delta_rd/100 - self.rdo_param * delta_time)

                elif new_thres[k]<last_thres[k] and selected_new[k+1]==True and selected_last[k+1]==False and \
                   soft[j][k]>new_thres[k] and soft[j][k]<=last_thres[k]:
                    delta_rd = images['gt'][j][k+1] - last_best_rd
                    if delta_rd > 0:
                        delta_rd = 0
                    
                    delta_time = images['gt'][j][6+k+1]
                    if k==0:
                        for z in range(1,5):
                            if selected_last[k+1]==True:
                                delta_time -= images['gt'][j][6+z+1]
    
                    rdtcost[k] += (rdo_gamma * delta_rd/100 + self.rdo_param * delta_time)
        
        return rdtcost

    def run(self, cuSize_idx, images,rdtcost,last_thres,new_thres):
        
        if cuSize_idx == 0:
            rdo_gamma = 0.2534
        elif cuSize_idx == 1:
            rdo_gamma = 0.0524
        elif cuSize_idx == 2:
            rdo_gamma = 0.0839
        elif cuSize_idx == 3:
            rdo_gamma = 0.0351
        elif cuSize_idx == 4:
            rdo_gamma = 0.0203
        elif cuSize_idx == 5:
            rdo_gamma = 0.0115
        elif cuSize_idx == 6:
            rdo_gamma = 0.0241
        elif cuSize_idx == 7:
            rdo_gamma = 0.0048
        elif cuSize_idx == 8:
            rdo_gamma = 0.0024
        elif cuSize_idx == 9:
            rdo_gamma = 0.0287
        elif cuSize_idx == 10:
            rdo_gamma = 0.0303
        elif cuSize_idx == 11:
            rdo_gamma = 0.0060
        else:
            print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        rdtcost_list = self.calculate_loss(images, rdtcost,last_thres, new_thres, rdo_gamma)
        return rdtcost_list
        

    def validate(self,valid_loader,cuSize_idx,thres=0.1):

        last_thres = [0.5, 0.15, 0.15, 0.15, 0.15]
        new_thres = [0.45, 0.12, 0.12, 0.12, 0.12]
        left_end = [-0.1, -0.1, -0.1, -0.1, -0.1]
        right_end = [0.9, 0.45, 0.45, 0.45, 0.45]
        current_range = [[-0.1,0.9], [-0.1,0.45], [-0.1,0.45], [-0.1,0.45], [-0.1,0.45]]
        step_len = [0.05, 0.03, 0.03, 0.03, 0.03]

        for iter in range(10):
            rdtcost = [0.,0.,0.,0.,0.]
            for i, images in enumerate(valid_loader):
                rdtcost = self.run(cuSize_idx,images,rdtcost,last_thres,new_thres)
            for k in range(5):
                if new_thres[k] < last_thres[k] and rdtcost[k] > 0:
                    current_range[k][0]=new_thres[k]
                    if current_range[k][1] == right_end[k]: 
                        new_thres[k] = last_thres[k] + step_len[k]
                elif new_thres[k] >= last_thres[k] and rdtcost[k] > 0: 
                    current_range[k][1]=new_thres[k]
                elif new_thres[k] < last_thres[k] and rdtcost[k] < 0: 
                    current_range[k][1] = last_thres[k]
                    if current_range[k][0] == left_end[k]: 
                        new_thres[k] = new_thres[k] - step_len[k]
                        last_thres[k] = last_thres[k] - step_len[k]
                elif new_thres[k] >= last_thres[k] and rdtcost[k] < 0: 
                    current_range[k][0] = last_thres[k]
                    if current_range[k][1] == right_end[k]:
                        new_thres[k] = new_thres[k] + step_len[k]
                        last_thres[k] = last_thres[k] + step_len[k]
                
                if current_range[k][0] != left_end[k] and current_range[k][1] != right_end[k]:    
                    last_thres[k] = (current_range[k][0] + current_range[k][1] * 2) / 3
                    new_thres[k] = (current_range[k][0] * 2 + current_range[k][1]) / 3
                
            print(cuSize_idx, iter, current_range)
            if iter==9:
                print("final", [(current_range[k][0]+current_range[k][1]) / 2 for k in range(5)])

    def train(self):
        print("start training")
        train_loader = []
        valid_loader = []
        for cuSize_idx in [[64, 64], [32, 64], [32, 32], [16, 64], [16, 32], [16, 16], [8, 64], [8, 16], [8, 8], \
               [4, 64], [4, 32], [4, 16]]:
            valid_loader.append(get_loader(cuSize=cuSize_idx, batch_size=self.batch_size, num_workers=2, mode='valid'))
            
        for cuSize_idx in range(3):
            self.validate(valid_loader[cuSize_idx],cuSize_idx)
                