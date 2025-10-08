import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F

import json
import cv2
import struct
import glob
import time, random
from sys import getsizeof as getsize
import array

class ImageFolder1(data.Dataset):

    def __init__(self,mode,batch_size,cuSize,debug):
        self.debug=debug  #train or debug
        self.mode=mode
        self.cuSize=cuSize
        self.list=self.getlist()
        self.batch_size=batch_size        
        
    def __getitem__(self, index):
        tmp={}
        tmp['gt'] = self.list[index][0]
        return tmp

    def getlist(self):
        datalist=[]
        print("getting list")
        def gen_datalist(qp,train_or_test):
            self.yuv_path_list=[]
            cuSize_str = str(self.cuSize[0]) + "_" + str(self.cuSize[1])
            name_list=glob.glob("../collected_" + cuSize_str + '/'+str(qp)+"/*")
            random.seed(65345)

            self.yuv_path_list = np.array(name_list)

            portion = 0.5

            for i, path in enumerate(self.yuv_path_list):
                #print(path)
                with open(path,"r") as write_file:
                    cu_pic=json.load(write_file)

                for splits in cu_pic["output"]:
                    if random.random()>portion * 0.05:
                        continue
                    data_item=[]
                    gt = torch.tensor(splits)
                    data_item.append(gt)
                    datalist.append(data_item)
                            
        if self.mode=='train':
            print("error!!!!!!!!!!!!!!")
        else:
            gen_datalist(37,'test')
            gen_datalist(32,'test')
            gen_datalist(27,'test')
            gen_datalist(22,'test')

        print("getting list finished")
        print(len(datalist),self.cuSize,self.mode)
        return datalist
    def __len__(self):
        return len(self.list)-len(self.list)%self.batch_size

def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([sample[key] for sample in batch])
    return collated
    
def get_loader(cuSize,batch_size, num_workers=2, mode='train',debug='train'):
    """Builds and returns Dataloader."""
    dataset = ImageFolder1(mode=mode,batch_size=batch_size,cuSize=cuSize,debug=debug)
    if debug=='train':
        return data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    collate_fn = collate_fn)
    else:
        return data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn = collate_fn)
