import argparse
import os
from solver import Solver
from data_loader1 import get_loader
from torch.backends import cudnn
import random
import time
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICE'] = '0'

def main(config):
    epoch = 5
    decay_ratio = 0.2
    decay_epoch = int(epoch*decay_ratio)

    config.num_epochs = epoch
    config.lr = 0.0001  #0.0001
    config.num_epochs_decay = decay_epoch
    solver = Solver(config, [0])
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rdo_param', type=int, default=500)
    config = parser.parse_args()
    main(config)
