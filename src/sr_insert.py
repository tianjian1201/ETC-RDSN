import torch
import numpy
import numpy as np
import pdb
import time
from tqdm import tqdm

def insert(sr, hr, lr, filename, scale=2):
    for x in range(0, numpy.shape(sr)[2], scale):
        for y in range(0, numpy.shape(sr)[3], scale):
            sr[0][0][x][y] = hr[0][0][x][y]
            sr[0][1][x][y] = hr[0][1][x][y]
            sr[0][2][x][y] = hr[0][2][x][y]



def insert_std(hr, lr, sr, scale=2):
    batch_size = 16
    for i in range(0, batch_size):
        for x in range(0, lr.shape[2]):
            for y in range(0, lr.shape[3]):
                '''r = int(hr[i][0][x*scale][y*scale]) - int(lr[i][0][x][y])
                g = int(hr[i][1][x*scale][y*scale]) - int(lr[i][1][x][y])
                b = int(hr[i][2][x*scale][y*scale]) - int(lr[i][2][x][y])'''
                sr[i][0][x*scale][y*scale] = hr[i][0][x*scale][y*scale]
                sr[i][1][x*scale][y*scale] = hr[i][1][x*scale][y*scale]
                sr[i][2][x*scale][y*scale] = hr[i][2][x*scale][y*scale] 



    
    
             


