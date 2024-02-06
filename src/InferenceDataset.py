import torch.utils.data
import rasterio
import torch
import numpy as np
import matplotlib.pyplot as plt



def readimg(file):
    with rasterio.open(file) as src:

        b1 = src.read(1)/10000
        b2 = src.read(2)/10000
        b3 = src.read(3)/10000
        b4 = src.read(4)/10000
        b5 = src.read(5)/10000
        b6 = src.read(6)/10000



        srcnorm = np.dstack((b1, b2, b3, b4, b5,b6))
        srcnorm = np.transpose(srcnorm,(2,0,1))
        srcnorm =torch.from_numpy(srcnorm)
        return srcnorm



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataW, dataS, window, stride):
        self.dataW = dataW
        self.dataS = dataS

        self.window = window
        self.stride = stride
        self.starting_points = list()

        x = readimg(self.dataW)

        # x=self.data
        h, w = x.shape[1:3]
        w_h, w_w =self.window
        s_h, s_w = self.stride

        # Generate a list of starting points a.k.a top left of window
        self.starting_points = [(x, y)  for x in set( list(range(0, h - w_h, s_h)) + [h - w_h] )
                                   for y in set( list(range(0, w - w_w, s_w)) + [w - w_w] )]

    def __getitem__(self, index):

        xW = readimg(self.dataW)
        xS = readimg(self.dataS)
        x = np.concatenate((xS, xW), axis=0)
        # x=self.data #For only one season
        w_h, w_w =self.window
        x = x[ :, self.starting_points[index][0]:self.starting_points[index][0] + w_h, self.starting_points[index][1]:self.starting_points[index][1] + w_w]
        x=torch.from_numpy(x)
        return x.float(), self.starting_points[index]

    def __len__(self):
        return len(self.starting_points)


