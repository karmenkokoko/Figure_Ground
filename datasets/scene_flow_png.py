import os
from posixpath import basename
import cv2
import glob
import torch
import random
import einops
import numpy as np
from torch.utils.data import Dataset
# from cvbase.optflow.visualize import flow2rgb


def readFlow(sample_dir, resolution):
    flow = np.load(sample_dir)
    h, w, _ = np.shape(flow)
    flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] = flow[:, :, 0] * resolution[1] / w
    flow[:, :, 1] = flow[:, :, 1] * resolution[0] / h
    return einops.rearrange(flow, 'h w c -> c h w')


def readRGB(sample_dir, resolution):
    rgb = cv2.imread(sample_dir)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # convert to (-1, 1)
    rgb = ((rgb / 255.0) - 0.5) * 2.0
    rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(rgb, -1., 1.)
    return einops.rearrange(rgb, 'h w c -> c h w')


class SceneFlow(Dataset):
    def __init__(self, data_dir, resolution):
        self.eval = eval
        self.data_dir = data_dir
        self.resolution = resolution
    
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        
        folders = [os.path.basename(x) for x in glob.glob(os.path.join(self.data_dir, '*'))]
        rand_num = int(torch.randint(0, len(folders), (1,)))

        select_fold = str(folders[rand_num])
        flow_name = '/' + select_fold + '_' + str("{:04d}".format(idx)) + '_sf.npy'
        flow_path = os.path.join(self.data_dir, select_fold+flow_name)
        flow = readFlow(flow_path, self.resolution)

        sample = {'image': flow}

        return sample


""" if __name__ == "__main__":
    colorwheel = make_color_wheel()
    # cv2.imwrite('./test.png', colorwheel.astype(np.uint8))
    print(colorwheel.shape) """
    
