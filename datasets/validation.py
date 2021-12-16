from torch.nn.functional import normalize
import torch.utils.data as data
import numpy as np
from path import Path
from PIL import Image
from .list_utils import *
import os
import torch
from torchvision import transforms as vision_transforms


def crawl_folders(folders_list):
        imgs = []
        depth = []
        for folder in folders_list:
            current_imgs = sorted(folder.files('*.jpg'))
            current_depth = []
            for img in current_imgs:
                d = img.dirname()/(img.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                depth.append(d)
            imgs.extend(current_imgs)
            depth.extend(current_depth)
        return imgs, depth

def crawl_folders_seq(folders_list, sequence_length):
        imgs1 = []
        imgs2 = []
        depth = []
        for folder in folders_list:
            current_imgs = sorted(folder.files('*.jpg'))
            current_imgs1 = current_imgs[:-1]
            current_imgs2 = current_imgs[1:]
            current_depth = []
            for (img1,img2) in zip(current_imgs1, current_imgs2):
                d = img1.dirname()/(img1.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                depth.append(d)
            imgs1.extend(current_imgs1)
            imgs2.extend(current_imgs2)
            depth.extend(current_depth)
        return imgs1, imgs2, depth


def load_as_float(path):
    return np.array(Image.open(path)).astype(np.float32)


def get_intrinsics(calib_file, cid='02'):
    #print(zoom_x, zoom_y)
    filedata = read_raw_calib_file(calib_file)
    P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
    return P_rect[:,:3]


def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                    data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                    pass
    return data


class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.imgs, self.depth = crawl_folders(self.scenes)
        self.transform = transform


    def __getitem__(self, index):
        img = load_as_float(self.imgs[index])
        depth = np.load(self.depth[index]).astype(np.float32)

        ## add img transform
        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]
        return img, depth

    def __len__(self):
        return len(self.imgs)


class ValidationFlow(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, sequence_length, transform=None, N=200, phase='training', occ='flow_occ'):
        self.root = Path(root)
        self.sequence_length = sequence_length

        ## 200 scenes
        self.N = N
        self.transform = transform
        self.phase = phase
        seq_ids = list(range(0, int(sequence_length/2)+1))
        seq_ids.remove(0)
        self.seq_ids = [x+10 for x in seq_ids]
        self.occ = occ

    def __getitem__(self, index):
        tgt_img_path =  self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        ref_img_paths = [self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_'+str(k).zfill(2)+'.png') for k in self.seq_ids]
        gt_flow_path = self.root.joinpath('data_scene_flow', self.phase, self.occ, str(index).zfill(6)+'_10.png')
        cam_calib_path = self.root.joinpath('data_scene_flow_calib', self.phase, 'calib_cam_to_cam', str(index).zfill(6)+'.txt')
        obj_map_path = self.root.joinpath('data_scene_flow', self.phase, 'obj_map', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_imgs = [load_as_float(ref_img) for ref_img in ref_img_paths]
        if os.path.isfile(obj_map_path):
            obj_map = load_as_float(obj_map_path)
        else:
            obj_map = np.ones((tgt_img.shape[0], tgt_img.shape[1]))
        flow, valid = read_png_flow(gt_flow_path)
        gtFlow = flow.transpose(2, 0, 1)
        valid_p = valid.transpose(2, 0, 1)
        gtFlow = np.concatenate((gtFlow, valid_p), axis=0)

        intrinsics = get_intrinsics(cam_calib_path).astype('float32')
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(intrinsics))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(intrinsics)

        return tgt_img, ref_imgs[0], intrinsics, np.linalg.inv(intrinsics), gtFlow, obj_map

    def __len__(self):
        return self.N


class ValidationMask(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, sequence_length, transform=None, N=200, phase='training'):
        self.root = Path(root)
        self.sequence_length = sequence_length

        ## 200 scenes
        self.transform = transform
        self.N = N
        self.phase = phase
        seq_ids = list(range(0, int(sequence_length/2)+1))
        seq_ids.remove(0)
        self.seq_ids = [x+10 for x in seq_ids]

    def __getitem__(self, index):
        tgt_img_path =  self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        ref_img_paths = [self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_'+str(k).zfill(2)+'.png') for k in self.seq_ids]
        gt_flow_path = self.root.joinpath('data_scene_flow', self.phase, 'flow_occ', str(index).zfill(6)+'_10.png')
        cam_calib_path = self.root.joinpath('data_scene_flow_calib', self.phase, 'calib_cam_to_cam', str(index).zfill(6)+'.txt')
        obj_map_path = self.root.joinpath('data_scene_flow', self.phase, 'obj_map', str(index).zfill(6)+'_10.png')
        semantic_map_path = self.root.joinpath('semantic_labels', self.phase, 'semantic', str(index).zfill(6)+'_10.png')
        
        tgt_img = load_as_float(tgt_img_path)
        ref_imgs = [load_as_float(ref_img) for ref_img in ref_img_paths]

        obj_map = torch.LongTensor(np.array(Image.open(obj_map_path)))
        semantic_map = torch.LongTensor(np.array(Image.open(semantic_map_path)))

        flow, valid = read_png_flow(gt_flow_path)
        gtFlow = flow.transpose(2, 0, 1)
        valid_p = valid.transpose(2, 0, 1)
        gtFlow = np.concatenate((gtFlow, valid_p), axis=0)
        
        intrinsics = get_intrinsics(cam_calib_path).astype('float32')
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(intrinsics))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(intrinsics)

        return tgt_img, ref_imgs[0], intrinsics, np.linalg.inv(intrinsics), gtFlow, obj_map, semantic_map

    def __len__(self):
        return self.N