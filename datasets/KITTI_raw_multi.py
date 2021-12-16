from __future__ import absolute_import, division, print_function
from os import error

import os.path
import torch
import torch.utils.data as data
import numpy as np
import glob
import copy

from torchvision import transforms as vision_transforms
from .list_utils import kitti_scale_intrinsic, list_chunks, read_calib_into_dict, read_image_as_byte, Normalize, load_as_float


class KITTI_Raw_Multi(data.Dataset):
    def __init__(self,
                args,
                images_root = None,
                flip_augmentations=True,
                preprocessing_crop=True,
                resolution=[256, 832], 
                num_examples=-1,
                index_file=None):
            
        self._args = args
        self._seq_dim = args.sequence_length  # default = 2
        self._seq_lists_l = [[]] * self._seq_dim
        # [[], [], [] ,[]]

        self._flip_augmentations = flip_augmentations
        self._preprocessing_crop = preprocessing_crop
        self._resolution = resolution

        self.seq_num = 0

        ## loading index file
        # return this directory name
        path_dir = os.path.dirname(os.path.realpath(__file__)) 
        path_index_file = os.path.join(path_dir, index_file)
        
        # os.path.exists 判断是否存在此路径
        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        # 读入index文件夹
        index_file = open(path_index_file, 'r')

        ## load image
        if not os.path.exists(images_root):
            raise ValueError("Image directory '%s' not found!")
        
        scene_list = [line.rstrip() for line in index_file.readlines()]
        view1 = 'image_02'
        view2 = 'image_03'
        ext = '.png'

        ## 处理图片序列

        for scene in scene_list:
            date = scene[:10]
            # 读入left image
            img_dir = os.path.join(images_root, date, scene, view1, 'data')
            img_list = sorted(glob.glob(img_dir+ '/*' + ext))

            # 0 - 3    
            for ss in range(self._seq_dim):
                # 0, 1, 2, 3开始 seq_dim = 4 
                seqs = list_chunks(img_list[ss:], self._seq_dim)
                # print(len(seqs))
                
                for seq in seqs:
                    # [0]
                    last_img = seq[-1]
                    
                    curridx = os.path.basename(last_img)[:-4]
                    nextidx = '{:010d}'.format(int(curridx)+1)

                    # 加入reference image
                    # KITTI flow: first img -> second img
                    # disp2: disp1 + flow = disp2
                    seq.append(last_img.replace(curridx, nextidx))
                
                # 最后会多出来一张图片，但是没有这个路径
                seqs.remove(seqs[-1])
        
                self._seq_lists_l[ss] = self._seq_lists_l[ss] + seqs
        
        min_num_examples = min([len(item) for item in self._seq_lists_l])
        if num_examples > 0:
            for ii in range(self._seq_dim):
                self._seq_lists_l[ii] = self._seq_lists_l[ii][:num_examples]
        else:
            for ii in range(self._seq_dim):
                self._seq_lists_l[ii] = self._seq_lists_l[ii][:min_num_examples]
        
        self._size = [len(seq) for seq in self._seq_lists_l]
        
        ## right images
        self._seq_lists_r = copy.deepcopy(self._seq_lists_l)
        for ii in range(len(self._seq_lists_r)):
            for jj in range(len(self._seq_lists_r[ii])):
                for kk in range(len(self._seq_lists_r[ii][jj])):
                    self._seq_lists_r[ii][jj][kk] = self._seq_lists_r[ii][jj][kk].replace(view1, view2)

        ## loading calibration 
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

        ## img.tobytes()  将图片转化成内存中的存储格式
        # torch.BytesStorage.frombuffer(img.tobytes() )  将字节以流的形式输入，转化成一维的张量
        # 对张量进行reshape
        # 对张量进行permute（2,0,1）
        # 将当前张量的每个元素除以255  [0, 1]
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self._to_tensor = vision_transforms.Compose([
            vision_transforms.Resize(self._resolution),
            normalize
        ])
    
    
    def __getitem__(self, index):
        
        self._seq_num = int(torch.randint(0, self._seq_dim, (1,)))
        self._seq_num = int(0)
        # 起始状态成为random了[0, 3]
        # print(self._seq_lists_l)
        index = index % self._size[self._seq_num]
        seq_list_l = self._seq_lists_l[self._seq_num][index]
        seq_list_r = self._seq_lists_r[self._seq_num][index]
        
        # read images
        # print(seq_list_r)
        img_list_l_np = [load_as_float(img) for img in seq_list_l]
        img_list_r_np = [load_as_float(img) for img in seq_list_r]
        
        # example filename
        im_l1_filename = seq_list_l[1]
        basename = os.path.basename(im_l1_filename)[:6]
        dirname = os.path.dirname(im_l1_filename)[-51:]
        datename = dirname[:10]
        # like format : 2011_10_03_48_sync
        scenename = dirname[11:32] + '_' + im_l1_filename[-8:-4]

        k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()

        # input size
        h_orig, w_orig, _ = img_list_l_np[0].shape
        input_im_size = self._resolution

        # resize the intrinsic
        sy = input_im_size[0] / h_orig
        sx = input_im_size[1] / w_orig
        k_l1, k_r1 = kitti_scale_intrinsic(k_l1, k_r1, sy, sx)

        # to tensors [t, c, h, w]
        tensors = []
        for img in img_list_l_np:
            img = np.transpose(img, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(img).float()/255)
        
        imgs_l_tensor = torch.stack([self._to_tensor(img) for img in tensors], dim=0)

        tensors = []
        for img in img_list_r_np:
            img = np.transpose(img, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(img).float()/255)

        imgs_r_tensor = torch.stack([self._to_tensor(img) for img in tensors], dim=0)


        common_dict = {
            "index": index,
            "basename": scenename,
            "datename": datename,
            "input_size": input_im_size
        }

        # random flip
        if self._flip_augmentations is True and torch.rand(1) > 0.5:
            _, _, _, ww = imgs_l_tensor.size()
            imgs_l_tensor_flip = torch.flip(imgs_l_tensor, dims=[3])
            imgs_r_tensor_flip = torch.flip(imgs_r_tensor, dims=[3])

            k_l1[0, 2] = ww - k_l1[0, 2]
            k_r1[0, 2] = ww - k_r1[0, 2]

            example_dict = {
                "input_left": imgs_r_tensor_flip,
                "input_right": imgs_l_tensor_flip,
                "input_k_l": k_r1,
                "input_k_r": k_l1
            }
            example_dict.update(common_dict)

        else:
            example_dict = {
                "input_left": imgs_l_tensor,
                "input_right": imgs_r_tensor,
                "input_k_l": k_l1,
                "input_k_r": k_r1
            }
            example_dict.update(common_dict)

        return example_dict
    
    def __len__(self):
        return self._size[0]


class KITTI_Raw_Multi_KittiSPlit_Train(KITTI_Raw_Multi):
    """
    KITTI train split 
    """
    def __init__(self, args, images_root=None, flip_augmentations=True, preprocessing_crop=True, resolution=[256, 832], num_examples=-1, index_file=None):
        super(KITTI_Raw_Multi_KittiSPlit_Train, self).__init__(args, 
                        images_root=images_root, 
                        flip_augmentations=flip_augmentations, 
                        preprocessing_crop=preprocessing_crop, 
                        resolution=resolution, 
                        num_examples=num_examples, 
                        index_file="index_txt/kitti_train_scenes.txt")

import random
from path import Path

def crawl_folders(folders_list, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        for folder in folders_list:
            intrinsics = np.genfromtxt(folder/'cam.txt', delimiter=',').astype(np.float32).reshape((3, 3))
            imgs = sorted(folder.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in range(-demi_length, demi_length + 1):
                    if j != 0:
                        sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        return sequence_set


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.samples = crawl_folders(self.scenes, sequence_length)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])

        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        
        common_dict = {
            "input_left": tgt_img,
            "input_ref": ref_imgs[1],
            "input_k_l": intrinsics,
            "input_k_inv": np.linalg.inv(intrinsics)
        }
        return common_dict

    def __len__(self):
        return len(self.samples)


""" if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-length', type=int, default=2)
    args = parser.parse_args()

    KITTI_load = KITTI_Raw_Multi_KittiSPlit_Train(args, images_root="/home/cshi/dataset/KITTI")
    print(KITTI_load[1]['input_size']) """
