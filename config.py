from __future__ import absolute_import
import os
from cv2 import transform
import torch
import glob as gb
import numpy as np
import json
import datasets
from datetime import datetime
import shutil
import custom_transforms



def setup_path(args):

    num_slots = args.num_slots
    iters = args.num_iterations
    batch_size = args.batch_size
    resolution = args.resolution
    # log output parameter
    verbose = args.verbose if args.verbose else 'none'
    inference = args.inference

    # make all the essential folders, e.g. models, logs, results, etc.
    global dt_string, logPath, modelPath, resultsPath
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H")

    os.makedirs('./slot_logs/', exist_ok=True)
    os.makedirs('./slot_modelsave/', exist_ok=True)
    os.makedirs('./slot_results/', exist_ok=True)

    logPath = os.path.join('./slot_logs/', f'{dt_string}-KITTI-'
                                           f'slots_{num_slots}-VGGNet_D256-'
                                           f'iter_{iters}-bs_{batch_size}-res_{resolution[0]}x{resolution[1]}-{verbose}')

    modelPath = os.path.join('./slot_modelsave/', f'{dt_string}-KITTI_'
                                               f'slots_{num_slots}-VGGNet_D256-'
                                               f'iter_{iters}-bs_{batch_size}-res_{resolution[0]}x{resolution[1]}-{verbose}')

    if inference:
        resultsPath = os.path.join('./slot_results/', args.resume_path.split('/')[-1])
        os.makedirs(resultsPath, exist_ok=True)
    else:
        os.makedirs(logPath, exist_ok=True)
        os.makedirs(modelPath, exist_ok=True)
        resultsPath = None

        # save all the experiment settings.
        with open('{}/running_command.txt'.format(modelPath), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return [logPath, modelPath, resultsPath]


def setup_geometric_path(args):

    epoches = args.epoches
    batch_size = args.batch_size
    resolution = args.resolution
    # log output parameter
    verbose = args.verbose if args.verbose else 'none'
    inference = args.inference

    # make all the essential folders, e.g. models, logs, results, etc.
    global dt_string, logPath, modelPath, resultsPath
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H")

    os.makedirs('./geometric_logs/', exist_ok=True)
    os.makedirs('./geometric_modelsave/', exist_ok=True)
    os.makedirs('./geometric_results/', exist_ok=True)

    logPath = os.path.join('./geometric_logs/', f'{dt_string}-KITTI-'
                                           f'epoch_{epoches}-bs_{batch_size}-res_{resolution[0]}x{resolution[1]}-{verbose}')

    modelPath = os.path.join('./geometric_modelsave/', f'{dt_string}-KITTI_'
                                               f'epoch{epoches}-bs_{batch_size}-res_{resolution[0]}x{resolution[1]}-{verbose}')

    if inference:
        resultsPath = os.path.join('./geometric_results/', args.resume_path.split('/')[-1])
        os.makedirs(resultsPath, exist_ok=True)
    else:
        os.makedirs(logPath, exist_ok=True)
        os.makedirs(modelPath, exist_ok=True)
        resultsPath = None

        # save all the experiment settings.
        with open('{}/running_command.txt'.format(modelPath), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return [logPath, modelPath, resultsPath]


def save_checkpoint(save_path, dispnet_state, posenet_state, flownet_state, slot_state, optimizer_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'posenet', 'flownet', 'slot_attention', 'optimizer']
    states = [dispnet_state, posenet_state, flownet_state, slot_state, optimizer_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path + '/{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path + '/{}_{}'.format(prefix,filename), save_path + '/{}_model_best.pth.tar'.format(prefix))


def setup_dataset(args):
    resolution = args.resolution
    basepath = "/home/cshi/dataset/KITTI/slot_flow_npy/"
    img_dir = "/home/cshi/dataset/KITTI/slot_sceneflow"

    trndataset = datasets.SceneFlow(data_dir=basepath, resolution=resolution)

    in_out_channels = 3
    loss_scale = 1e2
    ent_scale  = 1e-2
    cons_scale = 1e-2

    return [trndataset, resolution, in_out_channels, loss_scale, ent_scale, cons_scale]


def setup_KITTI_dataset(args):
    resolution = args.resolution
    images_root="/home/cshi/dataset/KITTI"
    format_data="/home/cshi/Project/Figure_ground/format_data"
    validation_depth_root = "./format_data/"
    validation_flow_root = "/home/cshi/dataset/KITTI/scene_flow/"
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    valid_flow_transform = custom_transforms.Compose([custom_transforms.Scale(h=256, w=832),
                            custom_transforms.ArrayToTensor(), normalize])

    trndataset = datasets.KITTI_train(format_data, seed=0, sequence_length=3, transform=train_transform)
    

    valdataset = datasets.Validation_Set(validation_depth_root, transform=valid_transform)

    val_flow_set = datasets.Validation_Flow(validation_flow_root, sequence_length=2, transform=valid_flow_transform)

    val_mask_set = datasets.Validation_mask(validation_flow_root, sequence_length=2, transform=valid_flow_transform)


    return [trndataset, valdataset, val_flow_set, val_mask_set, resolution]

