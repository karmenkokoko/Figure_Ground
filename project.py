from __future__ import absolute_import, division, print_function

import numpy as np

import os
import torch
# from torch._C import float32
import torch.nn as nn
import torch.nn.functional as F
import datasets
from torch.autograd import Variable
from occ import occulsion_mask
from utils.interpolation import *

pixel_coords = None

def disp_to_depth(disp, min_depth, max_depth):
    """
    (min_depth -- max_depth) => (0.1 - 100)
    Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """

    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def disp2depth_kitti(pred_disp, k_value, depth_clamp=True):
    """
    k_value = k[:, 0, 0]
    """
    pred_depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_disp + 1e-4)
    if depth_clamp:
        pred_depth = torch.clamp(pred_depth, 1e-3, 80)

    return pred_depth


def depth2disp_kitti(pred_depth, k_value, depth_clamp=True):

    if depth_clamp:
        pred_depth = torch.clamp(pred_depth, 1e-3, 80)
    pred_disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_depth)

    return pred_disp


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rotMat = xmat.bmm(ymat).bmm(zmat)
    return rotMat


def pose_vec2mat(vec):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]

    rot_mat = euler2mat(rot)  # [B, 3, 3]

    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1,h,w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) != h or pixel_coords.size(3) != w:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).contiguous().view(b, 3, -1)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)


def pose2rigid_flow(depth, pose_mat, intrinsics, intrinsics_inv):
    """
    compute rigid flow
    """
    b, _, h, w = depth.size()
    # index pts for w
    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth).squeeze()  # [bs, H, W]
    # index pts for h
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth).squeeze()  # [bs, H, W]

    ## output the reproject result
    cam_coordinate = pixel2cam(depth.squeeze(1), intrinsics_inv)
    ## camera egomotion
    cam_ego = intrinsics.bmm(pose_mat)

    src_pixel_coords = cam2pixel(cam_coordinate, cam_ego[:,:,:3], cam_ego[:,:,-1:], padding_mode=None)  # [B,H,W,2]

    X = (w-1)*(src_pixel_coords[:,:,:,0]/2.0 + 0.5) - grid_x
    Y = (h-1)*(src_pixel_coords[:,:,:,1]/2.0 + 0.5) - grid_y

    return torch.stack((X,Y), dim=1)



def flow_warp(img, flow):
    """
    we will warp the 3d coordinate
    """
    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    grid_tf = torch.stack((X,Y), dim=3)
    img_tf = F.grid_sample(img, grid_tf, align_corners=True)

    return img_tf


def rigid_background(depth, pose_mat, intrinsics_inv):
    """
    3d motion of background
    """
    bs, _, h, w = depth.size()

    cam_coords= pixel2cam(depth.squeeze(1), intrinsics_inv)  # [B,3,H,W]
    
    # Get projection matrix for tgt camera frame to source CAMERA frame
    # proj_cam_to_src_cam = intrinsics.bmm(pose_mat)  # [B, 3, 4]
    proj_cam_to_src_cam = pose_mat

    b, _, h, w = cam_coords.size()

    ## T (t->t+1) * coordinate
    cam_coordinate_flatten = cam_coords.view(b, 3, -1)
    source_cam_coord = (proj_cam_to_src_cam[:, :, :3]).bmm(cam_coordinate_flatten)
    source_cam_coord = source_cam_coord + proj_cam_to_src_cam[:, :, -1:]
    source_cam_coord = source_cam_coord.view(b, 3, h, w)

    ## rigid 3d motion
    motion_background = source_cam_coord - cam_coords

    return motion_background


def dynamic_motion(occlusion_mask, intrinsics_inv, depth, depth_ref, flow, pose_mat, motion_background):
    """
    3D motion in foreground
    """
    b, _, h, w = depth.size()

    ## t+1 image pixel to camera coordinate
    ref_cam_coordinate = pixel2cam(depth_ref.squeeze(1), intrinsics_inv)

    # t image  pixel to camera coordinate
    cam_coordinate = pixel2cam(depth.squeeze(1), intrinsics_inv)
    
    ## t+1 camera to t camera
    rot_matrices = torch.inverse(pose_mat[:,:,:3])
    tr_vectors = -rot_matrices @ pose_mat[:,:,-1:]
    pose_t_t1 = torch.cat((rot_matrices, tr_vectors), axis=-1)

    # multiply R,T (t+1 => t)
    cam_coordinate_flatten = ref_cam_coordinate.view(b, 3, -1)
    cam_coord = (pose_t_t1[:, :, :3]).bmm(cam_coordinate_flatten)
    cam_coord = cam_coord + pose_t_t1[:, :, -1:]
    cam_coord_t1 = cam_coord.view(b, 3, h, w)

    ## flow warp (t -> t+1)
    ## think the 3d coordinate is the img
    # print(torch.any(torch.isnan(flow)))
    camt = flow_warp(cam_coord_t1, flow)

    # print(torch.any(torch.isnan(camt)))
    ## compute the scene flow
    ## t -> t+1 in 3d coordinate 
    ## scene flow may has nan value
    ## nan value is from the camt
    camt[camt != camt] = cam_coordinate[camt != camt]
    scene_flow = camt - cam_coordinate

    # print(torch.any(torch.isnan(scene_flow)))

    motion = scene_flow - motion_background
    
    return (occlusion_mask[:, 0, :, :]).unsqueeze(1) * motion, scene_flow


def motion_soft_mask(dynamic_motion, alpha):
    """
    compute the dynamic mask
    """
    scaled_motion = -(alpha * torch.linalg.norm(dynamic_motion, dim=1)).unsqueeze(1)
    exp_motion = torch.exp(scaled_motion)

    return (1 - exp_motion)


def get_motion_mask(tgt_img, flow_bwd, flow_fwd, depth, depth_ref, intrinsic_inv, pose, alpha, test=False):
    """
    get the mask of different scale
    backward flow : ref -> tgt
    forward flow : tgt -> ref
    """
    motion_masks = []
    scene_flows = []
    if test:
        b, _, h, w = flow_fwd.size()
        # t->t+1
        depth_scale = depth
        depth_scale_ref = depth_ref
        # compute occ mask
        occ_mask = occulsion_mask(tgt_img, flow_fwd)
        # whether RT need to be scale ?
        motion_bgd = rigid_background(depth_scale, pose, intrinsic_inv)
        motion_fgd, scene_flow = dynamic_motion(occ_mask, intrinsic_inv, depth_scale, depth_scale_ref, flow_fwd, pose, motion_bgd)
        motion_m = motion_soft_mask(motion_fgd, alpha)
        
        return motion_m, scene_flow
    

    for idx, flow_b in enumerate(flow_bwd):
        b, _, h, w = flow_b.size()
        # t -> t+1 flow
        flow_f = flow_fwd[idx]
        depth_scale = depth[idx]
        depth_scale_ref = depth_ref[idx]
        downscale = tgt_img.size(2) / h

        scaled_img = F.adaptive_avg_pool2d(tgt_img, (h, w))
        # scaled_img_ref = F.adaptive_avg_pool2d(ref_img, (h, w))

        intrinsic_inv_scaled = torch.cat((intrinsic_inv[:, 0:2]/downscale, intrinsic_inv[:, 2:]), dim=1)
        
        # compute occ mask
        occ_mask = occulsion_mask(scaled_img, flow_f)
        
        motion_bgd = rigid_background(depth_scale, pose, intrinsic_inv_scaled)
        motion_fgd, scene_flow = dynamic_motion(occ_mask, intrinsic_inv_scaled, depth_scale, depth_scale_ref, flow_f, pose, motion_bgd)
        motion_scale = motion_soft_mask(motion_fgd, alpha)
        motion_masks.append(motion_scale)
        scene_flows.append(scene_flow)

    return motion_masks, scene_flows


def inverse_warp(img, depth, pose, intrinsics, intrinsics_inv):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics_inv)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode=None)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, align_corners=True)

    return projected_img


def spatial_normalize(disp):
    _mean = disp.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    disp = disp / _mean
    return disp





    