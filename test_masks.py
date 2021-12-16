import numpy as np
import argparse
import os
import model
from tqdm import tqdm
from path import Path
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
from losses import *
from logger import AverageMeter
from PIL import Image
from scipy.ndimage.interpolation import zoom
import utils.viz as visual
import matplotlib.pyplot as plt
import config as cg


parser = argparse.ArgumentParser(description='Test IOU of Mask predictions',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-slot', dest='pretrained_slot', default=None, metavar='PATH',
                    help='path to pre-trained slot model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                    help='path to pre-trained Flow net model')

parser.add_argument('--resolution', type=int, default=(256,832))
parser.add_argument('--THRESH', dest='THRESH', type=float, default=0.94, help='THRESH')
parser.add_argument('--dataset', dest='dataset', default='kitti2015', help='path to pre-trained Flow net model')
parser.add_argument('--output-dir', dest='output_dir', type=str, default=None, help='path to output directory')


def main():
    global args
    args = parser.parse_args()

    args.pretrained_disp = Path(args.pretrained_disp)
    args.pretrained_pose = Path(args.pretrained_pose)
    args.pretrained_slot = Path(args.pretrained_slot)
    args.pretrained_flow = Path(args.pretrained_flow)
    
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
        args.output_dir.makedirs_p()

        image_dir = args.output_dir/'images'
        gt_dir = args.output_dir/'gt'
        mask_dir = args.output_dir/'mask'
        viz_dir = args.output_dir/'viz'

        image_dir.makedirs_p()
        gt_dir.makedirs_p()
        mask_dir.makedirs_p()
        viz_dir.makedirs_p()

        output_writer = SummaryWriter(args.output_dir)

    # train dataset
    _, _, _, val_mask_set, resolution = cg.setup_KITTI_dataset(args)
    val_loader = torch.utils.data.DataLoader(val_mask_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True)

    print("======init model========>")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    flow_net = model.PWCNet(md=4).to(device)
    pose_net = model.PoseNetB6().to(device)
    disp_net = model.DispResNet6().to(device)
    slot_net = model.SlotAttentionAutoEncoder(resolution=resolution, num_slots=2, in_out_channels=3, iters=5).to(device)
    
    print("=> using pre-trained weights for explainabilty and pose net")
    weights = torch.load(args.pretrained_pose)
    model_state = pose_net.state_dict()
    weights['state_dict'] = {k:v for k, v in weights['state_dict'].items() if k in model_state and v.size() == model_state[k].size()}
    model_state.update(weights['state_dict'])
    pose_net.load_state_dict(model_state)

    print("=> using pre-trained weights for slot attention net")
    weights = torch.load(args.pretrained_slot)
    model_state = slot_net.state_dict()
    weights['model_state_dict'] = {k:v for k, v in weights['model_state_dict'].items() if k in model_state and v.size() == model_state[k].size()}
    model_state.update(weights['model_state_dict'])
    slot_net.load_state_dict(model_state)

    print("=> using pre-trained weights from {}".format(args.pretrained_disp))
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'])

    print("=> using pre-trained weights for FlowNet")
    weights = torch.load(args.pretrained_flow)
    flow_net.load_state_dict(weights)

    disp_net.eval()
    pose_net.eval()
    flow_net.eval()
    slot_net.eval()
    
    error_names = ['tp_0', 'fp_0', 'fn_0', 'tp_1', 'fp_1', 'fn_1']
    errors_geo = AverageMeter(i=len(error_names))
    errors_slot = AverageMeter(i=len(error_names))

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt, semantic_map_gt) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            tgt_img_var = Variable(tgt_img.cuda())
            ref_imgs_var = Variable(ref_imgs.cuda())

            intrinsics_var = Variable(intrinsics.cuda())
            intrinsics_inv_var = Variable(intrinsics_inv.cuda())

            flow_gt_var = Variable(flow_gt.cuda())
            obj_map_gt_var = Variable(obj_map_gt.cuda())
        
        disp = disp_net(tgt_img_var)
        disp_ref = disp_net(ref_imgs_var)
        disp = spatial_normalize(disp)
        disp_ref = spatial_normalize(disp_ref)

        depth = 1/disp
        depth_ref = 1/disp_ref

        pose = pose_net(tgt_img_var, [ref_imgs_var])

            # pose_mat [B, 4, 4] RT  (t => t+1)
        pose_mat = project.pose_vec2mat(pose[:, 0])

        flow_bwd = flow_net(tgt_img_var, ref_imgs_var)
        flow_fwd = flow_net(ref_imgs_var, tgt_img_var)

        gt_mask_np = obj_map_gt[0].numpy()
        semantic_map_np = semantic_map_gt[0].numpy()

        ## output the mask for the slot attention
        motion_masks, scene_flows = project.get_motion_mask(tgt_img_var, flow_bwd, flow_fwd, depth, depth_ref, intrinsics_inv_var, pose_mat, alpha=1, test=True)
        
        motion_mask = motion_masks[0] > args.THRESH
        motion_masks = 1 - motion_masks
        
        ## slot attention
        recon_flow, recons, masks, _ = slot_net(scene_flows)
        slot_mask = masks[0][1] > args.THRESH

        motion_mask_np = motion_mask.cpu().data.numpy()
        slot_mask_np = slot_mask.cpu().data.numpy()
        disp_np = disp.cpu().data[0].numpy().transpose((1, 2, 0))
        # print(disp_np.shape)

        _errors_geo = mask_error(gt_mask_np, semantic_map_np, motion_mask_np[0])
        _errors_slot = mask_error(gt_mask_np, semantic_map_np, slot_mask_np[0])

        errors_geo.update(_errors_geo)
        errors_slot.update(_errors_slot)
        
        if args.output_dir is not None:
            tgt_img_viz = visual.tensor2array(tgt_img[0].cpu())
            ref_img_viz = visual.tensor2array(ref_imgs[0].cpu())
            # depth_viz = visual.tensor2array(disp.data[0].cpu(), max_value=None, colormap='hot')
            plt.imsave(viz_dir / str(i).zfill(3) + '_disp.jpg', visual.disp_norm_for_vis(disp_np[:, :, 0]), cmap='plasma')
            mask_viz = visual.tensor2array(motion_masks[0].data.cpu(), max_value=1, colormap='bone')
            slot_mask = visual.tensor2array(slot_mask.data.cpu(), max_value=1, colormap='bone')
            gt_viz = visual.tensor2array(obj_map_gt[0].cpu())
            flow_viz = visual.flow_to_image(visual.tensor2array(flow_bwd.data[0].cpu()))
            row1_viz = np.hstack((flow_viz, mask_viz, slot_mask))

            source = np.hstack((tgt_img_viz, ref_img_viz, mask_viz))

            row1_viz *= 255
            source *= 255
            gt_viz *= 255
            #print(viz3.shape)
            row1_viz_im = Image.fromarray(np.transpose(row1_viz, (1, 2, 0)).astype(np.uint8))

            source_im = Image.fromarray(np.transpose(source, (1, 2, 0)).astype(np.uint8))
            masks_im = Image.fromarray(np.transpose(gt_viz, (1, 2, 0)).astype(np.uint8))


            row1_viz_im.save(viz_dir/str(i).zfill(3)+'01.png')

            source_im.save(viz_dir / str(i).zfill(3) + '02.png')
            masks_im.save(viz_dir / str(i).zfill(3) + '03.png')
 
    bg_iou = errors_geo.sum[0] / (errors_geo.sum[0] + errors_geo.sum[1] + errors_geo.sum[2]  )
    fg_iou = errors_geo.sum[3] / (errors_geo.sum[3] + errors_geo.sum[4] + errors_geo.sum[5]  )
    avg_iou = (bg_iou + fg_iou)/2

    bg_iou_slot = errors_slot.sum[0] / (errors_slot.sum[0] + errors_slot.sum[1] + errors_slot.sum[2]  )
    fg_iou_slot = errors_slot.sum[3] / (errors_slot.sum[3] + errors_slot.sum[4] + errors_slot.sum[5]  )
    avg_iou_slot = (bg_iou_slot + fg_iou_slot)/2

    print("Results Model")
    print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    print("IOU \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou, bg_iou, fg_iou))


    print("Results slot Model")
    print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    print("IOU \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou_slot, bg_iou_slot, fg_iou_slot))


def mask_error(mot_gt, seg_gt, pred):
    """
    mot_gt = obj_map
    seg_gt = semantic_map
    
    """
    max_label = 2
    tp = np.zeros((max_label))
    fp = np.zeros((max_label))
    fn = np.zeros((max_label))

    mot_gt[mot_gt != 0] = 1
    mov_car_gt = mot_gt
    mov_car_gt[seg_gt != 26] = 255
    mot_gt = mov_car_gt
    
    r_shape = [float(i) for i in list(pred.shape)]
    g_shape = [float(i) for i in list(mot_gt.shape)]
    pred = zoom(pred, (g_shape[0] / r_shape[0],
                      g_shape[1] / r_shape[1]), order  = 0)

    if len(pred.shape) == 2:
        mask = pred
        umask = np.zeros((2, mask.shape[0], mask.shape[1]))
        umask[0, :, :] = mask
        umask[1, :, :] = 1. - mask
        pred = umask

    pred = pred.argmax(axis=0)
    if (np.max(pred) > (max_label - 1) and np.max(pred)!=255):
        print('Result has invalid labels: ', np.max(pred))
    else:
        # For each class
        for class_id in range(0, max_label):
            class_gt = np.equal(mot_gt, class_id)
            class_result = np.equal(pred, class_id)
            class_result[np.equal(mot_gt, 255)] = 0
            tp[class_id] = tp[class_id] +\
                np.count_nonzero(class_gt & class_result)
            fp[class_id] = fp[class_id] +\
                np.count_nonzero(class_result & ~class_gt)
            fn[class_id] = fn[class_id] +\
                np.count_nonzero(~class_result & class_gt)

    return [tp[0], fp[0], fn[0], tp[1], fp[1], fn[1]]


if __name__ == '__main__':
    main()