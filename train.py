from __future__ import absolute_import

import os
import time
import csv
from numpy.lib.function_base import disp
import torch
import model
import einops
import project
from logger import *
from torch.utils import data
import torchvision
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torch.utils.data import dataloader
from tqdm import tqdm
from itertools import chain
from losses import *
from test_masks import *
import config as cg
import utils.viz as visual
import torch.nn as nn
import torch.optim as optim

best_error = -1
n_iter = 0

def main(args):
    global n_iter, best_error

    lr = args.lr 
    alpha = args.alpha
    batch_size = args.batch_size
    epoches = args.epoches
    sequence_length = args.sequence_length
    resolution = args.resolution

    ## slot attention
    num_slots = args.num_slots
    iters = args.num_iterations

    ## set resume path
    resume = args.resume

    # init log
    [logPath, modelPath, resultsPath] = cg.setup_geometric_path(args)
    writer = SummaryWriter(logdir=logPath)

    # train dataset
    trn_dataset, val_dataset, val_flow_set, val_mask_set, resolution = cg.setup_KITTI_dataset(args)

    train_loader = dataloader.DataLoader(trn_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    val_loader = dataloader.DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=True)
    
    if args.valid_flow_gt:
        val_loader = dataloader.DataLoader(val_flow_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    if args.valid_mask_gt:
        val_loader = dataloader.DataLoader(val_mask_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    # init model
    print("======init model========>")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    flow_net = model.PWCNet(md=4).to(device)
    pose_net = model.PoseNetB6().to(device)
    disp_net = model.DispResNet6().to(device)
    slot_net = model.SlotAttentionAutoEncoder(resolution=resolution, num_slots=num_slots, in_out_channels=3, iters=iters).to(device)
    
    # init opt
    print("======setting adam optimizer=====>")
    parameters = chain(pose_net.parameters(), flow_net.parameters(), disp_net.parameters(), slot_net.parameters())
    optimizer = optim.Adam(parameters, lr=lr, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    # load pretrain
    if args.pretrained_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_pose)
        model_state = pose_net.state_dict()
        weights['state_dict'] = {k:v for k, v in weights['state_dict'].items() if k in model_state and v.size() == model_state[k].size()}
        model_state.update(weights['state_dict'])
        pose_net.load_state_dict(model_state)
    else:
        pose_net.init_weights()

    if args.pretrained_slot:
        print("=> using pre-trained weights for slot attention net")
        weights = torch.load(args.pretrained_slot)
        model_state = slot_net.state_dict()
        weights['model_state_dict'] = {k:v for k, v in weights['model_state_dict'].items() if k in model_state and v.size() == model_state[k].size()}
        model_state.update(weights['model_state_dict'])
        slot_net.load_state_dict(model_state)


    if args.pretrained_disp:
        print("=> using pre-trained weights from {}".format(args.pretrained_disp))
        weights = torch.load(args.pretrained_disp)
        
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    if args.pretrained_flow:
        print("=> using pre-trained weights for FlowNet")
        weights = torch.load(args.pretrained_flow)
        flow_net.load_state_dict(weights)
    
    if args.pretrained_optimizer:
        print("=> loading optimizer from checkpoint")
        optimizer_weights = torch.load(args.pretrained_optimizer)
        optimizer.load_state_dict(optimizer_weights['state_dict'])


    ## if debug activate this to 20
    epoch_size = 1000
    if args.debug:
        epoch_size = 10

    with open(logPath + '/' + args.log_summary, 'w') as csvfile:
        writer_csv = csv.writer(csvfile, delimiter='\t')
        writer_csv.writerow(['train_loss, mask_geo_iou', 'error(disp => abs diff, flow => outlier, mask_slot_iou)'])

    # add valid code
    logger = TermLogger(n_epochs=epoches, train_size=min(len(train_loader), epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()


    for epoch in range(args.epoches):

        if args.fix_flownet:
            for fparams in flow_net.parameters():
                fparams.requires_grad = False

        if args.fix_slotnet:
            for fparams in slot_net.parameters():
                fparams.requires_grad = False

        if args.fix_posenet:
            for fparams in pose_net.parameters():
                fparams.requires_grad = False

        if args.fix_dispnet:
            for fparams in disp_net.parameters():
                fparams.requires_grad = False

        logger.epoch_bar.update(epoch)
        logger.reset_train_bar()

        train_loss = train(train_loader, flow_net, pose_net, disp_net, slot_net, alpha, optimizer, epoch_size, logger, writer)

        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))
        
        ## start valid
        logger.reset_valid_bar()
        
        ## flow valid
        if args.valid_flow_gt:
            flow_errors, flow_error_names = validate_flow_with_gt(val_loader, disp_net, pose_net, flow_net, slot_net, epoch, logger)
            for error, name in zip(flow_errors, flow_error_names):
                writer.add_scalar(name, error, epoch)

        if args.valid_mask_gt:
            geo_iou, slot_iou = validate_mask_with_gt(val_loader, disp_net, pose_net, flow_net, slot_net, epoch, logger)
        ## depth valid
        if args.valid_depth_gt:
            errors, error_names = validate_depth_with_gt(val_loader, disp_net, epoch, logger, writer)

            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
            logger.valid_writer.write(' * Avg {}'.format(error_string))
            for error, name in zip(errors, error_names):
                writer.add_scalar(name, error, epoch)

        ## test depth error code
        # if not args.fix_posenet:
        #     decisive_error = flow_errors[-2]    # epe_rigid_with_gt_mask
        
        if args.valid_selector == 0:
            decisive_error = errors[0]      #depth abs_diff
        elif args.valid_selector == 1:
            decisive_error = flow_errors[-1]    #epe_non_rigid_with_gt_mask
        elif args.valid_selector == 2:
            decisive_error = 1-slot_iou     # percent outliers
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error <= best_error
        best_error = min(best_error, decisive_error)
        cg.save_checkpoint(
            modelPath, {
                'epoch': epoch + 1,
                'state_dict': disp_net.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': flow_net.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': slot_net.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': optimizer.state_dict()
            },
            is_best) 

        with open(logPath + '/' + args.log_summary, 'a') as csvfile:
            writer_csv = csv.writer(csvfile, delimiter='\t')
            writer_csv.writerow([errors, error_names])

    logger.epoch_bar.finish()


def train(train_loader, flow_net, pose_net, disp_net, slot_net, alpha, optimizer, epoch_size, logger, training_writer):
    global args,  n_iter
    critertion = nn.MSELoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    ## init the loss weight
    w1, w2, w3, w4 = args.flow_reconstruction_loss, args.rigid_reconstruction_loss, args.smoothness_loss, args.slot_recon_loss
    w5, w6 = args.entropy_loss, args.consistency_loss

    # train setting
    flow_net.train()
    pose_net.train()
    disp_net.train()

    end = time.time()
    
    for i, sample_dic in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):
            data_time.update(time.time() - end)
                
            # input size (b, t, c, h ,w)
            tgt_img_l = Variable(sample_dic['input_left'].cuda())
            ref_img_l = Variable(sample_dic['input_ref'].cuda())


            ## input for the pose net
            # pose_pair = torch.cat((tgt_img_l, ref_img_l), dim=1)
            ## 内参 未归一化
            k_l1 = Variable(sample_dic['input_k_l'].cuda())
            k_l1_inv = Variable(sample_dic['input_k_inv'].cuda())

            # flow list
            # [4, 2, h, w]
            ## backward flow
            # ref + flow_bwd = tgt
            ## forward flow
            # tgt + flow_fwd = ref
            flow_bwd = flow_net(tgt_img_l, ref_img_l)
            flow_fwd = flow_net(ref_img_l, tgt_img_l)
            

            # pose output 
            # [4, 1, 1, 3]
            pose = pose_net(tgt_img_l, [ref_img_l])

            # pose_mat [B, 4, 4] RT  (t => t+1)
            pose_mat = project.pose_vec2mat(pose[:, 0])

            # pose的定义是从tgt image相机位姿到source image / SE(3) 
            # pose mat = [B, 3, 4]
            
            # disp list
            # disp size [4, 1, h, w]
            disp = disp_net(tgt_img_l)
            disp_r = disp_net(ref_img_l)
            disparity = [project.spatial_normalize(dis) for dis in disp]
            disparity_ref = [project.spatial_normalize(dis) for dis in disp_r]

            ## 这里将所有的视差转换为深度
            depth = []
            depth_ref= []
            for idx, disp_t in enumerate(disparity):
                ## warp the t frame to t+1
                depth.append(1/disp_t)
                depth_ref.append(1/disparity_ref[idx])

            ## 0 moving, 1 static
            # motion_mask, scene_flows = project.get_motion_mask(tgt_img_l, flow_bwd, flow_fwd, depth, depth_ref, k_l1_inv, pose_mat, alpha)

            ## see the rigid flow
            rigid_flow = project.pose2rigid_flow(depth[0], pose_mat, k_l1, k_l1_inv)

            dp_warped_img = project.inverse_warp(ref_img_l, depth[0].squeeze(1), pose_mat, k_l1, k_l1_inv)

            ## slot attention
            recon_frame_t, recons, masks, _ = slot_net(tgt_img_l)

            ## loss_1计算flow的photometric loss
            loss_1 = photometric_flow_loss(tgt_img_l, ref_img_l, flow_bwd)
            
            ## loss_2 compute the depth and pose photometric loss(has big problem)
            loss_2 = photometric_reconstruction_loss(tgt_img_l, ref_img_l, k_l1, k_l1_inv, depth, motion_mask, pose_mat)

            ## edge aware smoothness loss
            loss_3 = edge_aware_smoothness_loss(tgt_img_l, depth) + edge_aware_smoothness_loss(tgt_img_l, flow_bwd)
            loss_3 += edge_aware_smoothness_loss(tgt_img_l, flow_fwd)


            # loss for the slot attention
            # entropy loss for the masks 
            # recon_loss: recover the scene flow as input to slot attention 
            recon_loss = critertion(tgt_img_l, recon_frame_t)
            entropy_loss = 1e-2 * -(masks * torch.log(masks + epsilon)).sum(dim=1).mean()
            
            # consistency loss: for the masks
            # reverse_mask = 1 - motion_mask[0]
            # masks_geo = torch.stack((reverse_mask, motion_mask[0]), dim = 1)
            
            # mask_t = einops.rearrange(masks_geo, 'b s c h w -> b c s h w')
            # c=1, so this is to broadcast the difference matrix
            # temporal_diff = torch.pow((masks - mask_t), 2).mean([-1, -2])
            # consistency_loss = 1e-2 * temporal_diff.view(-1, 2 * 2).min(1)[0].mean()

            ## record the loss
            if i > 0 and n_iter % 10 == 0:
                training_writer.add_scalar('reconstruction loss for flow', loss_1.item(), n_iter)
                training_writer.add_scalar('reconstruction loss for depth & pose', loss_2.item(), n_iter)
                training_writer.add_scalar('smoothness_loss', loss_3.item(), n_iter)
                # training_writer.add_scalar('reconstruction loss for slot attention', recon_loss.item(), n_iter)
                # training_writer.add_scalar('entropy loss', entropy_loss.item(), n_iter)
                # training_writer.add_scalar('consistency loss', consistency_loss.item(), n_iter)
                training_writer.add_scalar('total_loss', loss.item(), n_iter)

                ## add images in training
                training_writer.add_image('image frame t', visual.tensor2array(tgt_img_l[0]), n_iter)
                training_writer.add_image('image frame t+1', visual.tensor2array(ref_img_l[0]), n_iter)

                training_writer.add_image('backward flow t+1=>t', visual.flow_to_image(visual.tensor2array(flow_bwd[0].data[0].cpu())), n_iter)
                training_writer.add_image('backward rigid flow', visual.flow_to_image(visual.tensor2array(rigid_flow.data[0].cpu())), n_iter)


                # training_writer.add_image('Geometric Motion Mask Outputs', visual.tensor2array(reverse_mask[0].data.cpu(), max_value=1, colormap='bone') , n_iter)
                
                training_writer.add_image('Slot Mask Outputs', visual.tensor2array(masks[0][1].data.cpu(), max_value=1, colormap='bone') , n_iter)
                training_writer.add_image('Slot Mask Outputs 2', visual.tensor2array(masks[0][0].data.cpu(), max_value=1, colormap='bone') , n_iter)
                
                training_writer.add_image('inverse warped image', visual.tensor2array(dp_warped_img[0]), n_iter)
                # training_writer.add_image('train occlusion mask Outputs', 
                #                 visual.tensor2array(occlusion_masks[0][0][0:1, :, :].data.cpu(), max_value=1, colormap='bone') , n_iter )
                for k, scaled_depth in enumerate(depth):
                    training_writer.add_image('train Dispnet Output Normalized {}'.format(k),
                                        visual.tensor2array(disparity[k].data[0].cpu(), max_value=None, colormap='bone'),
                                        n_iter)


            loss = w1 * loss_1  + w2 * loss_2 + w3 * loss_3
            
            ## backward update
            losses.update(loss.item(), args.batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update terminal train bar
            logger.train_bar.update(i+1)
            if i % 10 == 0:
                logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))

            if i >= epoch_size - 1:
                break

            n_iter = n_iter + 1

    return losses.avg[0]


def validate_depth_with_gt(val_loader, disp_net, epoch, logger, output_writers=[]):
    global args
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    # log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()

    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img_var = Variable(tgt_img.cuda(), requires_grad=False)

        output_disp = disp_net(tgt_img_var)

        output_disp = project.spatial_normalize(output_disp)

        output_depth = 1/output_disp

        depth = depth.cuda()

        # compute output

        errors.update(compute_errors(depth, output_depth.data.squeeze(1)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.valid_bar.update(i)
        if i % 10 == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))

    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names


def validate_flow_with_gt(val_flow_loader, disp_net, pose_net, flow_net, slot_net, epoch, logger):
    global args
    batch_time = AverageMeter()
    error_names = ['epe_total', 'outliers']
    errors = AverageMeter(i=len(error_names))

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    flow_net.eval()
    slot_net.eval()

    end = time.time()

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt) in enumerate(val_flow_loader):
        
        tgt_img_var = Variable(tgt_img.cuda(), requires_grad=False)
        ref_imgs_var = Variable(ref_imgs.cuda(), requires_grad=False)

        intrinsics_var = Variable(intrinsics.cuda(), requires_grad=False)
        intrinsics_inv_var = Variable(intrinsics_inv.cuda(), requires_grad=False)
        
        flow_gt_var = Variable(flow_gt.cuda(), requires_grad=False)
        obj_map_gt_var = Variable(obj_map_gt.cuda(), requires_grad=False)

        disp = disp_net(tgt_img_var)
        disp = spatial_normalize(disp)

        depth = 1/disp

        pose = pose_net(tgt_img_var, [ref_imgs_var])

            # pose_mat [B, 4, 4] RT  (t => t+1)
        pose_mat = project.pose_vec2mat(pose[:, 0])

        flow_bwd = flow_net(tgt_img_var, ref_imgs_var)
        flow_fwd = flow_net(ref_imgs_var, tgt_img_var)

        flow_cam = project.pose2rigid_flow(depth, pose_mat, intrinsics_var, intrinsics_inv_var)

        _epe_errors = compute_all_epes(flow_gt_var, flow_bwd)
        errors.update(_epe_errors)

        batch_time.update(time.time() - end)
        end = time.time()

        logger.valid_bar.update(i)
    logger.valid_bar.update(len(val_flow_loader))
    return errors.avg, error_names


def validate_mask_with_gt(val_mask_loader, disp_net, pose_net, flow_net, slot_net, epoch, logger):
    global args

    batch_time = AverageMeter()
    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    flow_net.eval()
    slot_net.eval()

    error_names = ['tp_0', 'fp_0', 'fn_0', 'tp_1', 'fp_1', 'fn_1']
    errors_geo = AverageMeter(i=len(error_names))
    errors_slot = AverageMeter(i=len(error_names))
    end = time.time()

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt, semantic_map_gt) in enumerate(val_mask_loader):
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
        motion_masks, scene_flows= project.get_motion_mask(tgt_img_var, flow_bwd, flow_fwd, depth, depth_ref, intrinsics_inv_var, pose_mat, alpha=1, test=True)
        motion_mask = motion_masks[0] > args.THRESH
        
        ## slot attention
        recon_flow, recons, masks, _ = slot_net(scene_flows)
        slot_mask = masks[0][1] > args.THRESH

        motion_mask_np = motion_mask.cpu().data.numpy()
        slot_mask_np = slot_mask.cpu().data.numpy()

        _errors_geo = mask_error(gt_mask_np, semantic_map_np, motion_mask_np[0])
        _errors_slot = mask_error(gt_mask_np, semantic_map_np, slot_mask_np[0])

        errors_geo.update(_errors_geo)
        errors_slot.update(_errors_slot)

        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i)
    logger.valid_bar.update(len(val_mask_loader))

    bg_iou = errors_geo.sum[0] / (errors_geo.sum[0] + errors_geo.sum[1] + errors_geo.sum[2]  )
    fg_iou = errors_geo.sum[3] / (errors_geo.sum[3] + errors_geo.sum[4] + errors_geo.sum[5]  )
    avg_iou = (bg_iou + fg_iou)/2

    bg_iou_slot = errors_slot.sum[0] / (errors_slot.sum[0] + errors_slot.sum[1] + errors_slot.sum[2]  )
    fg_iou_slot = errors_slot.sum[3] / (errors_slot.sum[3] + errors_slot.sum[4] + errors_slot.sum[5]  )
    avg_iou_slot = (bg_iou_slot + fg_iou_slot)/2

    # print("Results Model")
    # print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    # print("IOU \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou, bg_iou, fg_iou))


    # print("Results slot Model")
    # print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    # print("IOU \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou_slot, bg_iou_slot, fg_iou_slot))

    return avg_iou, avg_iou_slot


if __name__ == "__main__":
    parser = ArgumentParser()
    ## optimization
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoches', type=int, default=200)
    parser.add_argument('--sequence_length', type=int, default=2)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay')
    
    parser.add_argument('--debug', action='store_true', help='go to debug mode')
    # slot attention set
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--num_slots', type=int, default=2)

    ## control the mask
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--THRESH', dest='THRESH', type=float, default=0.94, help='THRESH')

    ## control the loss
    parser.add_argument('--flow-reconstruction-loss', type=float, help='weight for flow warp reconstruction loss',
                    metavar='W', default=0)
    parser.add_argument('--rigid-reconstruction-loss', type=float, help='weight for rigid reconstruction loss',
                    metavar='W', default=0)
    parser.add_argument('--smoothness-loss', type=float, help='weight for smoothness loss',
                    metavar='W', default=0)
    parser.add_argument('--slot-recon-loss', type=float, help='weight for slot attention reconstruction loss',
                    metavar='W', default=0)
    parser.add_argument('--entropy-loss', type=float, help='weight for mask entropy loss',
                    metavar='W', default=0)
    parser.add_argument('--consistency-loss', type=float, help='weight for mask consistency loss',
                    metavar='W', default=0)


    ## valid select
    parser.add_argument('--valid-depth-gt', action='store_true', help='use depth ground truth for validation')
    parser.add_argument('--valid-flow-gt', action='store_true', help='use flow ground truth for flow validation')
    parser.add_argument('--valid-mask-gt', action='store_true', help='use flow ground truth and obj map for mask validation')
    parser.add_argument('--valid-selector', type=int, default=0, help='0 depth, 1 flow, 2 mask')
    ## fix & pretrain
    parser.add_argument('--fix-slotnet', dest='fix_slotnet', action='store_true', help='do not train slotnet')
    parser.add_argument('--fix-posenet', dest='fix_posenet', action='store_true', help='do not train posenet')
    parser.add_argument('--fix-flownet', dest='fix_flownet', action='store_true', help='do not train flownet')
    parser.add_argument('--fix-dispnet', dest='fix_dispnet', action='store_true', help='do not train dispnet')

    parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                        help='path to pre-trained dispnet model')
    parser.add_argument('--pretrained-slot', dest='pretrained_slot', default=None, metavar='PATH',
                        help='path to pre-trained slot model')
    parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH',
                        help='path to pre-trained Exp Pose net model')
    parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                        help='path to pre-trained Flow net model')
    parser.add_argument('--pretrained-opti', dest='pretrained_optimizer', default=None, metavar='PATH',
                        help='path to pretrained optimizer')


    parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=(256,832))
    args = parser.parse_args()
    args.inference = False
    main(args)
