import torch
import occ
import project
import torch.nn.functional as F
from torch.autograd import Variable
from utils.ssim import ssim

epsilon = 1e-8


def spatial_normalize(disp):
    _mean = disp.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    disp = disp / _mean
    return disp


def robust_l1(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    x = x.mean()
    return x


def robust_l1_per_pix(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    return x


def photometric_flow_loss(tgt_img, ref_img, flows, lambda_oob=0, qch=0.5, wssim=0.997):
    """
    use flow to get photometric loss
    """
    def one_scale(img_t, img_r, occ_mask, flow):

        reconstruction_loss = 0
        weight = 1.

        ref_img_warp = project.flow_warp(img_r, flow)
        diff = img_t - ref_img_warp
        ssim_loss = (1 - ssim(img_t, ref_img_warp)) / 2.

        # diff = (diff * occ_mask).expand_as(diff)
        # ssim_loss = (ssim_loss * occ_mask).expand_as(ssim_loss)

        reconstruction_loss += (1-wssim)*(robust_l1(diff, q=qch)) + wssim*ssim_loss.mean()

        return reconstruction_loss

    loss = 0
    for idx, flow_ in enumerate(flows):
        _, _, h, w = flow_.size()
        scaled_img = F.adaptive_avg_pool2d(tgt_img, (h, w))
        scaled_img_ref = F.adaptive_avg_pool2d(ref_img, (h, w))
        occ_mask = occ.occulsion_mask(scaled_img, flow_)
        loss += one_scale(scaled_img, scaled_img_ref, occ_mask, flow_)

    return loss


def photometric_reconstruction_loss(tgt_img, ref_img, intrinsics, intrinsics_inv, depth, dynamic_mask, pose, qch=0.5, wssim=0.997):
    """
    use rigid flow to compute occlusion mask
    """
    def one_scale(d, motion_mask):
        
        # print(torch.any(torch.isnan(motion_mask)))

        reconstruction_loss = 0.
        b, _, h, w = d.size()
        downscale = tgt_img.size(2) / h

        tgt_img_scaled = F.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_img_scaled = F.adaptive_avg_pool2d(ref_img, (h, w))

        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_inv_scaled = torch.cat((intrinsics_inv[:, 0:2]/downscale, intrinsics_inv[:, 2:]), dim=1)

        ## generate the occ mask ?
        rigid_flow = project.pose2rigid_flow(d, pose, intrinsics_scaled, intrinsics_inv_scaled)
        occ_mask_scale = occ.occulsion_mask(tgt_img_scaled, rigid_flow)

        weight = 1.

        ref_img_warped = project.inverse_warp(ref_img_scaled, d, pose, intrinsics_scaled, intrinsics_inv_scaled)

        valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
        diff = (tgt_img_scaled - ref_img_warped) * valid_pixels

        ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
        oob_normalization_const = valid_pixels.nelement() / valid_pixels.sum()
        # print(motion_mask.size())
        
        # diff = diff * occ_mask_scale[:, 0:1].expand_as(diff)
        # ssim_loss = ssim_loss * occ_mask_scale[:, 0:1].expand_as(ssim_loss)

        reconstruction_loss += (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean())

        return reconstruction_loss

    loss = 0.
    for idx, depth_ in enumerate(depth):
        loss += one_scale(depth_, dynamic_mask[idx])
        
    return loss


def photometric_reconstruction_loss_mask(tgt_img, ref_img, intrinsics, intrinsics_inv, depth, dynamic_mask, pose, qch=0.5, wssim=0.997):
    """
    use rigid flow to compute occlusion mask
    """
    def one_scale(d, motion_mask):
        
        # print(torch.any(torch.isnan(motion_mask)))

        reconstruction_loss = 0.
        b, _, h, w = d.size()
        downscale = tgt_img.size(2) / h

        tgt_img_scaled = F.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_img_scaled = F.adaptive_avg_pool2d(ref_img, (h, w))

        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_inv_scaled = torch.cat((intrinsics_inv[:, 0:2]/downscale, intrinsics_inv[:, 2:]), dim=1)

        ## generate the occ mask ?
        rigid_flow = project.pose2rigid_flow(d, pose, intrinsics_scaled, intrinsics_inv_scaled)
        occ_mask_scale = occ.occulsion_mask(tgt_img_scaled, rigid_flow)

        weight = 1.

        ref_img_warped = project.inverse_warp(ref_img_scaled, d, pose, intrinsics_scaled, intrinsics_inv_scaled)

        valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
        diff = (tgt_img_scaled - ref_img_warped) * valid_pixels

        ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
        oob_normalization_const = valid_pixels.nelement() / valid_pixels.sum()
        # print(motion_mask.size())
        
        # diff = diff * occ_mask_scale[:, 0:1].expand_as(diff)
        # ssim_loss = ssim_loss * occ_mask_scale[:, 0:1].expand_as(ssim_loss)

        reconstruction_loss += (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean())

        return reconstruction_loss

    loss = 0.
    for idx, depth_ in enumerate(depth):
        loss += one_scale(depth_, dynamic_mask[idx])
        
    return loss


def edge_aware_smoothness_loss(img, pred_disp):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    def get_edge_smoothness(img, pred):
      pred_gradients_x = gradient_x(pred)
      pred_gradients_y = gradient_y(pred)

      image_gradients_x = gradient_x(img)
      image_gradients_y = gradient_y(img)

      weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
      weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

      smoothness_x = torch.abs(pred_gradients_x) * weights_x
      smoothness_y = torch.abs(pred_gradients_y) * weights_y
      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        scaled_img = F.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp)
        weight /= 2.3   # 2sqrt(2)

    return loss



def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


def compute_epe(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = F.interpolate(pred, size=(h_gt, w_gt), mode='bilinear', align_corners=False)
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    if nc == 3:
        valid = gt[:,2,:,:]
        epe = epe * valid
        avg_epe = epe.sum()/(valid.sum() + epsilon)
    else:
        avg_epe = epe.sum()/(bs*h_gt*w_gt)

    if type(avg_epe) == Variable: avg_epe = avg_epe.data

    return avg_epe.item()


def outlier_err(gt, pred, tau=[3,0.05]):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt, valid_gt = gt[:,0,:,:], gt[:,1,:,:], gt[:,2,:,:]
    pred = F.interpolate(pred, size=(h_gt, w_gt), mode='bilinear', align_corners=False)
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    epe = epe * valid_gt

    F_mag = torch.sqrt(torch.pow(u_gt, 2)+ torch.pow(v_gt, 2))
    E_0 = (epe > tau[0]).type_as(epe)
    E_1 = ((epe / (F_mag+epsilon)) > tau[1]).type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    #n_err   = length(find(F_val & E>tau(1) & E./F_mag>tau(2)));
    #n_total = length(find(F_val));
    f_err = n_err.sum()/(valid_gt.sum() + epsilon)
    if type(f_err) == Variable: f_err = f_err.data
    return f_err.item()


def compute_all_epes(gt, flow_pred):

    all_epe = compute_epe(gt, flow_pred)

    outliers = outlier_err(gt, flow_pred)

    return [all_epe, outliers]