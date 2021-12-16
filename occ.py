from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import skimage.io as io
import datasets
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import imageio


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000),
             'hot': cm.get_cmap('hot')}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array[:,:,:3]
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        if (tensor.size(0) == 3):
            array = 0.5 + tensor.numpy()*0.5
        elif (tensor.size(0) == 2):
            array = tensor.numpy()

    return array


def transformerFwd(U,
                   flo,
                   out_size,
                   name='SpatialTransformerFwd'):
    """Forward Warping Layer described in
    'Occlusion Aware Unsupervised Learning of Optical Flow by Yang Wang et al'
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    flo: float
        The optical flow used for forward warping
        having the shape of [num_batch, height, width, 2].
    backprop: boolean
        Indicates whether to back-propagate through forward warping layer
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        rep = torch.ones(size=[n_repeats], dtype=torch.long).unsqueeze(1).transpose(1,0)
        x = x.view([-1,1]).mm(rep)
        return x.view([-1]).int()

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, height, width, channels = im.shape[0], im.shape[1], im.shape[2], im.shape[3]
        out_height = out_size[0]
        out_width = out_size[1]
        max_y = int(height - 1)
        max_x = int(width - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width - 1.0) / 2.0
        y = (y + 1.0) * (height - 1.0) / 2.0

        # do sampling
        x0 = (torch.floor(x)).int()
        x1 = x0 + 1
        y0 = (torch.floor(y)).int()
        y1 = y0 + 1

        x0_c = torch.clamp(x0, 0, max_x)
        x1_c = torch.clamp(x1, 0, max_x)
        y0_c = torch.clamp(y0, 0, max_y)
        y1_c = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width).to(im.get_device())

        base_y0 = base + y0_c * dim2
        base_y1 = base + y1_c * dim2
        idx_a = base_y0 + x0_c
        idx_b = base_y1 + x0_c
        idx_c = base_y0 + x1_c
        idx_d = base_y1 + x1_c

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.reshape([-1, channels])
        im_flat = im_flat.float()

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1).to(torch.float32)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1).to(torch.float32)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1).to(torch.float32)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1).to(torch.float32)

        zerof = torch.zeros_like(wa)
        wa = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y0_c, y0)).unsqueeze(1), wa, zerof)
        wb = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y1_c, y1)).unsqueeze(1), wb, zerof)
        wc = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y0_c, y0)).unsqueeze(1), wc, zerof)
        wd = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y1_c, y1)).unsqueeze(1), wd, zerof)

        zeros = torch.zeros(
            size=[
                int(num_batch) * int(height) *
                int(width), int(channels)
            ],
            dtype=torch.float)
        output = zeros.to(im.get_device())

        output = output.scatter_add(dim=0, index=idx_a.long().unsqueeze(1).repeat(1,channels), src=im_flat * wa)
        output = output.scatter_add(dim=0, index=idx_b.long().unsqueeze(1).repeat(1,channels), src=im_flat * wb)
        output = output.scatter_add(dim=0, index=idx_c.long().unsqueeze(1).repeat(1,channels), src=im_flat * wc)
        output = output.scatter_add(dim=0, index=idx_d.long().unsqueeze(1).repeat(1,channels), src=im_flat * wd)

        return output

    def _meshgrid(height, width):
        # This should be equivalent to:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                                 np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        return torch.from_numpy(x_t).float(), torch.from_numpy(y_t).float()

    def _transform(flo, input_dim, out_size):
        num_batch, height, width, num_channels = input_dim.shape[0:4]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        x_s, y_s = _meshgrid(out_height, out_width)
        x_s = x_s.to(flo.get_device()).unsqueeze(0)
        x_s = x_s.repeat([num_batch, 1, 1])
        # print(x_s)
        y_s = y_s.to(flo.get_device()).unsqueeze(0)
        y_s =y_s.repeat([num_batch, 1, 1])
        # print(y_s)

        x_t = x_s + flo[:, :, :, 0] / ((out_width - 1.0) / 2.0)
        y_t = y_s + flo[:, :, :, 1] / ((out_height - 1.0) / 2.0)

        x_t_flat = x_t.view([-1])
        y_t_flat = y_t.view([-1])

        input_transformed = _interpolate(input_dim, x_t_flat, y_t_flat,
                                            out_size)

        output = input_transformed.view([num_batch, out_height, out_width, num_channels])
        return output

    #out_size = int(out_size)
    output = _transform(flo, U, out_size)
    return output


def occulsion_mask(img, flow):
    mask = torch.ones(img.size()).to(flow.get_device())
    h, w = mask.shape[2], mask.shape[3]
    occ_mask = transformerFwd(mask.permute(0, 2, 3, 1), flow.permute(0, 2, 3, 1), out_size=[h, w]).permute(0, 3, 1, 2)
    with torch.no_grad():
        occ_mask = torch.clamp(occ_mask, 0.0, 1.0)

    # print(occ_mask)
    return occ_mask


def flow2oob(flow):

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    oob = (X.abs()>1).add(Y.abs()>1)>0
    return oob


""" if __name__ == "__main__":
    imgpath = "./test_img/0000000003.png"
    img = datasets.read_image_as_byte(imgpath)
    img_tensor = datasets.numpy_to_torch(img).unsqueeze(0)/255.
    img_tensor = F.interpolate(img_tensor, size=(370,1224), mode="bilinear", align_corners=True)
    img_tensor = img_tensor.cuda()
    # print(img_tensor)
    # print(img_tensor.size())

    flow_path = "./test_img/flow1.png"
    flow, valid = datasets.read_png_flow(flow_path)
    flow = flow.transpose(2, 0, 1)
    flow_tensor = torch.from_numpy(flow).unsqueeze(0).cuda()
    # print(flow_tensor)
    # print(flow_tensor.size())

    occ_mask = occulsion_mask(img_tensor, -flow_tensor)
    # print(occ_mask.size())
    occ_mask = F.interpolate(occ_mask, size=(370,1224), mode="area")
    occ_mask = occ_mask.squeeze()
    mask_viz = occ_mask.cpu().numpy()
    mask_viz = np.transpose(mask_viz, (1, 2, 0))[:, :, 0]
    mask_viz = (mask_viz*255).astype(np.uint8)

    imageio.imwrite('gray.jpg', mask_viz) """


