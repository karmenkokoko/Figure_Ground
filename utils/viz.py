from __future__ import absolute_import, division, print_function

import numpy as np
from skimage.color import lab2rgb
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image

COS_45 = 1. / np.sqrt(2)
SIN_45 = 1. / np.sqrt(2)
UNKNOWN_FLOW_THRESH = 1e7


def compute_color_sceneflow(sf):
    """
    scene flow color coding using CIE-LAB space.
    sf: input scene flow, numpy type, size of (h, w, 3)
    """

    # coordinate normalize
    max_sf = np.sqrt(np.sum(np.square(sf), axis=2)).max()
    # x = x/sqrt(x^2+y^2+z^2) ...
    sf = sf / max_sf

    # scene flow x, y, z
    sf_x = sf[:, :, 0]
    sf_y = sf[:, :, 1]
    sf_z = sf[:, :, 2]
    
    # rotating 45 degree
    # transform X, Y, Z -> Y, X', Z' -> L, a, b 
    
    sf_x_tform = sf_x * COS_45 + sf_z * SIN_45
    sf_z_tform = -sf_x * SIN_45 + sf_z * COS_45
    sf = np.stack([sf_y, sf_x_tform, sf_z_tform], axis=2) # [-1, 1] cube
    
    # norm vector to lab space: x, y, z -> z, x, y -> l, a, b
    # L∈（0,100） brightness
    # a∈（-128，127） yellow-blue opponent
    # b∈（-128，127） red-green opponent
    sf[:, :, 0] = sf[:, :, 0] * 50 + 50
    sf[:, :, 1] = sf[:, :, 1] * 127
    sf[:, :, 2] = sf[:, :, 2] * 127
    
    lab_vis = lab2rgb(sf)
    lab_vis = np.uint8(lab_vis * 255)
    lab_vis = np.stack([lab_vis[:, :, 2], lab_vis[:, :, 1], lab_vis[:, :, 0]], axis=2)
    
    return lab_vis


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[0]
    v = flow[1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    #print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return img.transpose(2,0,1)/255.


def disp_norm_for_vis(disp):
    """
    normalizing disparity for visualization.
    disp: input disparity, np type, size of (h, w)
    """
    disp_sort = np.sort(disp.flatten())
    disp_min = disp_sort[0]
    disp_max = disp_sort[-1]

    return (np.clip((disp - disp_min) / (disp_max - disp_min), 0, 1) * 255).astype(np.uint8)


def imrotate(arr, angle):
    return np.array(Image.fromarray(arr.astype('uint8')).rotate(angle, resample=Image.BILINEAR))

def imresize(arr, sz):
    height, width = sz
    return np.array(Image.fromarray(arr.astype('uint8')).resize((width, height), resample=Image.BILINEAR))

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




""" if __name__ == "__main__":

    output = np.load('./2011_09_30_drive_0028_0036_sf.npy')
    io.imsave('./test_2.png', compute_color_sceneflow(output), check_contrast=False) """
