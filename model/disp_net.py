import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np

from torch.autograd import Variable


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    ## conv2d
    ## out shape = (H + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1)/stride[0] + 1
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1)
    )

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

def upsample_4x(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)

def predict_disp(in_planes):
    return nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=True)


class DepthCNN(nn.Module):
    """
    depthnet modified from pwcnet(out put is disparity)
    the deconv is removed by bilinear upsampling
    还得改，这个不太行
    """

    def __init__(self):
        super(DepthCNN, self).__init__()

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1, inplace=False)

        dd = [128,128,96,64,32]

        ## 每个分辨率都输出一个disp的值
        od = 196
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(dd[3],32,  kernel_size=3, stride=1)
        self.predict_disp6 = predict_disp(dd[4])
        # self.upsample_feat6 = upsample()

        
        self.conv5_0 = conv(32,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(dd[3],32,  kernel_size=3, stride=1)
        self.predict_disp5 = predict_disp(dd[4])

        self.conv4_0 = conv(32,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(dd[3],32,  kernel_size=3, stride=1)
        self.predict_disp4 = predict_disp(dd[4]) 

        self.conv3_0 = conv(32,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(dd[3],32,  kernel_size=3, stride=1)
        self.predict_disp3 = predict_disp(dd[4]) 

        self.conv2_0 = conv(32,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(dd[3],32,  kernel_size=3, stride=1)
        self.predict_disp2 = predict_disp(dd[4]) 


    def forward(self, img):
        # disp(It)可以输入 L和R两个双目图像来预测
        c11 = self.conv1b(self.conv1aa(self.conv1a(img)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))

        # c16 [4, 196, 4 ,13]
        x = self.conv6_0(c16)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        x = self.conv6_4(x)
        disp_6 = self.predict_disp6(x)
        upfeat_6 = upsample(x)

        x = self.conv5_0(upfeat_6)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        disp_5 = self.predict_disp5(x)
        upfeat_5 = upsample(x)

        x = self.conv4_0(upfeat_5)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        disp_4 = self.predict_disp4(x)
        upfeat_4 = upsample(x)

        x = self.conv3_0(upfeat_4)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        disp_3 = self.predict_disp3(x)
        upfeat_3 = upsample(x)

        x = self.conv2_0(upfeat_3)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        disp_2 = self.predict_disp2(x)
        
        disp2_up = upsample_4x(disp_2)
        disp3_up = upsample_4x(disp_3)
        disp4_up = upsample_4x(disp_4)
        disp5_up = upsample_4x(disp_5)

        ## disp2 output (4, 1, 64, 208)
        if self.training:
            return [disp2_up, disp3_up, disp4_up, disp5_up]
        else:
            return disp2_up


