# Self-supervised Figure-ground segmentation from camera ego-motion (未完成)
## Current Results
| Method[mean IOU] | FG   | BG   | TOTAL(KITTI 2015 scene flow) |
| ---------------- | ---- | ---- | ---------------------------- |
| just geometric   | 0.39 | 0.35 | 0.37                         |
| with masknet     | 0.57 | 0.52 | 0.55                         |
| slot attention on scene flow| 0.48 | 0.10 | 0.29              |
| TransMask | 0.5130 | 0.6320 | 0.5725             |

## Result samples
All results could be download [here](https://drive.google.com/file/d/1YzR0FIVM1U3eEr2aiY3S-SYeIObKza2H/view?usp=sharing)

## MODEL
![image](https://user-images.githubusercontent.com/54012489/153791403-35cea64c-1f09-4f60-aeeb-219a96057e3b.png)

### Networks
* PoseNet = PoseNetB6  from [CC](https://github.com/anuragranj/cc)
* FlowNet = FlowNetCorr, PWCNet 
* DispNet = DispResNet 
* MaskNet = MaskNetCNN, **MaskResNet**, **MaskTransNet** 

### Datasets
* KITTI rawdata Eigen split. (65 scenes 35226 samples for train. 13 scenes 9304 samples for valid.)
* Eigen split 697 raw data samples for disparity test. 
* KITTI scene flow 200 scenes for optical flow and mask test.
* KITTI odometry pose ground truth 08.txt 09.txt for test (未完成)

### Loss Design
### 1. reconstruction loss for optical flow (optical used for the fore_ground scene)
$ L_{Flow} = \sum_{l=0}^{3}((1-w)|I_{t}- \widehat{I_{t}}|+w\frac{1-SSIM(I_{t},\widehat{I_{t}})}{2})\cdot Occ_{Mask}\cdot\sum Mask_{forebground} $
### 2. reconstruction loss for depth and pose
$ L_{Dp} = \sum_{l=0}^{3}((1-w)|I_{t}- \widehat{I_{t}}|+w\frac{1-SSIM(I_{t},\widehat{I_{t}})}{2})\cdot Occ_{Mask}\cdot\sum Mask_{background} $
### 3. edge-aware smoothness loss
$ L_{smoothness} = \sum_{l=0}^{3}(\bigtriangledown_{depth}^{2}+\bigtriangledown_{backwardFlow}^{2}+\bigtriangledown_{forwardFlow}^{2} +\bigtriangledown_{Masks}^{2})$
### 4. Mask consisitency loss
$ L_{consistency} = CrossEntropy(mask_{rigid, non-rigid}, mask_{background}) $
### 5. entropy loss for output mask
$ L_{entropy} = \sum_{l=0}^{k}-(Mask_{background}^{l} \cdot log(Mask_{background}^{l}+eps)) $

## Training Loop
```
# 初始化depth pose  absrel收敛
CUDA_VISIBLE_DEVICES=7 python3 train.py /home/cshi/Project/Figure_ground/format_data \
    --kitti-dir /home/cshi/dataset/KITTI/scene_flow/ \
    --dispnet DispResNet6 \
    --posenet PoseNetB6 --masknet MaskNet6 --flownet Back2Future \
    -b 16 -pc 1.0 -pf 0.0 -m 0.6 -c 0.0 -s 0.005 --epoch-size 1000 --log-output \
    -f 10 --nlevels 6 --lr 1e-4 -wssim 0.997 \
    --epochs 100 --smoothness-type edgeaware --fix-flownet \
    --with-depth-gt --name Reconstruct_Code_Debug


# 初始化flow
python3 train.py /media/karmen/HDD/EPC/format_data --kitti-dir /media/karmen/HDD/datasets/kitti/kitti2015 \
    --dispnet DispResNet6 \
    --posenet PoseNetB6 --masknet MaskNet6 --flownet Back2Future \
    -b 8 -pc 0.0 -pf 1.0 -m 0.0 -c 0.0 -s 0.005 --epoch-size 1000 --log-output \
    -f 10 --nlevels 6 --lr 1e-4 -wssim 0.997 \
    --epochs 100 --smoothness-type edgeaware --fix-dispnet --fix-posenet  --fix-masknet \
    --log-terminal --no-non-rigid-mask --with-flow-gt --name initial100 --resume

## 初始化mask
python3 train.py /media/karmen/HDD/EPC/format_data --kitti-dir /media/karmen/HDD/datasets/kitti/kitti2015 \
    --dispnet DispResNet6 \
    --posenet PoseNetB6 --masknet MaskNet6 --flownet Back2Future \
    -b 8 -pc 1.0 -pf 0.5 -m 0.005 -c 0.3 -s 0.005 --epoch-size 1000 --log-output \
    -f 10 --nlevels 6 --lr 1e-4 -wssim 0.997 \
    --epochs 100 --smoothness-type edgeaware --fix-dispnet --fix-posenet --fix-flownet \
    --log-terminal --with-depth-gt --name initial100 --resume


# while true


python3 train.py /media/karmen/HDD/EPC/format_data --kitti-dir /media/karmen/HDD/datasets/kitti/kitti2015 \
    --dispnet DispResNet6 \
    --posenet PoseNetB6 --masknet MaskNet6 --flownet Back2Future \
    -b 8 -pc 1.0 -pf 0.0 -m 0.05 -c 0.0 -s 0.005 --epoch-size 1000 --log-output \
    -f 10 --nlevels 6 --lr 1e-4 -wssim 0.997 \
    --epochs 100 --smoothness-type edgeaware --fix-flownet --fix-masknet \
    --log-terminal --with-depth-gt --name initial100 --resume

# 第一次可能需要no-non-rigid-mask
python3 train.py /media/karmen/HDD/EPC/format_data --kitti-dir /media/karmen/HDD/datasets/kitti/kitti2015 \
    --dispnet DispResNet6 \
    --posenet PoseNetB6 --masknet MaskNet6 --flownet Back2Future \
    -b 8 -pc 0.0 -pf 1.0 -m 0.005 -c 0.0 -s 0.005 --epoch-size 1000 --log-output \
    -f 10 --nlevels 6 --lr 1e-4 -wssim 0.997 \
    --epochs 100 --smoothness-type edgeaware --fix-dispnet --fix-posenet --fix-masknet \
    --log-terminal (--no-non-rigid-mask) --with-flow-gt --name initial100 --resume

# train mask MaskNet
python3 train.py /media/karmen/HDD/EPC/format_data --kitti-dir /media/karmen/HDD/datasets/kitti/kitti2015 \
    --dispnet DispResNet6 \
    --posenet PoseNetB6 --masknet MaskNet6 --flownet Back2Future \
    -b 8 -pc 1.0 -pf 0.5 -m 0.005 -c 0.3 -s 0.005 --epoch-size 1000 --log-output \
    -f 10 --nlevels 6 --lr 1e-4 -wssim 0.997 \
    --epochs 100 --smoothness-type edgeaware --fix-dispnet --fix-posenet --fix-masknet \
    --log-terminal --with-depth-gt --name initial100 --resume
 ```
