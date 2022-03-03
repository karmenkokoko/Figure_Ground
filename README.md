# Self-supervised Figure-ground segmentation from camera ego-motion (未完成)
# 从摄像机自我运动中进行自监督的运动物体分割

## MODEL
![image](https://user-images.githubusercontent.com/54012489/153791403-35cea64c-1f09-4f60-aeeb-219a96057e3b.png)

### Networks
* PoseNet = PoseNetB6  from [CC](https://github.com/anuragranj/cc)
* FlowNet = FlowNetCorr, PWCNet 
* DispNet = DispResNet (MONOCULAR)
* MaskNet = MaskNetCNN, MaskResNet, **MaskTransNet(new)** 

### Datasets
* **Training(Valid)**：KITTI rawdata **Eigen split**. (65 scenes 35226 samples for train. 13 scenes 9304 samples for valid.)
* **depth test**: Eigen split 697 raw data samples for disparity test. 
* **mask, optical flow test**: KITTI scene flow 200 scenes for optical flow and mask test.
* **camera pose test**: KITTI odometry pose ground truth 08.txt 09.txt for test (未完成)

### Loss Design
### 1. reconstruction loss for optical flow (optical used for the foreground scene)
$ L_{Flow} = \sum_{level=1}^{6}((1-w)|I_{t}- \widehat{I_{t}}|+w\frac{1-SSIM(I_{t},\widehat{I_{t}})}{2})\cdot Occlusion \cdot (1-Mask) $
### 2. reconstruction loss for depth and pose
$ L_{Dp} = \sum_{level=1}^{6}((1-w)|I_{t}- \widehat{I_{t}}|+w\frac{1-SSIM(I_{t},\widehat{I_{t}})}{2})\cdot Occlusion \cdot (Mask) $
### 3. edge-aware smoothness loss
$ L_{smoothness} = \sum_{level=1}^{6}(\bigtriangledown_{depth}^{2}+\bigtriangledown_{backwardFlow}^{2}+\bigtriangledown_{forwardFlow}^{2} +\bigtriangledown_{Masks}^{2})$
### 4. Mask consisitency loss
$ L_{consistency} = CrossEntropy(mask_{rigid, non-rigid}, Mask) $
### 5. entropy loss for output mask
$ L_{entropy} = CrossEntropy(mask_{ones}, Mask) $

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
 
## Current Results (MIoU)
| Method | foreground scene   | background scene   | TOTAL(KITTI 2015 scene flow) |
| ---------------- | ---- | ---- | ---------------------------- |
| Previous(sota)     | 0.57 | 0.52 | 0.55                         |
| TransMask | 0.5130 | 0.6320 | 0.5725             |

## Result samples
All results could be download [here](https://drive.google.com/file/d/1YzR0FIVM1U3eEr2aiY3S-SYeIObKza2H/view?usp=sharing)
| IMG | DEPTH |
| ----- | ----- |
| ![00201](https://user-images.githubusercontent.com/54012489/152453327-d31c2396-1449-42b5-991e-67a3017384b9.png) | ![002_03](https://user-images.githubusercontent.com/54012489/152453397-525e6f97-40b5-4150-9039-5da59dcd0063.png) |
| ![04601](https://user-images.githubusercontent.com/54012489/152453466-7c66a7d7-3d08-4294-82ed-bd9955edbbec.png) | ![046_03](https://user-images.githubusercontent.com/54012489/152453482-c79ba389-b123-4741-9984-f2463ff4b666.png) |
