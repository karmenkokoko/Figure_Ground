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


