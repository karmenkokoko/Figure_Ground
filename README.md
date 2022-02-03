# Self-supervised Figure-ground segmentation from camera ego-motion

loading......

| Method[mean IOU] | FG   | BG   | TOTAL(KITTI 2015 scene flow) |
| ---------------- | ---- | ---- | ---------------------------- |
| just geometric   | 0.39 | 0.35 | 0.37                         |
| with masknet     | 0.57 | 0.52 | 0.55                         |
| slot attention on scene flow| 0.48 | 0.10 | 0.29              |
| TransMask | 0.5130 | 0.6320 | 0.5725             |
## masknet is good
![image](https://user-images.githubusercontent.com/54012489/143808098-cfc0f440-0cd7-4c8b-b6f9-0f3faf5c7140.png)
## scene flow not good enough
![slide2](https://user-images.githubusercontent.com/54012489/127940143-076d455e-329c-4113-83bc-0551cabb9cf8.png)

## a new way to upgrade the masknet
![Fig](https://user-images.githubusercontent.com/54012489/146138094-70be6586-e177-4ce6-8e90-7bcf62782d8a.png)

### 1. reconstruction loss for optical flow
$ L_{Flow} = \sum_{l=0}^{3}((1-w)|I_{t}- \widehat{I_{t}}|+w\frac{1-SSIM(I_{t},\widehat{I_{t}})}{2})\cdot Occ_{Mask} $
### 2. reconstruction loss for depth and pose
$ L_{Dp} = \sum_{l=0}^{3}((1-w)|I_{t}- \widehat{I_{t}}|+w\frac{1-SSIM(I_{t},\widehat{I_{t}})}{2})\cdot Occ_{Mask}\cdot\sum(1-Mask_{forebground})\cdot\sum Mask_{background} $
### 3. edge-aware smoothness loss
$ L_{smoothness} = \sum_{l=0}^{3}(\bigtriangledown_{depth}^{2}+\bigtriangledown_{backwardFlow}^{2}+\bigtriangledown_{forwardFlow}^{2}) $
### 4. reconstruction loss for image
$ L_{reconimg} = MSE(reconstructedframet, framet) $
### 5. entropy loss for slot output mask
$ L_{entropy} = \sum_{l=0}^{k}-(Mask_{slot}^{l} \cdot log(Mask_{slot}^{l}+eps)) $
