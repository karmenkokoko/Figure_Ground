CUDA_VISIBLE_DEVICES=3 python ./train.py --batch_size 4 --epoches 10 \
    --rigid-reconstruction-loss 1.0 --flow-reconstruction-loss 0.2 \
    --smoothness-loss 0.0 --slot-recon-loss 0 --entropy-loss 0 \
    --consistency-loss 0  --pretrained-pose geometric_modelsave/2021_11_30_17-KITTI_epoch2-bs_4-res_256x832-none/posenet_model_best.pth.tar \
    --pretrained-disp ./pretrain/dispnet_k.pth.tar \
    --pretrained-flow ./pretrain/pwc_net.pth.tar \
    --pretrained-slot ./pretrain/ckpt_davis.pth \
    --fix-slotnet --fix-flownet --fix-dispnet \
    --valid-depth-gt
    