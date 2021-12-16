CUDA_VISIBLE_DEVICES=7 python test_masks.py \
    --pretrained-pose geometric_modelsave/2021_11_30_17-KITTI_epoch2-bs_4-res_256x832-none/posenet_model_best.pth.tar \
    --pretrained-disp ./pretrain/dispnet_k.pth.tar \
    --pretrained-flow ./pretrain/pwc_net.pth.tar \
    --pretrained-slot ./pretrain/ckpt_davis.pth \
    --output-dir ./output/