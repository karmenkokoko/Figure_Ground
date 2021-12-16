from __future__ import absolute_import

import os
import time
import sys
import cv2
import torch
import torchvision
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from model.model import SlotAttentionAutoEncoder
import utils.util as ut
import utils.config as cg
import torch.nn as nn
import torch.optim as optim


def main(args):
    # hyperparameter setting
    lr = args.lr 
    epsilon = 1e-5
    num_slots = args.num_slots
    iters = args.num_iterations
    batch_size = args.batch_size
    decay_step = args.decay_steps
    warmup_it = args.warmup_steps
    num_it = args.num_train_steps
    resume_path = args.resume_path
    resolution = args.resolution
    
    [logPath, modelPath, resultsPath] = cg.setup_path(args)
    writer = SummaryWriter(logdir=logPath)

    ## initialize the dataloader
    trn_dataset, resolution, in_out_channels, loss_scale, ent_scale, cons_scale = cg.setup_dataset(args)

    train_loader = ut.FastDataLoader(
        trn_dataset,
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    ## initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SlotAttentionAutoEncoder(resolution=resolution, num_slots=num_slots, in_out_channels=in_out_channels, iters=iters)
    
    model.to(device)

    ## init opt
    criterition = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## 载入保存的断点
    it = 0
    if resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        it = checkpoint['iteration']
        loss = checkpoint['loss']
    else:
        print('traning from scratch')
    
    log_freq = 100 # log train
    print('=========> start training, device use {}'.format(device))

    ## 计时器
    timestart = time.time()

    while it < num_it:
        for _, sample in enumerate(train_loader):
            
            optimizer.zero_grad()
            flow = sample['image']
            flow = flow.float().to(device)
            
            ## masks shape: [4, 2, 1, 256, 832]
            ## reconstruct flow: [4, 3, 256, 832]
            recon_flow, recons, masks, _ = model(flow)

            recon_loss = loss_scale * criterition(flow, recon_flow)
            entropy_loss = ent_scale * -(masks * torch.log(masks + epsilon)).sum(dim=1).mean()
            
            loss = recon_loss + entropy_loss

            loss.backward()
            # 更新所有参数
            optimizer.step()
            print('iteration {},'.format(it),
                  'time {:.01f}s,'.format(time.time() - timestart),
                  'loss {:.02f}.'.format(loss.detach().cpu().numpy()))

            if it % log_freq == 0:
                writer.add_scalar('Loss/total', loss.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/reconstruction', recon_loss.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/entropy', entropy_loss.detach().cpu().numpy(), it)
                mask_viz = (255*torch.cat([masks, masks, masks], dim=2)).type(torch.ByteTensor)
                writer.add_image('train/masks_0_F', mask_viz[0][0], it)
                writer.add_image('train/masks_0_B', mask_viz[0][1], it)
            
            # LR warmup
            if it < warmup_it:
                ut.set_learning_rate(optimizer, lr * it / warmup_it)

            # LR decay
            if it % decay_step == 0 and it > 0:
                ut.set_learning_rate(optimizer, lr * (0.5 ** (it // decay_step)))
                ent_scale = ent_scale * 5.0
                cons_scale = cons_scale * 5.0

            it += 1
            timestart = time.time()


if __name__ == "__main__":
    parser = ArgumentParser()
    ## optimization
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_train_steps', type=int, default=5e9)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--decay_steps', type=int, default=8e4)
    parser.add_argument('--decay_rate', type=float, default=0.5)

    ## settings
    parser.add_argument('--num_slots', type=int, default=2)
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--resume_path', type=str, default=None)

    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=(256,832))
    args = parser.parse_args()
    args.inference = False
    main(args)
