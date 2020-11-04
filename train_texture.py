import argparse
import numpy as np
import os
import tensorboardX
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.uv_dataset import UVDataset
from model.texture import Texture

parser = argparse.ArgumentParser()
parser.add_argument('--texturew', type=int, default=640)
parser.add_argument('--textureh', type=int, default=480)
parser.add_argument('--texture_dim', type=int, default=16)
parser.add_argument('--use_pyramid', type=bool, default=True)
parser.add_argument('--view_direction', type=bool, default=False)
parser.add_argument('--data', type=str, default="/home/lukas/Desktop/0128667499bc73c869df6b20a2d4fe26", help='directory to data')
parser.add_argument('--checkpoint', type=str, default=".", help='directory to save checkpoint')
parser.add_argument('--logdir', type=str, default=".", help='directory to save checkpoint')
#parser.add_argument('--train', default=config.TRAIN_SET)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--cropw', type=int, default=640)
parser.add_argument('--croph', type=int, default=480)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--betas', type=str, default='0.9, 0.999')
parser.add_argument('--l2', type=str, default='0.01, 0.001, 0.0001, 0')
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--load_step', type=int, default=0)
parser.add_argument('--epoch_per_checkpoint', type=int, default=50)
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, original_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 3:
        lr = original_lr * 0.33 * epoch
    elif epoch < 5:
        lr = original_lr
    elif epoch < 10:
        lr = 0.1 * original_lr
    else:
        lr = 0.01 * original_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    named_tuple = time.localtime()
    time_string = time.strftime("%m_%d_%Y_%H_%M", named_tuple)
    log_dir = os.path.join(args.logdir, time_string)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tensorboardX.SummaryWriter(logdir=log_dir)

    checkpoint_dir = args.checkpoint + time_string
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    dataset = UVDataset(args.data, args.croph, args.cropw, False)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    if args.load:
        print('Loading Saved Model')
        model = torch.load(os.path.join(args.checkpoint, args.load))
        step = args.load_step
    else:
        model = Texture(args.texturew, args.textureh, 3, use_pyramid=args.use_pyramid)
        step = 0

    l2 = args.l2.split(',')
    l2 = [float(x) for x in l2]
    betas = args.betas.split(',')
    betas = [float(x) for x in betas]
    betas = tuple(betas)
    optimizer = Adam([
        {'params': model.layer1, 'weight_decay': l2[0]},
        {'params': model.layer2, 'weight_decay': l2[1]},
        {'params': model.layer3, 'weight_decay': l2[2]},
        {'params': model.layer4, 'weight_decay': l2[3]}],
        lr=args.lr, betas=betas, eps=args.eps)
    model = model.to('cuda')
    model.train()
    torch.set_grad_enabled(True)
    criterion = nn.L1Loss()

    print('Training started')
    for i in range(1, 1+args.epoch):
        print('Epoch {}'.format(i))
        adjust_learning_rate(optimizer, i, args.lr)
        for samples in dataloader:
            images, uv_maps, masks = samples
            step += images.shape[0]
            optimizer.zero_grad()
            preds = model(uv_maps.cuda()).cpu()

            preds = torch.masked_select(preds, masks)
            images = torch.masked_select(images, masks)
            loss = criterion(preds, images)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', loss.item(), step)
            print('loss at step {}: {}'.format(step, loss.item()))

    # save checkpoint
    print('Saving checkpoint')
    torch.save(model, args.checkpoint+time_string+'/epoch_{}.pt'.format(i))

if __name__ == '__main__':
    main()
