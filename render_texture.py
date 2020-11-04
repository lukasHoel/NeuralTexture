import argparse
import cv2
import numpy as np
import os
from skimage import img_as_ubyte
import sys
import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset.eval_dataset import EvalDataset
from model.pipeline import PipeLine

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="/home/lukas/Desktop/0128667499bc73c869df6b20a2d4fe26", help='directory to data')
#parser.add_argument('--test', default=config.TEST_SET, help='index list of test uv_maps')
parser.add_argument('--checkpoint', type=str, default=".11_04_2020_12_54", help='directory to load checkpoint')
parser.add_argument('--load', type=str, default="epoch_50.pt", help='checkpoint name')
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--save', type=str, default=".", help='save directory')
parser.add_argument('--out_mode', type=str, default="video", choices=('video', 'image'))
parser.add_argument('--fps', type=int, default="30")
args = parser.parse_args()


if __name__ == '__main__':

    checkpoint_file = os.path.join(args.checkpoint, args.load)
    if not os.path.exists(checkpoint_file):
        print('checkpoint not exists!')
        sys.exit()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    dataset = EvalDataset(args.data, False)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=EvalDataset.get_collect_fn(False))

    model = torch.load(checkpoint_file)
    model = model.to('cuda')
    model.eval()
    torch.set_grad_enabled(False)

    if args.out_mode == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(os.path.join(args.save, 'render.mp4'), fourcc, 16,
                                     (dataset.width, dataset.height), True)
    print('Evaluating started')
    for samples in tqdm.tqdm(dataloader):
        uv_maps, masks, idxs = samples
        preds = model(uv_maps.cuda()).cpu()

        preds.masked_fill_(masks, 0) # fill invalid with 0

        # save result
        if args.out_mode == 'video':
            preds = preds.numpy()
            preds = np.clip(preds, -1.0, 1.0)
            for i in range(len(idxs)):
                image = img_as_ubyte(preds[i])
                image = np.transpose(image, (1,2,0))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                writer.write(image)
        else:
            for i in range(len(idxs)):
                image = transforms.ToPILImage()(preds[i])
                image.save(os.path.join(args.save, '{}_render.png'.format(idxs[i])))
