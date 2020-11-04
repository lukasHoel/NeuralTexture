import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

from util import augment

from os.path import join


class UVDataset(Dataset):

    def __init__(self, dir, H, W, view_direction=False):
        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

        self.rgb, self.uv, self.size = self.parse()

    def parse(self):
        rgb_images = []
        uv_maps = []

        images_path = join(self.dir, "images")
        if os.path.exists(images_path):
            files = os.listdir(images_path)
            colors = sorted([join(images_path, f) for f in files if "color" in f])
            uvs = sorted([join(images_path, f) for f in files if "uv_map" in f])

            rgb_images.extend(colors)
            uv_maps.extend(uvs)
        else:
            raise ValueError(f"dir does not exist: {images_path}")

        assert (len(rgb_images) == len(uv_maps))

        return rgb_images, uv_maps, len(rgb_images)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.open(self.rgb[idx], 'r')
        uv_map = Image.open(self.uv[idx], 'r')
        uv_map = uv_map.resize(img.size, Image.NEAREST)
        uv_map = np.array(uv_map) / 255.0
        #print(np.min(uv_map), np.max(uv_map))
        uv_map = uv_map.astype(np.float32)

        uv_map = uv_map[:,:,:2]
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        img, uv_map, mask = augment(img, uv_map, self.crop_size)
        if self.view_direction:
            # view_map = np.load(os.path.join(self.dir, 'view_normal/'+self.idx_list[idx]+'.npy'))
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            return img, uv_map, extrinsics, mask
        else:
            
            return img, uv_map, mask

if __name__ == "__main__":
    d = UVDataset(dir="/home/lukas/Desktop/0128667499bc73c869df6b20a2d4fe26", H=256, W=256, view_direction=False)

    import matplotlib.pyplot as plt
    import torchvision.transforms as tf

    for idx, (rgb, uv, mask) in enumerate(d):
        print("ITEM: ", idx)

        print(uv.shape)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(tf.ToPILImage()(rgb))
        ax[1].imshow(tf.ToPILImage()(uv.permute(2,0,1)))
        ax[2].imshow(tf.ToPILImage()(mask.float()))
        plt.show()
