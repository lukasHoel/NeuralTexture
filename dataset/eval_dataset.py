import numpy as np
import os
import torch
from torch.utils.data import Dataset

from util import map_transform

from os.path import join

from PIL import Image

class EvalDataset(Dataset):

    def __init__(self, dir, view_direction=False):
        self.dir = dir
        self.view_direction = view_direction
        self.rgb, self.uv, self.size = self.parse()

        uv_map = self.load_uv(0)
        self.height, self.width, _ = uv_map.shape

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

    def load_uv(self, idx):
        img = Image.open(self.rgb[idx], 'r')
        uv_map = Image.open(self.uv[idx], 'r')
        uv_map = uv_map.resize(img.size, Image.NEAREST)
        uv_map = np.array(uv_map) / 255.0
        # print(np.min(uv_map), np.max(uv_map))
        uv_map = uv_map.astype(np.float32)
        uv_map = uv_map[:, :, :2]
        return uv_map

    def __getitem__(self, idx):
        uv_map = self.load_uv(idx)
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')

        # final transform
        uv_map = map_transform(uv_map)
        # mask for invalid uv positions
        mask = torch.max(uv_map, dim=2)[0].le(-1.0 + 1e-6)
        mask = mask.repeat((3, 1, 1))

        if self.view_direction:
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            return uv_map, extrinsics, mask, idx
        else:
            return uv_map, mask, idx

    @staticmethod
    def _collect_fn(data, view_direction=False):
        if view_direction:
            uv_maps, extrinsics, masks, idxs = zip(*data)
            uv_maps = torch.stack(tuple(uv_maps), dim=0)
            extrinsics = torch.FloatTensor(extrinsics)
            masks = torch.stack(tuple(masks), dim=0)
            return uv_maps, extrinsics, masks, idxs
        else:
            uv_maps, masks, idxs = zip(*data)
            uv_maps = torch.stack(tuple(uv_maps), dim=0)
            masks = torch.stack(tuple(masks), dim=0)
            return uv_maps, masks, idxs

    @staticmethod
    def get_collect_fn(view_direction=False):
        collect_fn = lambda x: EvalDataset._collect_fn(x, view_direction)
        return collect_fn
