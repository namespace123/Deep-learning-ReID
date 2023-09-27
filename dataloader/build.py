# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):
    # normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        transform = T.Compose([
            # T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.Resize([256, 128]),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.RandomHorizontalFlip(p=0.5),
            # T.Pad(cfg.INPUT.PADDING),
            T.Pad(10),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            normalize_transform,
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
    else:
        transform = T.Compose([
            # T.Resize(cfg.INPUT.SIZE_TEST),
            T.Resize([256, 128]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
