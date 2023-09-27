# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/9/4
# -------------------------------------------------------------------------------
from torch.utils.data import DataLoader

from dataloader import data_manager
from dataloader.data_loader import ImageDataset, VideoDataset, PairVideoDataset
import dataloader.transforms as transforms

from dataloader.samplers import ClassUniformlySampler, RandomIdentitySampler

# dataloader for MultiShot-Stage2
class LoaderMultiShotStage2:

    def __init__(self, config):
        # 初始化dataset
        self.dataset = data_manager.init_vid_dataset(name=config.dataset, root=config.root, logger=config.logger)
        config.num_train_pids = self.dataset.num_train_pids

        if config.dataset == 'mars':
            # dataloader & augmentation (train query gallery)
            self.transform_train = transforms.Compose([
                transforms.Random2DTranslation(config.image_size[0], config.image_size[1]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # test不需要增广，但需要resize与train大小统一
            self.transform_test = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif config.dataset == 'occluded_duke':
            # dataloader & augmentation (train query gallery)
            self.transform_train = transforms.Compose([
                transforms.Resize(config.image_size, interpolation=3),
                transforms.RandomHorizontalFlip(p=0.5),  # 依据概率p对PIL图片进行水平翻转，p默认0.5
                transforms.Pad(10),  # padding=10，则上下左右均填充10个pixel
                transforms.RandomCrop(config.image_size),  # 依据给定的size随机裁剪
                transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]（归一化至[0-1]是直接除以255）
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
                transforms.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
            ])
            # test不需要增广，但需要resize与train大小统一
            self.transform_test = transforms.Compose([
                transforms.Resize(config.image_size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # 与Stage1不同的是，VideoDataset 变成了 PairVideoDataset
        # 无需sampler，因为 PairVideoDataset 已经做好了sample
        self.trainloader = DataLoader(
            # train指所有的轨迹片段，每一个轨迹片段中存放的内容是(img_paths, pid, camid)
            # 取train_batch个条数据，每条数据里面是PairVideoDataset返回的结果，即两个pid对应的长度为k的轨迹
            PairVideoDataset(self.dataset.train, transform=self.transform_train, seq_len=config.k, sample='random'),
            batch_size=config.train_batch, num_workers=config.workers,
            pin_memory=config.pin_memory, drop_last=True
        )
        self.queryloader = DataLoader(
            # seq_len=1, sample='dense'表示获取所有图像
            VideoDataset(self.dataset.query, transform=self.transform_test, seq_len=1, sample='dense'),
            batch_size=config.test_batch, num_workers=config.workers,
            shuffle=False,
            pin_memory=config.pin_memory, drop_last=False
        )
        self.galleryloader = DataLoader(
            # seq_len=1, sample='dense'表示获取所有图像
            VideoDataset(self.dataset.gallery, transform=self.transform_test, seq_len=1, sample='dense'),
            batch_size=config.test_batch, num_workers=config.workers,
            shuffle=False,
            pin_memory=config.pin_memory, drop_last=False
        )




# dataloader for MultiShot-Stage1
class LoaderMultiShotStage1:

    def __init__(self, config):
        # 初始化dataset
        self.dataset = data_manager.init_vid_dataset(name=config.dataset, root=config.root, logger=config.logger)

        config.num_train_pids = self.dataset.num_train_pids

        # dataloader & augmentation (train query gallery)
        self.transform_train = transforms.Compose([
            transforms.Random2DTranslation(config.image_size[0], config.image_size[1]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # test不需要增广，但需要resize与train大小统一
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # drop_last: 训练时是否不处理多余数据，比如100个数据，batch为32，三个batch之后还剩4个数据，可选择性放弃处理，目的是为了使batch_size不变
        # 如果 drop_last=False 且最后有多余不足数据，则这些数据组成 the last batch
        # total:12936, batch=32, len(trainloader)=12936/32=404……9
        self.trainloader = DataLoader(
            VideoDataset(self.dataset.train, transform=self.transform_train, seq_len=config.k, sample='random'),
            # 为传入的数据中的每个id选择config.k个样本
            sampler=RandomIdentitySampler(self.dataset.train, num_instances=config.k),
            # 传入的数据中第2维是类别，所以class_position=1
            batch_size=config.p * config.k, num_workers=config.workers,
            # shuffle=True, # 有了ClassUniformlySampler就不用shuffle了
            pin_memory=config.pin_memory, drop_last=True
        )
        self.queryloader = DataLoader(
            VideoDataset(self.dataset.query, transform=self.transform_test, seq_len=config.k, sample='random'),
            batch_size=config.test_batch, num_workers=config.workers,
            shuffle=False,
            pin_memory=config.pin_memory, drop_last=False
        )
        self.galleryloader = DataLoader(
            VideoDataset(self.dataset.gallery, transform=self.transform_test, seq_len=config.k, sample='random'),
            batch_size=config.test_batch, num_workers=config.workers,
            shuffle=False,
            pin_memory=config.pin_memory, drop_last=False
        )


# dataloader for HOReid
class LoaderHoreid:

    def __init__(self, config):
        # 初始化dataset
        self.dataset = data_manager.init_img_dataset(name=config.dataset, root=config.root)
        config.num_train_pids = self.dataset.num_train_pids
        # dataloader & augmentation train query gallery
        self.transform_train = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),  # 依据概率p对PIL图片进行水平翻转，p默认0.5
            transforms.Pad(10),  # padding=10，则上下左右均填充10个pixel
            transforms.RandomCrop(config.image_size),  # 依据给定的size随机裁剪
            transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]（归一化至[0-1]是直接除以255）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
            transforms.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
        # test不需要增广，但需要resize与train大小统一
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # drop_last: 训练时是否不处理多余数据，比如100个数据，batch为32，三个batch之后还剩4个数据，可选择性放弃处理，目的是为了使batch_size不变
        # 如果 drop_last=False 且最后有多余不足数据，则这些数据组成 the last batch
        # total:12936, batch=32, len(trainloader)=12936/32=404……9
        self.trainloader = DataLoader(
            ImageDataset(self.dataset.train, transform=self.transform_train),
            # 为传入的数据中的每个id选择config.k个样本
            sampler=ClassUniformlySampler(self.dataset.train, class_position=1, k=config.k),
            # 传入的数据中第2维是类别，所以class_position=1
            batch_size=config.p * config.k, num_workers=config.workers,
            # shuffle=True, # 有了ClassUniformlySampler就不用shuffle了
            pin_memory=config.pin_memory, drop_last=False
        )
        # 测试的时候所有数据都要处理，所以 drop_last=False
        self.queryloader = DataLoader(
            ImageDataset(self.dataset.query, transform=self.transform_test),
            batch_size=config.test_batch, num_workers=config.workers,
            shuffle=False,
            pin_memory=config.pin_memory, drop_last=False
        )
        self.galleryloader = DataLoader(
            ImageDataset(self.dataset.gallery, transform=self.transform_test),
            batch_size=config.test_batch, num_workers=config.workers,
            shuffle=False,
            pin_memory=config.pin_memory, drop_last=False
        )
