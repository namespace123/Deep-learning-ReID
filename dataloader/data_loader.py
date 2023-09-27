# -------------------------------------------------------------------------------
# Description:  有了数据之后，如何去反复迭代地去吐数据
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/6
# -------------------------------------------------------------------------------
import os
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset


def read_img(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist.".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("{} was not read successfully.".format(img_path))
            pass
    return img


# 重构一下 torch的Dataset
class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    # 获取dataset中的一条数据：图像，pid，camid
    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_img(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all', 'dense']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]  #
        num = len(img_paths)

        if self.sample == 'random':
            """
                Randomly sample seq_len consecutive frames from num frames,
                if num is smaller than seq_len, then replicate items.
                This sampling strategy is used in training phase.
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            try:
                indices = np.random.choice(indices, size=self.seq_len, replace=replace)  # replace:True表示可以取相同数字，False表示不可以取相同数字
            except RuntimeError:
                print("error!")
            # sort indices to keep temporal order
            # comment it to be order-agnostic
            indices = np.sort(indices)
        elif self.sample == 'evenly':  # 等间隔均匀选取
            """Evenly sample seq_len items from num items."""
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num / self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs to be set to 1.
            """
            indices = np.arange(num)
        elif self.sample == 'dense':  # from code of MultiShot， but it was not used
            """ 
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = list(range(num))
            indices_list = []
            # 把所有frames分成 len(frames)/seq_len 份，每份为一个sequence
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len

            last_seq = frame_indices[cur_index:]
            # 最后剩下的sequence如果数量不足seq_len，则补齐
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)

            imgs_list = []
            for indices in indices_list:  # 遍历每一个 sequence
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_img(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                # imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            # imgs_array 由 所有 sequence 组成
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[index]
            img = read_img(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid

# for multi-shot ReID via Sequential Decision Making Stage2
class PairVideoDataset(VideoDataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    # 这里的dataset是一系列pid-cid对应的轨迹片段，每个轨迹片段中储存着一系列同一pid-cid的frames
    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        super().__init__(dataset=dataset, seq_len=seq_len, sample=sample, transform=transform)

        # in dataset, each instance contains img paths, person id, cam id
        # arrange the details based on pid for pairwise training
        self.pid_based_dataset = self.arrange_based_on_pid()  # 构建 pid-cid-轨迹index，删除同一ID在同一Camera下只出现一次的图像
        self.pids = list(self.pid_based_dataset.keys())
        print('In total ', len(self.pids), ' persons')

    def __len__(self):
        # for each person, consider positive and negative pairs
        return len(self.pids) * 2

    def __getitem__(self, index):  # index为偶数，则取相同pid，不同camera；index为基数，则取不同pid
        # select the person
        # print(len(self.pids), index, index//2)
        pid1 = self.pids[index // 2]
        camid1 = np.random.choice(list(self.pid_based_dataset[pid1].keys()))

        # even indices are of same pid, odd indices are different pid
        if index % 2 == 0:
            # select same pid frames
            pid2 = pid1
            list_except_camid1 = list(self.pid_based_dataset[pid1].keys())
            # print(pid1, list_except_camid1)
            list_except_camid1.remove(camid1)
            # print(pid1, list_except_camid1)
            camid2 = np.random.choice(list_except_camid1)
            assert (pid1 == pid2) and (camid1 != camid2)
        else:
            # select different pid frames
            list_except_pid1 = list(self.pids)
            list_except_pid1.remove(pid1)
            pid2 = np.random.choice(list_except_pid1)
            camid2 = np.random.choice(list(self.pid_based_dataset[pid2].keys()))
            assert pid1 != pid2

        # 在该pid-cid下的众多轨迹里面任选一个轨迹
        person1_track = np.random.choice(self.pid_based_dataset[pid1][camid1])
        person2_track = np.random.choice(self.pid_based_dataset[pid2][camid2])
        # 在该轨迹下取出seq_len，即k个frames，此处默认sample='evenly'，表示等间隔均匀选取
        person1_track, pid1_returned, camid1_returned = super().__getitem__(person1_track)
        person2_track, pid2_returned, camid2_returned = super().__getitem__(person2_track)

        # check again
        assert pid1_returned == pid1, 'pid1 equality assertion failed'
        assert camid1_returned == camid1, 'camid1 equality assertion failed'
        assert pid2_returned == pid2, 'pid2 equality assertion failed'
        assert camid2_returned == camid2, 'camid2 equality assertion failed'

        return person1_track, pid1, person2_track, pid2

    def arrange_based_on_pid(self):
        pid_based_dataset = {}

        #  for each instance of the dataset, arrange it based on pid
        # dataset 中有许多轨迹片段，将其中同一pid同一cid的片段归拢到一起
        for index in range(len(self.dataset)):
            img_paths, pid, camid = self.dataset[index]  # 单条pid-cid轨迹片段

            # init empty buffer if key does not exist
            if pid not in pid_based_dataset:
                pid_based_dataset[pid] = {}

            if camid not in pid_based_dataset[pid]:
                pid_based_dataset[pid][camid] = []

            # store the indices in the buffer
            # 把同一pid-cid对应的不同轨迹片段都存储到 pid_based_dataset[pid][camid] 中
            pid_based_dataset[pid][camid].append(index)

        # remove the persons that appear in only one cam
        only_one_cam_persons = []
        for pid in pid_based_dataset.keys():
            cams = list(pid_based_dataset[pid].keys())
            if len(cams) == 1:
                only_one_cam_persons.append(pid)

        print('these IDs are removed from training', only_one_cam_persons)
        for pid in only_one_cam_persons:
            del pid_based_dataset[pid]

        return pid_based_dataset


# if __name__ == '__main__':
    # import data_manager
    #
    # dataset = data_manager.Market1501(root="F://Sophia/dataset/", name="market1501")
    # train_loader = ImageDataset(dataset.train)
    # for batch_id, (imgs, pid, camid) in enumerate(train_loader):
    #     imgs.save('../images/sample.png')
