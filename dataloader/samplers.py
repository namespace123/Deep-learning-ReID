# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/12
# -------------------------------------------------------------------------------
from collections import defaultdict
import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler


class ClassUniformlySampler(Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''

    def __init__(self, data_source, class_position, k):

        self.class_position = class_position
        self.k = k

        self.samples = data_source
        self.class_dict = self._tuple2dict(self.samples)  # 返回一个字典，key为类别，value为属于该类别的所有数据的索引

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''

        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict

    def _generate_list(self, dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''

        sample_list = []

        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            value = dict_copy[key]
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k
                random.shuffle(value)
                sample_list.extend(value[0: self.k])

        return sample_list


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        # img_path, pid, camid
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)  # 每个ID对应的所有图片索引
        self.pids = list(self.index_dic.keys())  # 所有的ID
        self.num_identities = len(self.pids)  # ID个数

    def __iter__(self):
        # 如：751*4=3004 [0, 3, 6, 3, 31, 543, 676, ... , 300]  每4个为一个ID
        indices = torch.randperm(self.num_identities)  # shuffle
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]  # 该ID对应的所有图片
            replace = False if len(t) >= self.num_instances else True
            # 表示从 t 的元素中任选 num_instances 个元素，replace=True表示允许重复选择
            # 如果 t 的长度不足 num_instances，则允许重复选择
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)

        return iter(ret)

    def __len__(self):
        return self.num_instances * self.num_identities  # 4*751


if __name__ == '__main__':
    from data_manager import Market1501

    dataset = Market1501(root='D://data')
    sampler = RandomIdentitySampler(dataset.train)
    a = sampler.__iter__()
    # print(a)
