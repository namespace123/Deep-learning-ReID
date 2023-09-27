# -------------------------------------------------------------------------------
# Description:  一些可能遇到的功能性代码
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/6
# -------------------------------------------------------------------------------

from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import time

import torch

from PIL import Image, ImageOps, ImageDraw
import numpy as np
from more_itertools import chunked

import torch.backends.cudnn as cudnn

# for multi-shot ReID via Sequential Decision Making
# 将 imgs 分批传入 model 得到图像特征后，整合得到所有图像特征的张量形式
# 该步骤的目的就是给图像分批提取特征，减轻gpu压力
def get_features(model, imgs, test_frames_num):
    """to handle higher seq length videos due to OOM error
    specifically used during test

    Arguments:
        model -- model under test
        imgs -- imgs to get features for

    Returns:
        features
    """

    # handle chunked data
    all_features = []
    # print(imgs.shape)
    for test_imgs in chunked(imgs, test_frames_num):  # 每次迭代 test_num_tracks 个元素
        cur_test_imgs = torch.stack(test_imgs)
        num_cur_test_imgs = cur_test_imgs.shape[0]
        video_features, _ = model.forward(cur_test_imgs)  # (test_frames_num, 2048) 每个图像特征
        # print(current_test_imgs.shape, video_features.shape)
        video_features = video_features.view(num_cur_test_imgs, -1)
        all_features.append(video_features)

    all_features = torch.cat(all_features)  # 所有图像特征
    # print(all_features.shape)
    return all_features

# for multi-shot ReID via Sequential Decision Making
# 将 pretrained_model 中与 model 同名的变量值更新到 model 中，并返回 model
def load_pretrained_model(model, pretrained_model_path):
    '''To load the pretrained model considering the number of keys and their sizes

    Arguments:
        model {loaded model} -- already loaded model
        pretrained_model_path {str} -- path to the pretrained model file

    Raises:
        IOError -- if the file path is not found

    Returns:
        model -- model with loaded params
    '''

    if not os.path.exists(pretrained_model_path):
        raise IOError("Can't find pretrained model: {}".format(pretrained_model_path))

    print("Loading checkpoint from '{}'".format(pretrained_model_path))
    pretrained_state = torch.load(pretrained_model_path)['state_dict']
    print(len(pretrained_state), ' keys in pretrained model')

    current_model_state = model.state_dict()
    print(len(current_model_state), ' keys in current model')
    pretrained_state = {key: val
                        for key, val in pretrained_state.items()
                        if key in current_model_state and val.size() == current_model_state[key].size()}

    print(len(pretrained_state), ' keys in pretrained model are available in current model')
    current_model_state.update(pretrained_state)
    model.load_state_dict(current_model_state)
    return model

def init_logfile(args):
    logger_name = 'log_' + args.mode + '_' + args.cur_filename + '_' + args.cur_time_str + '.txt'
    args.logger = Logger2(osp.join(args.save_logs_path, logger_name))
    args.logger("=======================\nArgs:{}\n=======================".format(args))
    return args

def init_path(args):
    # log file
    args.cur_filename = osp.split(osp.abspath(sys.argv[0]))[1].split('.')[0]
    args.cur_time_str = time_now_to_log()
    args.save_logs_path = osp.join(args.output_path, 'logs')  # 存储日志的完整路径
    args.save_model_path = osp.join(args.output_path, "models")  # 存储预训练模型的完整路径
    # create dir
    make_dir(args.save_logs_path)
    make_dir(args.save_model_path)
    # dataset path
    args.dataset_path = osp.join(args.root, args.dataset)

    return args

def init_device(args):
    # gpu or cpu
    # args.use_gpu = torch.cuda.is_available()  # 是否存在gpu
    args.use_gpu = False if args.use_cpu else True  # 若指定使用cpu，则不用gpu
    args.pin_memory = True if args.use_gpu else False  # pytorch dataloader 用于节约显存, CPU&GPU共用一个指针
    if args.use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True  # 把cudnn库用上，使卷积计算得更快
        torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置种子用于生成随机数，以使得结果是确定的
    else:
        torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
        print("Currently using CPU (GPU is highly recommended)")

    args.device = torch.device("cuda" if args.use_gpu else "cpu")

    return args


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# 存储累计值
class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# from horeid
class NumpyCatMeter:
    '''
    Concatenate Meter for np.array
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = np.concatenate([self.val, val], axis=0)

    def get_val(self):
        return self.val


class TorchCatMeter:
    '''
    Concatenate Meter for torch.Tensor
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = torch.cat([self.val, val], dim=0)

    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.data.cpu().numpy()


class MultiItemAverageMeter:

    def __init__(self):
        self.content = {}

    def update(self, val):
        '''
        :param val: dict, keys are strs, values are torch.Tensor or np.array
        '''
        for key in list(val.keys()):
            value = val[key]
            if key not in list(self.content.keys()):
                self.content[key] = {'avg': value, 'sum': value, 'count': 1.0}
            else:
                self.content[key]['sum'] += value
                self.content[key]['count'] += 1.0
                self.content[key]['avg'] = self.content[key]['sum'] / self.content[key]['count']

    def get_val(self):
        keys = list(self.content.keys())
        values = []
        for key in keys:
            try:
                values.append(self.content[key]['avg'].data.cpu().numpy())
            except:
                values.append(self.content[key]['avg'])
        return keys, values

    def get_str(self):

        result = ''
        keys, values = self.get_val()

        for key, value in zip(keys, values):
            result += key
            result += ': '
            result += str(value)
            result += ';  '

        return result


# 模型存档
def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


# from horeid
class Logger2:

    def __init__(self, logger_path):
        self.logger_path = logger_path

    def __call__(self, input):
        input = str(input)
        with open(self.logger_path, 'a') as f:
            f.writelines(input + '\n')
        print(input)


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def time_now_to_log():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def make_dir(dir):
    if not osp.exists(dir):
        os.makedirs(dir)
        print('Successfully make dir: {}'.format(dir))
    else:
        print('Existed dir: {}'.format(dir))


def label2similarity(label1, label2):
    '''
    compute similarity matrix of label1 and label2
    :param label1: torch.Tensor, [m]
    :param label2: torch.Tensor, [n]
    :return: torch.Tensor, [m, n], {0, 1}
    '''
    m, n = len(label1), len(label2)
    l1 = label1.view(m, 1).expand([m, n])
    l2 = label2.view(n, 1).expand([n, m]).t()
    similarity = l1 == l2
    return similarity


# from horeid
def visualize_ranked_results(distmat1, distmat2, dataset, save_dir='', topk=20, sort='ascend'):
    """Visualizes ranked results.
    Supports both image-reid and video-reid.
`    Args:
        distmat1 (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
    """

    num_q, num_g = distmat1.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    assert sort in ['descend', 'ascend']

    if sort == 'ascend':
        indices = np.argsort(distmat1, axis=1)
    elif sort == 'descend':
        indices = np.argsort(distmat1, axis=1)[:, ::-1]

    make_dir(save_dir)

    def cat_imgs_to(image_list, hit_list, text_list, text2_list, target_dir):

        images = []
        for img, hit, text, text2 in zip(image_list, hit_list, text_list, text2_list):
            img = Image.open(img).resize((64, 128))
            d = ImageDraw.Draw(img)
            d.text((3, 1), "{:.3}".format(text), fill=(255, 255, 0))
            d.text((3, 10), "{:.3}".format(text2), fill=(255, 255, 0))
            if hit:
                img = ImageOps.expand(img, border=4, fill='green')
            else:
                img = ImageOps.expand(img, border=4, fill='red')
            images.append(img)

        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(target_dir)

    counts = 0
    for q_idx in range(num_q):
        flag = True

        image_list = []
        hit_list = []
        text_list = []
        text2_list = []

        # query image
        qimg_path, qpid, qcamid = query[q_idx]
        image_list.append(qimg_path)
        hit_list.append(True)
        text_list.append(0.0)
        text2_list.append(0.0)

        # target dir
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))

        # matched images
        rank_idx = 1
        for ii, g_idx in enumerate(indices[q_idx, :]):
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid and qcamid == gcamid) or (gpid == -1 or gpid == 0)
            if not invalid:
                if rank_idx == 1 and qpid == gpid:
                    flag = False
                image_list.append(gimg_path)
                hit_list.append(qpid == gpid)
                text_list.append(distmat1[q_idx, g_idx])
                try:
                    text2_list.append(distmat2[q_idx, ii])
                except:
                    text2_list.append(0.0)
                rank_idx += 1
                if rank_idx > topk:
                    break

        if flag:
            counts += 1
            cat_imgs_to(image_list, hit_list, text_list, text2_list, qdir)
            print(counts, qdir)
