# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/10/18
# -------------------------------------------------------------------------------
import sys
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from util.utils import init_path, init_device, init_logfile
import argparse
from dataloader import data_manager
import warnings

warnings.filterwarnings('ignore')

root = '/seu_share/home/liwei01/220205671/'


def stage1_ops():
    parser = argparse.ArgumentParser(
        description='Stage1 in Multi-shot Pedestrian Re-identification via Sequential Decision Making')

    # Datasets
    parser.add_argument('--root', type=str, default=root + 'dataset', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='occluded_duke', choices=data_manager.get_names(),
                        help=['market1501', 'cuhk03', 'dukemtmcreid', 'msmt17', 'occluded_duke', 'mars', 'ilidsvid',
                              'prid', 'dukemtmcvidreid'])
    parser.add_argument('-j', '--workers', default=8, type=int, help="number of data loading workers (default: 4)")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128],
                        help="[height, weight]")  # nargs='+' 表示可设置一个或多个
    # path
    parser.add_argument('--output_path', type=str, default='./results', help='path to save related informations')

    # gpu or cpu
    parser.add_argument('--use_cpu', action='store_true', help="use cpu")  # 是否只用cpu
    parser.add_argument('-g', '--gpu_devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')  # 指定某个gpu

    # model configuration
    parser.add_argument('-a', '--arch', type=str, default='resnet50', help="resnet50, alexnet")
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='model file path, need to be set for resnet3d models')

    # train configuration
    parser.add_argument('--p', type=int, default=4, help='person count in a batch')
    parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')  # batch_size=p*k
    parser.add_argument('--start_train_epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--total_train_epochs', type=int, default=800)

    # test configuration
    parser.add_argument('--resume_test_path', type=str, default='./results/models')
    parser.add_argument('--resume_test_epoch', type=int, default=90, help='0 for no resuming')
    parser.add_argument('--test_batch', default=1, type=int, help="test batch size")
    parser.add_argument('--test_num_tracks', type=int, default=16,
                        help="number of tracklets to pass to GPU during test (to avoid OOM error)")

    # Optimization options
    parser.add_argument('--optim', default='adam', type=str, help='select an optimizer')
    parser.add_argument('--lr', '--initial_learning_rate', type=float, default=0.0001,
                        help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
    parser.add_argument('--stepsize', default=200, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")  # 每 stepsize 个 epoch，就降低一次学习率
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")  # 每次学习率降低为 lr*0.1
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")  # 模型正则化的参数，使模型的复杂度成为loss的一部分
    parser.add_argument('--triplet_margin', type=float, default=0.3, help='margin for the triplet loss')

    # print
    parser.add_argument('--print-freq', type=int, default=77, help="print frequency")  # 多少个batch打印一次log
    parser.add_argument('--seed', type=int, default=1, help="manual seed")  # 控制随机参数，保证每次结果可以复现
    parser.add_argument('--resume', type=str, default='', metavar='PATH')  # 读取某个存档的checkpoint
    parser.add_argument('--eval_step', type=int, default=100,  # 训练多少个epoch进行一次测试，默认为-1，即训练完了再测试
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--start_eval', type=int, default=100,
                        help="start to evaluate after specific epoch")  # 从多少个epoch开始进行测试
    parser.add_argument('--start_save_model_epoch', type=int, default=100, help='')
    parser.add_argument('--save_model_steps', type=int, default=100, help='')

    args = parser.parse_args()

    args = init_path(args)
    args = init_device(args)
    args = init_logfile(args)

    return args


def stage2_ops():
    parser = argparse.ArgumentParser(
        description='Stage2 in Multi-shot Pedestrian Re-identification via Sequential Decision Making')

    # Datasets
    parser.add_argument('--root', type=str, default=root + 'dataset', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='occluded_duke', choices=data_manager.get_names(),
                        help=['market1501', 'cuhk03', 'dukemtmcreid', 'msmt17', 'occluded_duke', 'mars', 'ilidsvid',
                              'prid', 'dukemtmcvidreid'])
    parser.add_argument('-j', '--workers', default=8, type=int, help="number of data loading workers (default: 4)")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128],
                        help="[height, weight]")  # nargs='+' 表示可设置一个或多个
    # path
    parser.add_argument('--output_path', type=str, default='./results', help='path to save related informations')

    # gpu or cpu
    parser.add_argument('--use_cpu', action='store_true', help="use cpu")  # 是否只用cpu
    parser.add_argument('-g', '--gpu_devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')  # 指定某个gpu

    # model configuration
    parser.add_argument('-a', '--arch', type=str, default='resnet50', help="resnet50, alexnet")
    parser.add_argument('--reward_per_step', type=float, default=0.2, help="reward per step")
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--pretrained_base_model', type=str,
                        default='./results/models-resnet50-4-4-occluded-duke2/checkpoint_ep600_best.pth.tar',
                        help='model file path, need to be set for loading pretrained models')
    parser.add_argument('--pretrained_rl_model', type=str, default='./results/models-4-occluded-duke-rl/checkpoint_ep400.pth.tar',
                        help='need to be set for loading pretrained rl models')

    # train configuration
    parser.add_argument('--p', type=int, default=4, help='person count in a batch')
    parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')  # batch_size=p*k
    parser.add_argument('--rl_k', type=int, default=4, help='images count of a person in a batch')  # batch_size=p*k
    parser.add_argument('--start_train_epoch', default=400, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--total_train_epochs', type=int, default=800)
    parser.add_argument('--start_save_model_epoch', type=int, default=100, help='')
    parser.add_argument('--save_model_steps', type=int, default=100, help='')
    parser.add_argument('--num_train_iterations', default=600, type=int, help="train iterations")
    # add
    parser.add_argument('--num-train-iterations', default=1200, type=int, help="train iterations")

    # test configuration
    parser.add_argument('--resume_test_path', type=str, default='./results/models')
    parser.add_argument('--resume_test_epoch', type=int, default=90, help='0 for no resuming')
    parser.add_argument('--train_batch', default=1, type=int, help="test batch size")
    parser.add_argument('--test_batch', default=1, type=int, help="test batch size")
    parser.add_argument('--test_num_tracks', type=int, default=16,
                        help="number of tracklets to pass to GPU during test (to avoid OOM error)")

    # Optimization options
    parser.add_argument('--rl_algorithm', type=str, default='ql', help='ql - Q learning, pg - Policy gradient')
    parser.add_argument('--optim', default='adam', type=str, help='select an optimizer')
    parser.add_argument('--lr', '--initial_learning_rate', type=float, default=0.0001,
                        help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
    # stepsize 比 base_model 变大了
    parser.add_argument('--stepsize', default=400, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")  # 每 stepsize 个 epoch，就降低一次学习率
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")  # 每次学习率降低为 lr*0.1
    parser.add_argument('--weight_decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")  # 模型正则化的参数，使模型的复杂度成为loss的一部分
    parser.add_argument('--triplet_margin', type=float, default=0.3, help='margin for the triplet loss')

    # print
    parser.add_argument('--save_model_prefix', type=str, default='rl')
    parser.add_argument('--print-freq', type=int, default=77, help="print frequency")  # 多少个batch打印一次log
    parser.add_argument('--seed', type=int, default=1, help="manual seed")  # 控制随机参数，保证每次结果可以复现
    parser.add_argument('--resume', type=str, default='', metavar='PATH')  # 读取某个存档的checkpoint
    parser.add_argument('--buffer_file', type=str, default='./results/buffer-files/',
                        help='store query & gallery features')  # 存储 query 和 gallery 的特征
    parser.add_argument('--eval_step', type=int, default=100,  # 训练多少个epoch进行一次测试，默认为-1，即训练完了再测试
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--start_eval', type=int, default=100,
                        help="start to evaluate after specific epoch")  # 从多少个epoch开始进行测试

    args = parser.parse_args()

    args = init_path(args)
    args = init_device(args)
    args = init_logfile(args)

    return args
