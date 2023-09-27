# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/10/18
# -------------------------------------------------------------------------------
import ast
import argparse
from util.utils import *
from dataloader import data_manager
import warnings

warnings.filterwarnings('ignore')

# mars\occluded_duke:256*128
def ops():
    parser = argparse.ArgumentParser(description='Train image model with center loss')

    # Datasets
    parser.add_argument('--root', type=str, default='C://dataset', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='occluded_duke', choices=data_manager.get_names())
    parser.add_argument('-j', '--workers', default=0, type=int, help="number of data loading workers (default: 4)")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128],
                        help="[height, weight]")  # nargs='+' 表示可设置一个或多个
    parser.add_argument('--split_id', type=int, default=0, help="split index")  # train & test dataset split index
    # path
    parser.add_argument('--output_path', type=str, default='./results', help='path to save related informations')

    # gpu or cpu
    parser.add_argument('--use-cpu', action='store_true', help="use cpu")  # 是否只用cpu
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')  # 指定某个gpu

    # model configuration
    parser.add_argument('--mode', type=str, default='test', help='train, test or visualize')
    parser.add_argument('--p', type=int, default=4, help='person count in a batch')
    parser.add_argument('--k', type=int, default=2, help='images count of a person in a batch')
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    parser.add_argument('--branch_num', type=int, default=14, help='')
    parser.add_argument('--start_save_model_epoch', type=int, default=80, help='')
    parser.add_argument('--save_model_steps', type=int, default=10, help='')

    # keypoints model
    parser.add_argument('--weight_global_feature', type=float, default=1.0, help='')
    parser.add_argument('--norm_scale', type=float, default=10.0, help='')

    # gcn model
    parser.add_argument('--use_gcn', type=ast.literal_eval, default=True)
    parser.add_argument('--gcn_scale', type=float, default=20.0, help='')
    parser.add_argument('--gcn_lr_scale', type=float, default=0.1, help='')

    # graph matching model
    parser.add_argument('--use_gm_after', type=int, default=20, help='')
    parser.add_argument('--gm_lr_scale', type=float, default=1.0, help='')
    parser.add_argument('--weight_p_loss', type=float, default=1.0, help='')

    # verification model
    parser.add_argument('--weight_ver_loss', type=float, default=0.1, help='')
    parser.add_argument('--ver_lr_scale', type=float, default=1.0, help='')
    parser.add_argument('--ver_topk', type=int, default=1, help='')
    parser.add_argument('--ver_alpha', type=float, default=0.5, help='')
    parser.add_argument('--ver_in_scale', type=float, default=10.0, help='')

    # train configuration
    parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval,
                        default=True)  # 是否从读档处继续train
    parser.add_argument('--total_train_epochs', type=int, default=120)

    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    parser.add_argument('--base_learning_rate', type=float, default=0.00035)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--max_save_model_num', type=int, default=0, help='0 for max num is infinit')

    # test configuration
    parser.add_argument('--resume_test_path', type=str, default='./results/models', help=' for no resuming')
    parser.add_argument('--resume_test_epoch', type=int, default=90, help='0 for no resuming')

    # Optimization options
    parser.add_argument('--optim', default='adam', type=str, help='select an optimizer')
    parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--test-batch', default=128, type=int, help="test batch size")
    parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float, help="initial learning rate")
    parser.add_argument('--stepsize', default=60, type=int,  # 每 stepsize 个 epoch，就降低一次学习率
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--gamma', default=0.1, type=float,  # 每次学习率降低为 lr*0.1
                        help="learning rate decay")
    parser.add_argument('--weight-decay', default=5e-04, type=float,  # 模型正则化的参数，使模型的复杂度成为loss的一部分
                        help="weight decay (default: 5e-04)")
    # print
    parser.add_argument('--print-freq', type=int, default=10, help="print frequency")  # 多少个batch打印一次log
    parser.add_argument('--seed', type=int, default=1, help="manual seed")  # 控制随机参数，保证每次结果可以复现
    parser.add_argument('--resume', type=str, default='', metavar='PATH')  # 读取某个存档的checkpoint
    parser.add_argument('--eval_step', type=int, default=5,  # 训练多少个epoch进行一次测试，默认为-1，即训练完了再测试
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--start_eval', type=int, default=0,
                        help="start to evaluate after specific epoch")  # 从多少个epoch开始进行测试

    args = parser.parse_args()

    args = init_path(args)
    args = init_device(args)

    return args
