# -------------------------------------------------------------------------------
# Description:  主文件，包含训练、测试等代码，参数的设置
# Description:  度量学习 ReID 代码实践
# Description:  softmax_loss表示分类损失，triplet_loss表示triplet hard损失
# Reference:
# Author: Sophia
# Date:   2021/7/6
# -------------------------------------------------------------------------------
import os
import sys
import time
import argparse
import os.path as osp
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import models
from dataloader import data_manager
from dataloader.data_loader import ImageDataset
from dataloader import transforms as T

from core.losses import CrossEntropyLabelSmooth, TripletLoss
from util.utils import AverageMeter, Logger, save_checkpoint
from core.evaluations import evaluate
from dataloader.samplers import RandomIdentitySampler
from core.optimizers import init_optim

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Train image model with center loss')
# Datasets
parser.add_argument('--root', type=str, default='F:\\Sophia\\dataset\\', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', default='adam', type=str, help='select an optimizer')
parser.add_argument('--max-epoch', default=150, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=50, type=int,  # 每 stepsize 个 epoch，就降低一次学习率
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,  # 每次学习率降低为 lr*0.1
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,  # 模型正则化的参数，使模型的复杂度成为loss的一部分
                    help="weight decay (default: 5e-04)")
# triplet hard loss
parser.add_argument('--margin', type=float, default=0.3, help='margin for triplet loss')
parser.add_argument('--num-instances', type=int, default=4,
                    help='number of instances per identity')  # 要保持: batch-size % num-instances == 0
# 默认是分类损失和trilplet hard损失一起用
parser.add_argument('--htri-only', action='store_true', default=False,
                    help='if this is True, only htri loss is used in training')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())  # 选择的模型
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")  # 多少个batch打印一次log
parser.add_argument('--seed', type=int, default=1, help="manual seed")  # 控制随机参数，保证每次结果可以复现
parser.add_argument('--resume', type=str, default='', metavar='PATH')  # 读取某个存档的checkpoint
parser.add_argument('--evaluate', action='store_true', help="evaluation only")  # 训练还是测试，默认是训练，即该项关闭
parser.add_argument('--eval-step', type=int, default=5,  # 训练多少个epoch进行一次测试，默认为-1，即训练完了再测试
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")  # 从多少个epoch开始进行测试
parser.add_argument('--save-dir', type=str, default='log')  # 存储log的路径
parser.add_argument('--use-cpu', action='store_true', help="use cpu")  # 是否只用cpu
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')  # 指定某个gpu

args = parser.parse_args()

'''
use_gpu sys.stdout 
dataset transform_train transform_test
trainloader queryloader galleryloader
'''


def main():
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:  # 若只使用cpu
        use_gpu = False
    pin_memory = True if use_gpu else False  # pytorch dataloader 用于节约显存, CPU&GPU共用一个指针

    filename = osp.split(osp.abspath(sys.argv[0]))[1].split('.')[0]
    cur_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    if not args.evaluate:  # 如果是训练阶段，则打印日志到train日志；否则打印到tese日志
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train_' + filename + cur_time_str + '.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test_' + filename + cur_time_str + '.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True  # 把cudnn库用上，是卷积计算得更快
        torch.cuda.manual_seed_all(args.seed)  # # 为所有GPU设置种子用于生成随机数，以使得结果是确定的
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # 初始化dataset
    dataset = data_manager.init_img_dataset(name=args.dataset, root=args.root, split_id=args.split_id,
                                            cuhk03_labeled=args.cuhk03_labeled,
                                            cuhk03_classic_split=args.cuhk03_classic_split)
    # dataloader & augmentation train query gallery
    transform_train = T.Compose([T.Random2DTranslation(args.height, args.width),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor(),
                                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 约定俗成
    # test不需要水平镜像增广，但需要resize与train大小统一
    transform_test = T.Compose([T.Resize((args.height, args.width)),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 约定俗成

    # drop_last: 训练时是否不处理多余数据，比如100个数据，batch为32，三个batch之后还剩4个数据，可选择性放弃处理，目的是为了使batch_size不变
    # total:12936, batch=32, len(trainloader)=12936/32=404……9
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),  # 有这个就不用shuffle了
        batch_size=args.train_batch, num_workers=args.workers,
        # shuffle=True,
        pin_memory=pin_memory, drop_last=True
    )
    # 测试的时候所有数据都要处理，所以 drop_last=False
    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, num_workers=args.workers,
        shuffle=False,
        pin_memory=pin_memory, drop_last=False
    )
    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, num_workers=args.workers,
        shuffle=False,
        pin_memory=pin_memory, drop_last=False
    )

    # 构建模型
    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'softmax', 'metrics'})
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))  # numel()函数：返回数组中元素的个数

    criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_metric = TripletLoss(margin=args.margin)
    # model.parameters() 表示训练所有的权重，model.conv1表示只训练指定conv1层的权重
    # 如果更新指定多层的权重：params=nn.Sequential([model.conv1, model.conv2])
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)  # ==torch.optim.adam

    # 学习率衰减，方式有很多：阶梯型衰减，震荡型衰减，平滑型衰减
    if args.stepsize > 0:  # 此处采用阶梯型衰减
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    start_epoch = args.start_epoch

    if args.resume:  # 若要恢复模型，则读档
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    # if use_gpu:
    # 采用本方法之后，模型权重要改为这样表示：model.module.parameter()
    # model = nn.DataParallel(model).cuda()  # 用pytorch的并行库，但机器多gpu训练

    if args.evaluate:
        print('Evaluate only!')
        tst(model, queryloader, galleryloader, use_gpu)
        return 0

    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        if args.stepsize > 0:
            scheduler.step()  # 学习率衰减，需要执行这一步才能起效
        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            # state_data:模型参数，rank1:当前模型准确度，epoch:训练周期
            rank1 = tst(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
                # if use_gpu:
                #     state_dict = model.module.state_dict()
                # else:
                #     state_dict = model.state_dict()
                state_dict = model.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best,
                    osp.join(args.save_dir, 'checkpoint' + cur_time_str, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

            print("==> Best Rank-1 {:.1%}, achieve at epoch {}".format(best_rank1, best_epoch))


def train(epoch, model, criterion_class, criterion_metric, optimizer, trainloader, use_gpu):
    model.train()  # 确保在训练模式

    losses = AverageMeter()
    softmax_losses = AverageMeter()
    triplet_losses = AverageMeter()
    batch_time = AverageMeter()  # 计算每个batch所用时间
    data_time = AverageMeter()  # 读取每个batch所用时间

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        outputs, features = model(imgs)  # 输出
        softmax_loss = criterion_class(outputs, pids)  # 算分类损失
        triplet_loss = criterion_metric(features, pids)  # 算相似度损失
        loss = softmax_loss + triplet_loss
        optimizer.zero_grad()  # 梯度置零
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))
        softmax_losses.update(softmax_loss.item(), pids.size(0))
        triplet_losses.update(triplet_loss.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:  # print-freq==10
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                  'TLoss {triplet_loss.val:.4f} ({triplet_loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, softmax_loss=softmax_losses, triplet_loss=triplet_losses))


def tst(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()  # main 函数的作用是，循环epoch，每次epoch调用一次train函数
