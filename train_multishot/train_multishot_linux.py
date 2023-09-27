# -------------------------------------------------------------------------------
# Description:  主文件，包含训练、测试等代码，参数的设置
# Reference:
# Author: Sophia
# Date:   2021/7/6
# -------------------------------------------------------------------------------
import sys
from os import path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from util.utils import get_features, time_now, MultiItemAverageMeter

import numpy as np
import time
import torch
from dataloader.data_loader_instance import LoaderMultiShotStage1
from core.evaluations import evaluateMultiShot
from MultiShotReID import MultiShotReid
from ops_argparse_linux import stage1_ops
import warnings

warnings.filterwarnings('ignore')


def main():
    # init ops
    config = stage1_ops()

    # init dataloader
    dataloader = LoaderMultiShotStage1(config)

    model = MultiShotReid(config)

    if config.mode == 'train':

        if config.start_train_epoch != 0:  # resume model from the resume_train_epoch
            config.logger(
                'Time: {}, resume training from the epoch (model {})'.format(time_now(), config.start_train_epoch))

        train_time = 0
        best_rank1 = -np.inf

        for epoch in range(config.start_train_epoch, config.total_train_epochs):
            if config.stepsize > 0: model.lr_scheduler.step()

            cur_epoch = epoch + 1
            start_train_time = time.time()
            _, results = train_an_epoch(dataloader, model, config)
            config.logger(
                'Time: {}; Epoch: {}/{}; {}'.format(time_now(), cur_epoch, config.total_train_epochs, results))
            train_time += round(time.time() - start_train_time)

            # test
            if (cur_epoch) >= config.start_eval and config.eval_step > 0 and \
                    (cur_epoch) % config.eval_step == 0 or \
                    (cur_epoch) == config.total_train_epochs:
                config.logger("==> Test")
                rank1 = tstwithVer(model, config, dataloader)
                is_best = rank1 >= best_rank1
                if is_best: best_rank1 = rank1
                # save model  sophia
                if cur_epoch >= config.start_save_model_epoch and (
                        cur_epoch) % config.save_model_steps == 0 or cur_epoch == config.total_train_epochs - 1:
                    model.save_model(cur_epoch, rank1, is_best)

    elif config.mode == 'test':

        # model.save_model(1, 0.4, True)
        # test
        config.logger("Evaluate only")
        tstwithVer(model, config, dataloader)


def train_an_epoch(dataloader, model, config):
    model.set_train()  # 确保在训练模式
    meter = MultiItemAverageMeter()

    start_time = time.time()

    for batch_idx, (imgs, pids, _) in enumerate(dataloader.trainloader):
        if config.use_gpu:
            imgs, pids = imgs.to(config.device), pids.to(config.device)
        dataload_time = time.time() - start_time
        # forward
        results, video_features, _ = model.forward(imgs)

        id_loss = model.id_criterion(results, pids)
        triplet_loss = model.triplet_criterion(video_features, pids)
        # overall loss
        loss = id_loss + triplet_loss

        ### optimize
        model.optimizer.zero_grad()  # 梯度置零
        loss.backward()  # 反向传播求梯度
        model.optimizer.step()  # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用

        ### record
        meter.update({'id_loss': id_loss.data.cpu().numpy(), 'triplet_loss': triplet_loss.data.cpu().numpy(),
                      'loss': loss.data.cpu().numpy(), 'batch_time': time.time() - start_time,
                      'dataload_time': dataload_time})

        start_time = time.time()

    return meter.get_val(), meter.get_str()


def tstwithVer(model, config, dataloader, ranks=[1, 5, 10, 20]):
    model.set_eval()

    config.logger('extracting query & gallery feats')
    query_features, gallery_features = [], []
    query_pids, gallery_pids = [], []
    query_cids, gallery_cids = [], []

    # init dataset
    loaders = [dataloader.queryloader, dataloader.galleryloader]

    # compute query and gallery features
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            # loader_id=0, loader=dataloader.queryloader
            # loader_id=1, loader=dataloader.galleryloader
            for data in loader:
                # compute features
                images, pids, cids = data
                images, pids, cids = images.to(config.device), pids.to(config.device), cids.to(config.device)
                # imgs = Variable(imgs, volatile=True)
                # b=1, n=number of clips, s=16

                if images.ndimension() <= 5:  # 如果images只有五维，则在第一维添加batch_size=1
                    images = images.unsqueeze(0)

                b, n, s, c, h, w = images.size()
                assert (b == 1)
                # handle chunked data
                images = images.view(b * n, s, c, h, w)
                features = get_features(model, images, config.test_num_tracks)
                # take average of features
                features = torch.mean(features, 0)
                features = features.data.cpu()  # ???

                # save as query features
                if loader_id == 0:
                    query_features.append(features)
                    query_pids.extend(pids.cpu())
                    query_cids.extend(cids.cpu())
                elif loader_id == 1:
                    gallery_features.append(features)
                    gallery_pids.extend(pids.cpu())
                    gallery_cids.extend(cids.cpu())
                torch.cuda.empty_cache()  # ???

    query_features = torch.stack(query_features)
    query_pids = np.asarray(query_pids)
    query_cids = np.asarray(query_cids)
    config.logger("Extracted features for query set, obtained {}-by-{} matrix".format(query_features.size(0),
                                                                                      query_features.size(1)))

    gallery_features = torch.stack(gallery_features)
    gallery_pids = np.asarray(gallery_pids)
    gallery_cids = np.asarray(gallery_cids)
    config.logger("Extracted features for gallery set, obtained {}-by-{} matrix".format(gallery_features.size(0),
                                                                                        gallery_features.size(1)))

    config.logger("Computing distance matrix")
    m, n = query_features.size(0), gallery_features.size(0)
    distmat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    config.logger("Computing CMC and mAP")
    cmc, mAP = evaluateMultiShot(distmat, query_pids, gallery_pids, query_cids, gallery_cids)

    config.logger("Results ----------")
    config.logger("mAP: {:.1%}".format(mAP))
    config.logger("CMC curve")
    for r in ranks:
        config.logger("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    config.logger("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()  # main 函数的作用是，循环epoch，每次epoch调用一次train函数
