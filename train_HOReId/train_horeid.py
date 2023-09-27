# -------------------------------------------------------------------------------
# Description:  主文件，包含训练、测试等代码，参数的设置
# Reference:
# Author: Sophia
# Date:   2021/7/6
# -------------------------------------------------------------------------------

from util.utils import *
from ops_argparse import ops
import torch
from dataloader.data_loader_instance import LoaderHoreid
from core.evaluations import visualize_ranked_images
from HOReID import Horeid
import warnings

warnings.filterwarnings('ignore')

def main():
    # init ops
    config = ops()

    # init dataloader and model
    dataloader = LoaderHoreid(config)
    config.num_train_pids = dataloader.dataset.num_train_pids
    model = Horeid(config)

    if config.mode == 'train':

        start_train_epoch = 0  # resume model from the resume_train_epoch
        if config.auto_resume_training_from_lastest_steps:
            # automatically resume model from the latest one
            root, _, files = os_walk(config.save_model_path)
            if len(files) > 0:
                # get indexes of saved models
                indexes = []
                for file in files:
                    indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
                indexes = sorted(list(set(indexes)))
                # resume model from the lastest model
                model.resume_model_from_path(config.save_model_path, indexes[-1])
                start_train_epoch = indexes[-1]
                model.logger(
                    'Time: {}, automatically resume training from the latest step (model {})'.format(time_now(),
                                                                                                     start_train_epoch))
        train_time = 0
        for cur_epoch in range(start_train_epoch, config.total_train_epochs):
            model.lr_scheduler.step(cur_epoch)  # 从一开始就执行自定义学习率动态调整计划
            start_train_time = time.time()
            _, results = train_an_epoch(cur_epoch, dataloader, model, config)
            model.logger('Time: {}; Epoch: {}; {}'.format(time_now(), cur_epoch, results))
            train_time += round(time.time() - start_train_time)
            # test
            if (cur_epoch + 1) > config.start_eval and config.eval_step > 0 and \
                    (cur_epoch + 1) % config.eval_step == 0 or \
                    (cur_epoch + 1) == config.total_train_epochs:
                print("==> Test")
                tstwithVer2(model, config, dataloader, use_gcn=True, use_gm=True)
            # save model  sophia
            if cur_epoch + 1 > config.start_save_model_epoch and (
                    cur_epoch + 1) % config.save_model_steps == 0 or cur_epoch == config.total_train_epochs - 1:
                model.save_model(cur_epoch + 1)

    elif config.mode == 'test':
        # resume from the resume_test_epoch
        if config.resume_test_path != '' and config.resume_test_epoch != 0:
            model.resume_model_from_path(config.resume_test_path, config.resume_test_epoch)
        else:
            assert 0, 'please set resume_test_path and resume_test_epoch'
        # test
        map, rank = tstwithVer2(model, config, dataloader, use_gcn=False, use_gm=False)
        model.logger('Time: {},  base, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), map, rank))
        map, rank = tstwithVer2(model, config, dataloader, use_gcn=True, use_gm=False)
        model.logger('Time: {},  base+gcn, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), map, rank))
        map, rank = tstwithVer2(model, config, dataloader, use_gcn=True, use_gm=True)
        model.logger('Time: {},  base+gcn+gm, Dataset: Duke  \nmAP: {} \nRank: {}'.format(time_now(), map, rank))
        model.logger('')

    elif config.mode == 'visualize':
        # resume from the resume_visualize_epoch
        if config.resume_visualize_path != '' and config.resume_visualize_epoch != 0:
            model.resume_model_from_path(config.resume_visualize_path, config.resume_visualize_epoch)
            print('Time: {}, resume model from {} {}'.format(time_now(), config.resume_visualize_path,
                                                             config.resume_visualize_epoch))
        # visualization
        visualize_ranked_images(config, model, dataloader)


def train_an_epoch(cur_epoch, dataloader, model, config):
    model.set_train()  # 确保在训练模式
    meter = MultiItemAverageMeter()

    start_time = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(dataloader.trainloader):
        if config.use_gpu:
            imgs, pids = imgs.to(config.device), pids.to(config.device)
        dataload_time = time.time() - start_time
        # forward
        feature_info, cls_score_info, ver_probs, gmp_info, gmn_info, keypoints_confidence = model.forward(imgs, pids,
                                                                                                          training=True)
        feature_vector_list, gcn_feature_vector_list = feature_info
        cls_score_list, gcn_cls_score_list = cls_score_info
        ver_prob_p, ver_prob_n = ver_probs
        s_p, emb_p, emb_pp = gmp_info
        s_n, emb_n, emb_nn = gmn_info

        ### loss
        id_loss = model.compute_id_loss(cls_score_list, pids, keypoints_confidence)
        id_triplet_loss = model.compute_triplet_loss(feature_vector_list, pids)
        ### gcn loss
        gcn_id_loss = model.compute_id_loss(gcn_cls_score_list, pids, keypoints_confidence)
        gcn_triplet_loss = model.compute_triplet_loss(gcn_feature_vector_list, pids)
        ### graph matching loss ???
        s_gt = torch.eye(14).unsqueeze(0).repeat([s_p.shape[0], 1, 1]).detach().to(config.device)
        pp_loss = model.permutation_loss(s_p, s_gt)
        pn_loss = model.permutation_loss(s_n, s_gt)
        p_loss = pp_loss
        ### verification loss
        ver_loss = model.bce_loss(ver_prob_p, torch.ones_like(ver_prob_p)) + model.bce_loss(ver_prob_n,
                                                                                            torch.zeros_like(
                                                                                                ver_prob_n))
        # overall loss
        loss = id_loss + gcn_id_loss + id_triplet_loss + gcn_triplet_loss
        if cur_epoch >= config.use_gm_after:
            loss += config.weight_p_loss * p_loss + config.weight_ver_loss * ver_loss
        acc = model.compute_accuracy(cls_score_list, pids)
        gcn_acc = model.compute_accuracy(gcn_cls_score_list, pids)
        from models.model_graph_matching import analyze_ver_prob
        ver_p_ana = analyze_ver_prob(ver_prob_p, True)
        ver_n_ana = analyze_ver_prob(ver_prob_n, False)

        ### optimize
        model.optimizer.zero_grad()  # 梯度置零
        loss.backward()  # 反向传播求梯度
        model.optimizer.step()  # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用

        ### record
        meter.update({'batch_time': time.time() - start_time, 'dataload_time': dataload_time,
                      'id_loss': id_loss.data.cpu().numpy(), 'gcn_id_loss': gcn_id_loss.data.cpu().numpy(),
                      'id_triplet_loss': id_triplet_loss.data.cpu().numpy(),
                      'gcn_triplet_loss': gcn_triplet_loss.data.cpu().numpy(),
                      'acc': acc, 'gcn_acc': gcn_acc,
                      'ver_loss': ver_loss.data.cpu().numpy(), 'ver_p_ana': torch.tensor(ver_p_ana).data.cpu().numpy(),
                      'ver_n_ana': torch.tensor(ver_n_ana).data.cpu().numpy(),
                      'pp_loss': pp_loss.data.cpu().numpy(), 'pn_loss': pn_loss.data.cpu().numpy()})

        start_time = time.time()

    return meter.get_val(), meter.get_str()


def tstwithVer2(model, config, dataloader, use_gcn=True, use_gm=True):
    model.set_eval()

    # meters
    query_features_meter, query_features2_meter, query_pids_meter, \
    query_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()
    gallery_features_meter, gallery_features2_meter, gallery_pids_meter, \
    gallery_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()

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
                info, gcn_info = model.forward(images, pids, training=False)
                features_stage1, features_stage2 = info
                if config.use_gcn:
                    gcn_features_stage1, gcn_features_stage2 = gcn_info
                    features_stage1 = gcn_features_stage1
                    features_stage2 = gcn_features_stage2

                # save as query features
                if loader_id == 0:
                    query_features_meter.update(features_stage1.data.cpu().numpy())
                    query_features2_meter.update(features_stage2.data.cpu().numpy())
                    query_pids_meter.update(pids.cpu().numpy())
                    query_cids_meter.update(pids.cpu().numpy())
                elif loader_id == 1:
                    gallery_features_meter.update(features_stage1.data.cpu().numpy())
                    gallery_features2_meter.update(features_stage2.data.cpu().numpy())
                    gallery_pids_meter.update(pids.cpu().numpy())
                    gallery_cids_meter.update(pids.cpu().numpy())
    query_features = query_features_meter.get_val()
    query_features2 = query_features2_meter.get_val()
    gallery_features = gallery_features_meter.get_val()
    gallery_features2 = gallery_features2_meter.get_val()

    query_info = (query_features, query_features2, query_cids_meter.get_val(), query_pids_meter.get_val())
    gallery_info = (gallery_features, gallery_features2, gallery_cids_meter.get_val(), gallery_pids_meter.get_val())

    alpha = 0.1 if use_gm else 1.0
    topk = 8
    from core.evaluations import CMCWithVer
    mAP, cmc = CMCWithVer()(query_info, gallery_info, model.verificator, model.gmnet, topk, alpha)

    return mAP, cmc


if __name__ == '__main__':
    main()  # main 函数的作用是，循环epoch，每次epoch调用一次train函数
