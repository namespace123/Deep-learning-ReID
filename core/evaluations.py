# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/7
# -------------------------------------------------------------------------------
import numpy as np
from collections import defaultdict
from sklearn import metrics as sk_metrics

import torch
import torch.nn.functional as F

from util.utils import visualize_ranked_results, time_now, NumpyCatMeter, TorchCatMeter
from core.compute_distance import cosine_dist

# for multi-shot ReID via Sequential Decision Making
def evaluateMultiShot(distmat, query_pids, gallery_pids, query_cids, gallery_cids, max_rank=50):
    query_track_num, gallery_track_num = distmat.shape
    if gallery_track_num < max_rank:
        max_rank = gallery_track_num
        print("Note: number of gallery samples is quite small, got {}".format(gallery_track_num))
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_query = 0.
    for query_index in range(query_track_num):
        # get query pid and camid
        query_pid = query_pids[query_index]
        query_cid = query_cids[query_index]

        # remove gallery samples that have the same pid and camid with query
        gallery_order = indices[query_index]
        remove = (gallery_pids[gallery_order] == query_pid) & (gallery_cids[gallery_order] == query_cid)  # 需要把同一pid同一cid的索引排除
        keep = np.invert(remove)  # 按位取反，非同一pid同一cid的索引keep值为False，其他为True

        # compute cmc curve
        orig_cmc = matches[query_index][keep]  # 最完美的情况是，orig_cmc中True值全部靠前. binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):  # np.any 存在True，则返回True；not np.any(orig_cmc)为True时表示一个True也没有
            # this condition is true when query identity does not appear in gallery
            continue  # gallery中不存在当前query_pid的照片
        num_valid_query += 1.  # 在gallery中匹配到一个query_pid的照片

        cmc = orig_cmc.cumsum()  # 求累计和
        cmc[cmc > 1] = 1  # 把 cmc 中 >=1 的值都设为 1，也就是说只有前几个数可能是 0，或者没有0都是1

        all_cmc.append(cmc[:max_rank])  # 取前 max_rank 的 cmc 加入到 all_cmc，cmc中的bool指的是，目前序号为止是否匹配到过正确pid

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]  # 最理想的情况是1全在前面，后面才递减
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel  # 对每一个成功匹配的序号执行：目前匹配成功的数量/目前为止总数量；求和；除以匹配成功的次数；得到 AP（average precise）
        all_AP.append(AP)

    assert num_valid_query > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)  # 1840*50
    all_cmc = all_cmc.sum(0) / num_valid_query
    mAP = np.mean(all_AP)

    return all_cmc, mAP

# from horeid
def accuracy(output, target, topk=[1]):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # output:(8,702), 返回每行最大值索引
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 预测结果与实际标签相比对

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval_cuhk03(distmat, query_pids, gallery_pids, query_cids, gallery_cids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    query_track_num, gallery_track_num = distmat.shape
    if gallery_track_num < max_rank:
        max_rank = gallery_track_num
        print("Note: number of gallery samples is quite small, got {}".format(gallery_track_num))
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for query_index in range(query_track_num):
        # get query pid and camid
        query_pid = query_pids[query_index]
        query_cid = query_cids[query_index]

        # remove gallery samples that have the same pid and camid with query
        order = indices[query_index]
        remove = (gallery_pids[order] == query_pid) & (gallery_cids[order] == query_cid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[query_index][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_gallery_pids = gallery_pids[order][keep]
        gallery_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_gallery_pids):
            gallery_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in gallery_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, query_pids, gallery_pids, query_cids, gallery_cids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    query_track_num, gallery_track_num = distmat.shape
    if gallery_track_num < max_rank:
        max_rank = gallery_track_num
        print("Note: number of gallery samples is quite small, got {}".format(gallery_track_num))
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for query_index in range(query_track_num):
        # get query pid and camid
        query_pid = query_pids[query_index]
        query_cid = query_cids[query_index]

        # remove gallery samples that have the same pid and camid with query
        order = indices[query_index]
        remove = (gallery_pids[order] == query_pid) & (gallery_cids[order] == query_cid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[query_index][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate(distmat, query_pids, gallery_pids, query_cids, gallery_cids, max_rank=50, use_metric_cuhk03=False):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, query_pids, gallery_pids, query_cids, gallery_cids, max_rank)
    else:
        return eval_market1501(distmat, query_pids, gallery_pids, query_cids, gallery_cids, max_rank)

class CMC:
    '''
    Compute Rank@k and mean Average Precision (mAP) scores
    Used for Person ReID
    Test on MarKet and Duke
    '''

    def __init__(self):
        pass

    def __call__(self, query_info, gallery_info, dist):

        query_feature, query_cam, query_label = query_info
        gallery_feature, gallery_cam, gallery_label = gallery_info
        assert dist in ['cosine', 'euclidean']
        print(query_feature.shape, gallery_feature.shape)

        if dist == 'cosine':
            # distance = self.cosine_dist_torch(
            #     torch.Tensor(query_feature).cuda(),
            #     torch.Tensor(gallery_feature).cuda()).data.cpu().numpy()
            distance = self.cosine_dist(query_feature, gallery_feature)
        elif dist == 'euclidean':
            # distance = self.euclidean_dist_torch(
            #     torch.Tensor(query_feature).cuda(),
            #     torch.Tensor(gallery_feature).cuda()).data.cpu().numpy()
            distance = self.euclidean_dist(query_feature, gallery_feature)

        APs = []
        CMC = []
        query_num = query_feature.shape[0]
        for i in range(query_num):
            AP, cmc = self.evaluate(
                distance[i],
                query_cam[i], query_label[i],
                gallery_cam, gallery_label, dist)
            APs.append(AP)
            CMC.append(cmc)

        mAP = np.mean(np.array(APs))

        min_len = 99999999
        for cmc in CMC:
            if len(cmc) < min_len:
                min_len = len(cmc)
        for i, cmc in enumerate(CMC):
            CMC[i] = cmc[0: min_len]
        CMC = np.mean(np.array(CMC), axis=0)

        return mAP, CMC


    def evaluate(self, distance, query_cam, query_label, gallery_cam, gallery_label, dist):

        if dist == 'cosine':
            index = np.argsort(distance)[::-1]
        elif dist == 'euclidean':
            index = np.argsort(distance)

        junk_index_1 = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)

        good_index = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
        index_wo_junk = self.notin1d(index, junk_index)

        return self.compute_AP(index_wo_junk, good_index)


    def compute_AP(self, index, good_index):
        '''
        :param index: np.array, 1d
        :param good_index: np.array, 1d
        :return:
        '''

        num_good = len(good_index)
        hit = np.in1d(index, good_index)
        index_hit = np.argwhere(hit == True).flatten()

        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index)])
        else:
            precision = []
            for i in range(num_good):
                precision.append(float(i+1) / float((index_hit[i]+1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index)])
            cmc[index_hit[0]: ] = 1

        return AP, cmc


    def in1d(self, array1, array2, invert=False):
        '''
        :param set1: np.array, 1d
        :param set2: np.array, 1d
        :return:
        '''
        mask = np.in1d(array1, array2, invert=invert)
        return array1[mask]


    def notin1d(self, array1, array2):
        return self.in1d(array1, array2, invert=True)


    def cosine_dist_torch(self, x, y):
        '''
        :param x: torch.tensor, 2d
        :param y: torch.tensor, 2d
        :return:
        '''
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return torch.mm(x, y.transpose(0, 1))


    def euclidean_dist_torch(self, mtx1, mtx2):
        """
        mtx1 is an autograd.Variable with shape of (n,d)
        mtx1 is an autograd.Variable with shape of (n,d)
        return a nxn distance matrix dist
        dist[i,j] represent the L2 distance between mtx1[i] and mtx2[j]
        """
        m = mtx1.size(0)
        p = mtx1.size(1)
        mmtx1 = torch.stack([mtx1] * m)
        mmtx2 = torch.stack([mtx2] * m).transpose(0, 1)
        dist = torch.sum((mmtx1 - mmtx2) ** 2, 2).squeeze()
        return dist


    def cosine_dist(self, x, y):
        return 1 - sk_metrics.pairwise.cosine_distances(x, y)


    def euclidean_dist(self, x, y):
        return sk_metrics.pairwise.euclidean_distances(x, y)




class CMCWithVer(CMC):
    '''
    Compute Rank@k and mean Average Precision (mAP) scores
    Used for Person ReID
    Test on MarKet and Duke
    '''


    def __call__(self, query_info, gallery_info, verificator, gmnet, topk, alpha):
        '''
        use cosine + verfication loss as distance
        '''

        query_features_stage1, query_features_stage2, query_cam, query_label = query_info
        gallery_features_stage1, gallery_features_stage2, gallery_cam, gallery_label = gallery_info

        APs = []
        CMC = []

        # compute distance
        # distance_stage1 = self.cosine_dist_torch(
        #     torch.Tensor(query_features_stage1).cuda(),
        #     torch.Tensor(gallery_features_stage1).cuda()).data.cpu().numpy()
        distance_stage1 = self.cosine_dist(query_features_stage1, gallery_features_stage1)

        #
        for sample_idnex in range(distance_stage1.shape[0]):
            a_sample_query_cam = query_cam[sample_idnex]
            a_sample_query_label = query_label[sample_idnex]

            # stage 1, compute distance, return index and topk
            a_sample_distance_stage1 = distance_stage1[sample_idnex]
            a_sample_index_stage1 = np.argsort(a_sample_distance_stage1)[::-1]
            a_sample_topk_index_stage1 = a_sample_index_stage1[:topk]

            # stage2: feature extract topk features
            a_sample_query_feature_stage2 = query_features_stage2[sample_idnex]
            topk_gallery_features_stage2 = gallery_features_stage2[a_sample_topk_index_stage1]
            a_sample_query_feature_stage2 = \
                torch.Tensor(a_sample_query_feature_stage2).cuda().unsqueeze(0).repeat([topk, 1, 1])
            topk_gallery_features_stage2 = torch.Tensor(topk_gallery_features_stage2).cuda()

            # stage2: compute verification score
            with torch.no_grad():
                _, a_sample_query_feature_stage2, topk_gallery_features_stage2 = \
                    gmnet(a_sample_query_feature_stage2, topk_gallery_features_stage2, None)
                probs = verificator(a_sample_query_feature_stage2, topk_gallery_features_stage2)
                probs = probs.detach().view([-1]).cpu().data.numpy()

            # stage2 index
            # print(a_sample_distance_stage1[a_sample_topk_index_stage1])
            # print(probs)
            # print(1-probs)
            # print('*******')
            topk_distance_stage2 = alpha * a_sample_distance_stage1[a_sample_topk_index_stage1] + (1 - alpha) * (1-probs)
            topk_index_stage2 = np.argsort(topk_distance_stage2)[::-1]
            topk_index_stage2 = a_sample_topk_index_stage1[topk_index_stage2.tolist()]
            a_sample_index_stage2 = np.concatenate([topk_index_stage2, a_sample_index_stage1[topk:]])

            #
            ap, cmc = self.evaluate(
                a_sample_index_stage2, a_sample_query_cam, a_sample_query_label, gallery_cam, gallery_label, 'cosine')
            APs.append(ap)
            CMC.append(cmc)

        mAP = np.mean(np.array(APs))

        min_len = 99999999
        for cmc in CMC:
            if len(cmc) < min_len:
                min_len = len(cmc)
        for i, cmc in enumerate(CMC):
            CMC[i] = cmc[0: min_len]
        CMC = np.mean(np.array(CMC), axis=0)

        return mAP, CMC



    def evaluate(self, index, query_cam, query_label, gallery_cam, gallery_label, dist):

        junk_index_1 = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)

        good_index = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
        index_wo_junk = self.notin1d(index, junk_index)

        return self.compute_AP(index_wo_junk, good_index)



def visualize_ranked_images(config, base, loaders, dataset):

    base.set_eval()

    # init dataset
    if dataset == 'market':
        _datasets = [loaders.market_query_samples.samples, loaders.market_gallery_samples.samples]
        _loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
        save_visualize_path = base.save_visualize_market_path
    elif dataset == 'duke':
        _datasets = [loaders.duke_query_samples.samples, loaders.duke_gallery_samples.samples]
        _loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]
        save_visualize_path = base.save_visualize_duke_path

    # compute featuress
    query_features, query_features2, gallery_features, gallery_features2 = compute_features(base, _loaders, True)

    # compute cosine similarity
    cosine_similarity = cosine_dist(
        torch.tensor(query_features).cuda(),
        torch.tensor(gallery_features).cuda()).data.cpu().numpy()

    # compute verification score
    ver_scores = compute_ver_scores(cosine_similarity, query_features2, gallery_features2, base.verificator, topk=25, sort='descend')

    # visualize
    visualize_ranked_results(cosine_similarity, ver_scores, _datasets, save_dir=save_visualize_path, topk=20, sort='descend')



def compute_features(base, loaders, use_gcn):

    # meters
    query_features_meter, query_features2_meter, query_pids_meter, query_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()
    gallery_features_meter, gallery_features2_meter, gallery_pids_meter, gallery_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()

    # compute query and gallery features
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in loader:
                # compute feautres
                images, pids, cids = data
                images, pids, cids = images.to(base.device), pids.to(base.device), cids.to(base.device)
                info, gcned_info = base.forward(images, pids, training=False)
                features_stage1, features_stage2 = info
                gcned_features_stage1, gcned_features_stage2 = gcned_info
                if use_gcn:
                    features_stage1 = gcned_features_stage1
                    features_stage2 = gcned_features_stage2
                else:
                    features_stage1 = features_stage1
                    features_stage2 = features_stage2

                # save as query features
                if loader_id == 0:
                    query_features_meter.update(features_stage1.data.cpu().numpy())
                    query_features2_meter.update(features_stage2.data.cpu().numpy())
                    query_pids_meter.update(pids.cpu().numpy())
                    query_cids_meter.update(cids.cpu().numpy())
                # save as gallery features
                elif loader_id == 1:
                    gallery_features_meter.update(features_stage1.data.cpu().numpy())
                    gallery_features2_meter.update(features_stage2.data.cpu().numpy())
                    gallery_pids_meter.update(pids.cpu().numpy())
                    gallery_cids_meter.update(cids.cpu().numpy())

    #
    query_features = query_features_meter.get_val()
    query_features2 = query_features2_meter.get_val()
    gallery_features = gallery_features_meter.get_val()
    gallery_features2 = gallery_features2_meter.get_val()

    return query_features, query_features2, gallery_features, gallery_features2


def compute_ver_scores(cosine_similarity, query_features_stage2, gallery_features_stage2, verificator, topk, sort='descend'):
    assert sort in ['ascend', 'descend']
    ver_scores_list = []
    distance_stage1 = cosine_similarity
    #
    for sample_idnex in range(distance_stage1.shape[0]):
        # stage 1, compute distance, return index and topk
        a_sample_distance_stage1 = distance_stage1[sample_idnex]
        if sort == 'descend':
            a_sample_index_stage1 = np.argsort(a_sample_distance_stage1)[::-1]
        elif sort == 'ascend':
            a_sample_index_stage1 = np.argsort(a_sample_distance_stage1)
        a_sample_topk_index_stage1 = a_sample_index_stage1[:topk]
        # stage2: feature extract topk features
        a_sample_query_feature_stage2 = query_features_stage2[sample_idnex]
        topk_gallery_features_stage2 = gallery_features_stage2[a_sample_topk_index_stage1]
        a_sample_query_feature_stage2 = \
            torch.Tensor(a_sample_query_feature_stage2).cuda().unsqueeze(0).repeat([topk, 1, 1])
        topk_gallery_features_stage2 = torch.Tensor(topk_gallery_features_stage2).cuda()

        # stage2: compute verification score
        with torch.no_grad():
            probs = verificator(a_sample_query_feature_stage2, topk_gallery_features_stage2)
            probs = probs.detach().view([-1]).cpu().data.numpy()

        ver_scores_list.append(np.expand_dims(probs, axis=0))

    ver_scores = np.concatenate(ver_scores_list, axis=0)
    return ver_scores

