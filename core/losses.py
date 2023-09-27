# -------------------------------------------------------------------------------
# Description:
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/7
# -------------------------------------------------------------------------------
'''
标签平滑（Label Smoothing）:
是一个有效的正则化方法，可以在分类任务中提高模型的泛化能力。
其思想相当简单，即在通常的Softmax-CrossEntropy中的OneHot编码上稍作修改，
将非目标类概率值设置为一个小量，相应地在目标类上减去一个值，从而使得标签更加平滑。
'''
import torch
from torch import nn
from torch.autograd import Variable
from core.compute_distance import hard_example_mining, batch_local_dist, euclidean_dist, cosine_dist

import warnings

warnings.filterwarnings('ignore')

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['CrossEntropyLoss', 'DeepSupervision', 'CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss', 'RingLoss']


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    """
    def __init__(self, use_gpu=True):
        super(CrossEntropyLoss, self).__init__()
        self.use_gpu = use_gpu
        self.crossentropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        if self.use_gpu: targets = targets.cuda()
        loss = self.crossentropy_loss(inputs, targets)
        return loss

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduce=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduce = reduce
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = torch.zeros(log_probs.size()).scatter_(1, (targets.unsqueeze(1).data.cpu()).long(), 1)
        if self.use_gpu: targets = targets.cuda()
        # if self.use_gpu: targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        if self.reduce:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


class TripletLossMultiShotReid(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLossMultiShotReid, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())  # 该用法过期，改为以下
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        # nn.MarginRankingLoss 功能：
        # 计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0。
        # ReLU(ap - an + margin)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # d.addmm_(a1, a2, a, b) == d*a1 + a2*a*b
        # dist.addmm_(1, -2, inputs, inputs.t())  # 该用法过期，改为以下
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        # clamp做简单数值处理(为了数值的稳定性)：小于min参数的dist元素值由min值取代。
        dist = dist.clamp(min=1e-12).sqrt()  # 得到样本对之间距离

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        # 将list转换为Tensor
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_ap)
        loss = self.ranking_loss(dist_an, dist_ap, y)  # dist_an, dist_ap 顺序不能错
        return loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), 1, -2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class RingLoss(nn.Module):
    """Ring loss.

    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """

    def __init__(self, weight_ring=1.):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.weight_ring = weight_ring

    def forward(self, x):
        l = ((x.norm(p=2, dim=1) - self.radius) ** 2).mean()
        return l * self.weight_ring


class TripletLossAlignedReID(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLossAlignedReID, self).__init__()
        self.margin = margin
        # nn.MarginRankingLoss 功能：
        # 计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0。
        # ReLU(ap - an + margin)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets, local_features):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # d.addmm_(a1, a2, a, b) == d*a1 + a2*a*b
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        # clamp做简单数值处理(为了数值的稳定性)：小于min参数的dist元素值由min值取代。
        dist = dist.clamp(min=1e-12).sqrt()  # 得到样本对之间距离

        # global distance
        dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist, targets, return_inds=True)
        p_inds, n_inds = p_inds.long(), n_inds.long()
        # global losses
        y = torch.ones_like(dist_an)  # an or ap
        global_loss = self.ranking_loss(dist_an, dist_ap, y)  # dist_an, dist_ap 顺序不能错

        # local distance
        local_features = local_features.permute(0, 2, 1)
        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)
        # local loss
        local_loss = self.ranking_loss_local(local_dist_an, local_dist_ap, y)
        if self.mutual:
            return global_loss + local_loss, dist
        return global_loss, local_loss


class RankingLoss:

    def __init__(self):
        pass

    def _label2similarity(sekf, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])  # view==reshape, 行相同
        l2 = label2.view(n, 1).expand([n, m]).t()  # 行相同，转置，变为列相同
        similarity = l1 == l2  # 对比两个张量对应位置的元素是否相同
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar):  # 特征的欧式距离，标签的相似度

        # 难样本
        if more_similar == 'smaller':
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,
                                                descending=True)  # 按行从大到小
            hard_p = sorted_mat_distance[:, 0]  # 最远的正样本：-9999999系数要为0，sim=1即正样本，dist取最大
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1,
                                                descending=False)  # 按行从小到大
            hard_n = sorted_mat_distance[:, 0]  # 最近的负样本：9999999系数要为0，sim=0即负样本，dist取最小
            return hard_p, hard_n

        # 易样本
        elif more_similar == 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1,
                                                descending=False)  # 按行从小到大
            hard_p = sorted_mat_distance[:, 0]  # 最近的正样本：9999999系数要为0，sim=1即正样本，dist取最小
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, 0]  # 最远的负样本：-9999999系数要为0，sim=0即负样本，dist取最大
            return hard_p, hard_n


# TripletLoss from HOReid
class TripletLossHOReid(RankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, metric):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance
        '''
        self.margin = margin
        # nn.MarginRankingLoss：计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0。
        self.margin_loss = nn.MarginRankingLoss(margin=margin)  # 输入为3个长度为 batchsize 的一维向量
        self.metric = metric

    def __call__(self, emb1, emb2, emb3, label1, label2, label3):
        '''
        X=[x1,x2,x3,...], xi代表 one-hot 行向量 emb = [emb1,emb2,emb3,...], embi=xi * W
        此处的 emb 可以理解为 feature
        :param emb1: torch.Tensor, [m, dim]
        :param emb2: torch.Tensor, [n, dim]
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [b]
        :return:
        '''

        if self.metric == 'cosine':
            mat_dist = cosine_dist(emb1, emb2)  # 特征的欧式距离
            mat_sim = self._label2similarity(label1, label2)  # 标签的相似度
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            mat_dist = cosine_dist(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            margin_label = -torch.ones_like(hard_p)

        elif self.metric == 'euclidean':
            mat_dist = euclidean_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')  # 最远的正样本之间的距离

            mat_dist = euclidean_dist(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')  # 最近的负样本之间的距离

            margin_label = torch.ones_like(hard_p)

        # 调用 MarginRankingLoss 实现 TripletLoss
        return self.margin_loss(hard_n, hard_p, margin_label)

if __name__ == '__main__':
# import numpy as np
# target = []
# for i in range(1, 9):
#     one = (np.ones(4).astype(int) * i).tolist()
#     target.extend(one)
# print(target)
#     target = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]
#     target = torch.Tensor(target)
#     features = torch.Tensor(32, 2048)
#     a = TripletLoss()
#     print(a.forward(features, target))

    target = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]
    target = torch.Tensor(target)
    features = torch.Tensor(32, 2048)
    local_features = torch.randn(32, 128, 8)
    b = TripletLossAlignedReID()
    gl, local = b(features, target, local_features)
    print(gl.data)
    print(local.data)

# criterion = torch.nn.MarginRankingLoss(margin=0.3, reduction='mean')
# x1 = torch.Tensor([3, 2])
# x2 = torch.Tensor([1, 4])
# y = torch.Tensor([1, 2])
# loss = criterion(x1, x2, y)
# print(loss)  # tensor(2.1500)
