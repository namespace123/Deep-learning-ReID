# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/8/27
# -------------------------------------------------------------------------------

from util.utils import *

import torch
import torch.nn as nn
import torch.optim as optim

from models.model_keypoints import compute_local_features
from models.model_graph_matching import mining_hard_pairs

from bisect import bisect_right

from core.losses import CrossEntropyLabelSmooth, TripletLossHOReid
from models.model_graph_matching.permutation_loss import PermutationLoss
from core.evaluations import accuracy

from models.model_keypoints import ScoremapComputer

from models.model_reid import Encoder, BNClassifiers
from models.model_gcn import generate_adj, GraphConvNet
from models.model_graph_matching import GMNet, Verificator

class Horeid:
    def __init__(self, config):
        self.config = config
        # self.loaders = loaders

        # init model
        self._init_logfile()
        self._init_model()
        self._init_criterion()
        self._init_optimizer()

    def _init_logfile(self):

        if self.config.mode == 'train':
            logger_name = 'log_train_' + self.config.cur_filename + '.txt'
        elif self.config.mode == 'test':
            logger_name = 'log_test_' + self.config.cur_filename + '.txt'
        elif self.config.mode == 'visualize':
            logger_name = 'log_visualize_' + self.config.cur_filename + '.txt'

        self.logger = Logger2(os.path.join(self.config.save_logs_path, logger_name))
        self.logger("======================\nArgs:{}\n====================".format(self.config))

    def _init_model(self):
        # feature learning
        self.encoder = Encoder(class_num=self.config.num_train_pids)  # CNN backbone: ResNet50
        self.bnclassifiers = BNClassifiers(2048, self.config.num_train_pids, self.config.branch_num)
        self.bnclassifiers2 = BNClassifiers(2048, self.config.num_train_pids, self.config.branch_num)  # for gcned features

        # nn.DataParallel: train in multi gpu
        self.encoder = nn.DataParallel(self.encoder).to(self.config.device)
        self.bnclassifiers = nn.DataParallel(self.bnclassifiers).to(self.config.device)
        self.bnclassifiers2 = nn.DataParallel(self.bnclassifiers2).to(self.config.device)

        # keypoints model
        self.scoremap_computer = ScoremapComputer(self.config.norm_scale).to(self.config.device)
        self.scoremap_computer = self.scoremap_computer.eval()

        # GCN
        self.linked_edges = \
            [[13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6],
             [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12],  # global
             [0, 1], [0, 2],  # head
             [1, 2], [1, 7], [2, 8], [7, 8], [1, 8], [2, 7],  # body
             [1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12],  # libs
             # [3,4],[5,6],[9,10],[11,12], # semmetric libs links
             ]
        self.adj = generate_adj(self.config.branch_num, self.linked_edges, self_connect=0.0).to(self.config.device)
        self.gcn = GraphConvNet(self.adj, 2048, 2048, 2048, self.config.gcn_scale).to(self.config.device)

        # graph matching
        self.gmnet = GMNet().to(self.config.device)

        # verificator
        self.verificator = Verificator(self.config).to(self.config.device)

    # 评估标准
    def _init_criterion(self):
        self.id_criterion = CrossEntropyLabelSmooth(self.config.num_train_pids, use_gpu=self.config.use_gpu, reduce=False)  # reduce ???
        self.triplet_criterion = TripletLossHOReid(self.config.margin, 'euclidean')
        self.bce_loss = nn.BCELoss()
        self.permutation_loss = PermutationLoss()

    # 针对人体关键点进行打分，比如有8个id，对每个id可以进行14次打分，
    # 每个id的打分是one-hot行向量，分别对应在702个id上面的可能性
    # predict_pids_score_list：14 * 8 * 702
    # pids: (8,)   score_i: (8, 702)   weight: (8, 14)  weights[:, i]: (8,)
    def compute_id_loss(self, predict_pids_score_list, pids, weights):
        loss_all = 0
        for i, score_i in enumerate(predict_pids_score_list):
            # 当前关键点下，每个身份上的损失 (8,)
            loss_i = self.id_criterion(score_i, pids)  # id_criterion(prediction id, ground truth id)
            loss_i = (weights[:, i] * loss_i).mean()  # 乘以id上的权重，得到一个关键点的损失（单个数值）
            loss_all += loss_i
        return loss_all  # 所有关键点的loss总和（单个数值）

    def compute_triplet_loss(self, feature_list, pids):
        '''we suppose the last feature is global, and only compute its loss'''
        loss_all = 0
        for i, feature_i in enumerate(feature_list):
            if i == len(feature_list) - 1:
                loss_i = self.triplet_criterion(feature_i, feature_i, feature_i, pids, pids, pids)  # 默认得到batch中的平均loss
                loss_all += loss_i
        return loss_all  # 每个batchsize的平均loss之和

    def compute_accuracy(self, cls_score_list, pids):
        overall_acc_score = 0
        for cls_score in cls_score_list:
            overall_acc_score += cls_score
        acc = accuracy(overall_acc_score, pids, [1])[0]
        return acc

    def _init_optimizer(self):
        params = []
        lr = self.config.base_learning_rate
        weight_decay = self.config.weight_decay

        for key, value in self.encoder.named_parameters():  # 打印每一次迭代元素的名字和param
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        for key, value in self.bnclassifiers.named_parameters():  # 打印每一次迭代元素的名字和param
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        for key, value in self.bnclassifiers2.named_parameters():  # 打印每一次迭代元素的名字和param
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        for key, value in self.gcn.named_parameters():  # 打印每一次迭代元素的名字和param
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": self.config.gcn_lr_scale * lr, "weight_decay": weight_decay}]

        for key, value in self.gmnet.named_parameters():  # 打印每一次迭代元素的名字和param
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": self.config.gm_lr_scale * lr, "weight_decay": weight_decay}]

        for key, value in self.verificator.named_parameters():  # 打印每一次迭代元素的名字和param
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": self.config.ver_lr_scale * lr, "weight_decay": weight_decay}]

        self.optimizer = optim.Adam(params)
        self.lr_scheduler = WarmupMultiStepLR(self.optimizer, self.config.milestones, gamma=0.1, warmup_factor=0.01,
                                              warmup_iters=10)

    ## save model as save_epoch
    def save_model(self, save_epoch):

        torch.save(self.encoder.state_dict(),
                   os.path.join(self.config.save_model_path, 'encoder_{}.pkl'.format(save_epoch)))
        torch.save(self.bnclassifiers.state_dict(),
                   os.path.join(self.config.save_model_path, 'bnclassifiers_{}.pkl'.format(save_epoch)))
        torch.save(self.bnclassifiers2.state_dict(),
                   os.path.join(self.config.save_model_path, 'bnclassifiers2_{}.pkl'.format(save_epoch)))
        torch.save(self.gcn.state_dict(), os.path.join(self.config.save_model_path, 'gcn_{}.pkl'.format(save_epoch)))
        torch.save(self.gmnet.state_dict(),
                   os.path.join(self.config.save_model_path, 'gmnet_{}.pkl'.format(save_epoch)))
        torch.save(self.verificator.state_dict(),
                   os.path.join(self.config.save_model_path, 'verificator_{}.pkl'.format(save_epoch)))

        # if saved model is more than max num, delete the model with smallest iter
        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.config.save_model_path)
            new_file = []
            for file_ in files:
                if file_.endswith('.pkl'):
                    new_file.append(file_)
            file_iters = sorted(list(set([int(file.replace('.', '_').split('_')[-2]) for file in new_file])),
                                reverse=False)
            if len(file_iters) > self.max_save_model_num:
                for i in range(len(file_iters) - self.max_save_model_num):
                    file_path = os.path.join(root, '*_{}.pkl'.format(file_iters[i]))
                    print('remove files:', file_path)
                    os.remove(file_path)
            os.system('rm -f {}'.format(file_path))

    def resume_model_from_path(self, path, resume_epoch):
        self.encoder.load_state_dict(
            torch.load(os.path.join(path, 'encoder_{}.pkl').format(resume_epoch)))
        self.bnclassifiers.load_state_dict(
            torch.load(os.path.join(path, 'bnclassifiers_{}.pkl'.format(resume_epoch))))
        self.bnclassifiers2.load_state_dict(
            torch.load(os.path.join(path, 'bnclassifiers2_{}.pkl'.format(resume_epoch))))
        self.gcn.load_state_dict(torch.load(
            os.path.join(path, 'gcn_{}.pkl'.format(resume_epoch))))
        self.gmnet.load_state_dict(torch.load(
            os.path.join(path, 'gmnet_{}.pkl'.format(resume_epoch))))
        self.verificator.load_state_dict(
            torch.load(os.path.join(path, 'verificator_{}.pkl'.format(resume_epoch))))

    # set model as train model
    def set_train(self):
        self.encoder = self.encoder.train()
        self.bnclassifiers = self.bnclassifiers.train()
        self.bnclassifiers2 = self.bnclassifiers2.train()
        self.gcn = self.gcn.train()
        self.gmnet = self.gmnet.train()
        self.verificator = self.verificator.train()

    # set model as eval model
    def set_eval(self):
        self.encoder = self.encoder.eval()
        self.bnclassifiers = self.bnclassifiers.eval()
        self.bnclassifiers2 = self.bnclassifiers2.eval()
        self.gcn = self.gcn.eval()
        self.gmnet = self.gmnet.eval()
        self.verificator = self.verificator.eval()

    # pids:(8,)  images:(8,3,256,128) 8张图片
    def forward(self, images, pids, training):

        ''' 第一阶段：S: One-Order Semantic Module '''
        ''' 第一阶段：S: One-Order Semantic Module '''
        # feature （8,1024,16,8）
        feature_maps = self.encoder(images)  # CNN backbone: ResNet50
        with torch.no_grad():  # 不采用grad使用预训练模型
            score_maps, keypoints_confidence, _ = self.scoremap_computer(images)
        # 计算局部特征，最后一个存储全局特征
        feature_vector_list, keypoints_confidence = compute_local_features(
            self.config, feature_maps, score_maps, keypoints_confidence
        )
        # (14,8,1024),(14,8,702) 每个关键点下的批次特征&one-hot预测值
        bn_feature_vector_list, cls_score_list = self.bnclassifiers(feature_vector_list)

        ''' 第二阶段：R: High-Order Relation Module '''
        # gcn
        gcn_feature_vector_list = self.gcn(feature_vector_list)
        bn_gcn_feature_vector_list, gcn_cls_score_list = self.bnclassifiers2(gcn_feature_vector_list)

        if training:
            # mining hard samples
            new_bn_gcn_feature_vector_list, bn_gcn_feature_vector_list_p, bn_gcn_feature_vector_list_n = mining_hard_pairs(
                bn_gcn_feature_vector_list, pids)

            # graph matching
            s_p, emb_p, emb_pp = self.gmnet(new_bn_gcn_feature_vector_list, bn_gcn_feature_vector_list_p, None)
            s_n, emb_n, emb_nn = self.gmnet(new_bn_gcn_feature_vector_list, bn_gcn_feature_vector_list_n, None)

            # verificate
            ver_prob_p = self.verificator(emb_p, emb_pp)
            ver_prob_n = self.verificator(emb_n, emb_nn)

            return (feature_vector_list, gcn_feature_vector_list), \
                   (cls_score_list, gcn_cls_score_list), \
                   (ver_prob_p, ver_prob_n), \
                   (s_p, emb_p, emb_pp), \
                   (s_n, emb_n, emb_nn), \
                   keypoints_confidence

        else:
            bs, keypoints_num = keypoints_confidence.shape  # bs=8  keypoints_num=14
            # (8,14) unsqueeze(2) => (8,14,1) repeat([1,1,2048]) => (8,14,2048) view => (8,2048*14)
            keypoints_confidence = torch.sqrt(keypoints_confidence).unsqueeze(2).repeat([1, 1, 2048]).view(
                [bs, 2048 * keypoints_num]
            )
            # torch.cat(bn_feature_vector_list, dim=1): 14(8,2048)=>(8,14*2048)
            feature_stage1 = keypoints_confidence * torch.cat(bn_feature_vector_list, dim=1)
            feature_stage2 = torch.cat([i.unsqueeze(1) for i in bn_feature_vector_list], dim=1)  # (8,14,2048)
            gcn_feature_stage1 = keypoints_confidence * torch.cat(bn_gcn_feature_vector_list, dim=1)
            gcn_feature_stage2 = torch.cat([i.unsqueeze(1) for i in bn_gcn_feature_vector_list], dim=1)

            return (feature_stage1, feature_stage2), (gcn_feature_stage1, gcn_feature_stage2)


# lr 先预热（缓慢提升），再缓慢衰减，中间的阈值就是 warmup_iters
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):  # milestones应该是有序列表
            raise ValueError(
                "Milestones should be a list of increasing integers. "
                "Got {}", milestones
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted. "
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma
            ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
