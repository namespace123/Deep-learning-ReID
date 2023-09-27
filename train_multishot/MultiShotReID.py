# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/8/27
# -------------------------------------------------------------------------------

import sys
from os import path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import shutil
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from util.utils import load_pretrained_model
from models.ResNet import ResNet50TP
from models.Alexnet import AlexNet
from models.model_multishot_rl.RL_model import Agent as Agent_QL
from models.model_multishot_rl.RL_model_policygradient import Agent as Agent_PG
from core.losses import CrossEntropyLabelSmooth, TripletLossMultiShotReid


class RLMultiShotReid:
    def __init__(self, config):
        self.config = config

        # init model
        self._init_base_model()
        self._init_rl_model()

    def _init_base_model(self):
        self.config.logger("Initializing base model: {}".format(self.config.arch))
        if self.config.arch.lower() == 'resnet50':
            self.base_model = ResNet50TP(num_classes=self.config.num_train_pids)
        elif self.config.arch.lower() == 'alexnet':
            self.base_model = AlexNet(num_classes=self.config.num_train_pids)
        else:
            assert False, 'unknown model: ' + self.config.arch
        # pretrain base_model loading
        if self.config.pretrained_base_model is not None:
            self.base_model = load_pretrained_model(self.base_model, self.config.pretrained_base_model)

        self.base_model = self.base_model.to(self.config.device)  # important

    def _init_rl_model(self):
        self.config.logger("Initializing rl model: {}".format(self.config.arch))
        if self.config.rl_algorithm.lower() == 'ql':
            self.config.logger('creating agent for Q learning')
            self.agent_model = Agent_QL(self.base_model, self.config)
        elif self.config.rl_algorithm.lower() == 'pg':
            self.config.logger('creating agent for Policy gradient')
            self.agent_model = Agent_PG(self.base_model, self.config)
        else:
            assert False, 'unknown rl algorithm: ' + self.config.rl_algorithm

        # pretrain model loading
        if self.config.pretrained_rl_model is not None:
            self.agent_model = load_pretrained_model(self.agent_model, self.config.pretrained_rl_model)

        self.agent_model = self.agent_model.to(self.config.device)  # important
        # torch.nn.DataParallel(self.agent_model, device_ids=self.config.device_ids) # 多gpu

    ## save model as save_epoch
    def save_model(self, save_epoch, cmc, is_best):

        # state_dict = self.agent_model.module.state_dict() if self.config.use_gpu else self.state_dict()
        state_dict = self.agent_model.state_dict()
        state = {
            'state_dict': state_dict,
            'cmc': cmc,
            'epoch': save_epoch,
        }
        fpath = osp.join(self.config.save_model_path, 'checkpoint_ep' + str(save_epoch) + '.pth.tar')
        torch.save(state, fpath)
        if is_best:
            shutil.copy(fpath, osp.join(self.config.save_model_path, 'checkpoint_ep' + str(save_epoch) + '_best.pth.tar'))

    def forward(self, images):
        return self.agent_model(images)


class MultiShotReid:
    def __init__(self, config):
        self.config = config

        # init model
        self._init_model()
        self._init_criterion()
        self._init_optimizer()

    def _init_model(self):
        self.config.logger("Initializing model: {}".format(self.config.arch))
        if self.config.arch.lower() == 'resnet50':
            self.model = ResNet50TP(num_classes=self.config.num_train_pids)
        elif self.config.arch.lower() == 'alexnet':
            self.model = AlexNet(num_classes=self.config.num_train_pids)
        else:
            assert False, 'unknown model: ' + self.config.arch
        # pretrain model loading
        if self.config.pretrained_model is not None:
            self.model = load_pretrained_model(self.model, self.config.pretrained_model)

        self.model = self.model.to(self.config.device)  # important

    # set model as train model
    def set_train(self):
        self.model = self.model.train()

    # set model as test model
    def set_eval(self):
        self.model = self.model.eval()

    # 评估标准
    def _init_criterion(self):
        self.id_criterion = CrossEntropyLabelSmooth(num_classes=self.config.num_train_pids, use_gpu=self.config.use_gpu)
        self.triplet_criterion = TripletLossMultiShotReid(self.config.triplet_margin)

    def _init_optimizer(self):

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        if self.config.stepsize > 0:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.config.stepsize,
                                                    gamma=self.config.gamma)

    ## save model as save_epoch
    def save_model(self, save_epoch, rank1, is_best):

        # state_dict = self.model.module.state_dict() if self.config.use_gpu else self.state_dict()
        state_dict = self.model.state_dict()
        state = {
            'state_dict': state_dict,
            'rank1': rank1,
            'epoch': save_epoch,
        }
        fpath = osp.join(self.config.save_model_path, 'checkpoint_ep' + str(save_epoch) + '.pth.tar')
        torch.save(state, fpath)
        if is_best:
            shutil.copy(fpath, osp.join(self.config.save_model_path, 'checkpoint_ep' + str(save_epoch) + '_best.pth.tar'))

    def forward(self, images):
        return self.model(images)
