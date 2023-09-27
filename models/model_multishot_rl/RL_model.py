import os
import numpy as np
import torch, torch.nn as nn
import random, collections, copy
from collections import deque
from tqdm import tqdm
from aenum import Enum, MultiValue
from core.evaluations import evaluateMultiShot
from util.utils import get_features

import sys
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

MEMORY_CAPACITY = 10000
BATCH_SIZE = 128

LAMBDA = 0.0001  # speed of decay
MAX_EPSILON = 1
MIN_EPSILON = 0.01  # stay a bit curious even when getting old
GAMMA = 0.99  # discount factor


class Decision(Enum):
    _init_ = 'value fullname'
    _settings_ = MultiValue

    SAME = 0, 'SAME'
    DIFFERENT = 1, 'DIFFERENT'
    UNSURE = 2, 'UNSURE'

    def __int__(self):
        return self.value


class Environment:
    '''
    A gym-like environment for handling person-reid sequential multishot decision  making
    '''

    def __init__(self, person1, person2, reward_per_step, tmax=8):
        # print('tmax', tmax)
        self.person1 = person1
        self.person2 = person2
        # features of size #frames x #featdim  eg.(2,2048)
        self.p1_track_frames_features = self.person1['features']
        self.p2_track_frames_features = self.person2['features']
        self.pid1 = self.person1['id']
        self.pid2 = self.person2['id']
        self.current_index = 0
        self.reward_per_step = reward_per_step

        # mandatory assertions   要求：轨迹帧数 > 1，两个轨迹长度相同
        assert len(self.p1_track_frames_features) >= 1, 'length of person1 features is <= 1'
        assert len(self.p2_track_frames_features) >= 1, 'length of person2 features is <= 1'
        assert self.p1_track_frames_features.shape[1] == self.p2_track_frames_features.shape[
            1], 'features are not of same dimension'

        # features are expected to be of shape #frames x #feat_dim
        self.feat_dim = self.p1_track_frames_features.shape[1]
        # len(self.p1_track_frames_features 轨迹帧数
        self.N = min(tmax, len(self.p1_track_frames_features), len(self.p2_track_frames_features))  # 最多匹配次数
        self.abs_features = self.get_abs_features(self.p1_track_frames_features, self.p2_track_frames_features,
                                                  self.N)  # 计算特征差异
        self.abs_features_norm = self.get_norm_of_features(self.abs_features)  # 取2范数
        self.q_values_history = []

        # print(self.N, self.feat_dim, self.abs_features.shape, len(self.abs_features_norm), self.current_index, self.pid1, self.pid2)

    def get_norm_of_features(self, abs_features):
        '''to get the norm of features
        
        Arguments:
            abs_features {array float} -- list corresponding to features
        
        Returns:
            [float array] -- list of norm corresponding to features
        '''

        features_norm = []
        for i in range(len(abs_features)):
            features_norm.append(abs_features[i].norm(p=2))  # 平方和，开根号

        features_norm = torch.stack(features_norm)
        return features_norm

    # 每个对应位置元素，求差，取绝对值
    def get_abs_features(self, p1_track_frames_features, p2_track_frames_features, N):
        '''API to get absolute differences of features
        
        Arguments:
            p1_track_frames_features {features tensor} -- list of features for person1
            p2_track_frames_features {features tensor} -- list of features for person2
            N {int} -- length of this episode
        
        Returns:
            list -- list of absolute difference features 
        '''

        abs_features = []
        for i in range(N):
            abs_features.append((p1_track_frames_features[i] - p2_track_frames_features[i]).abs())

        abs_features = torch.stack(abs_features)
        return abs_features

    def get_weighted_history(self, abs_features_history, q_values_history):
        '''to get weighted history of features in this episode so far
        
        Arguments:
            features {tensor} -- features so far
            q_values_history {tensor} -- values so far in the episode
        
        Returns:
            tensor -- weighted history of features
        '''
        # print(q_values_history[:, int(Decision.UNSURE)].unsqueeze(1).shape)
        # print(q_values_history.exp().sum(dim=1, keepdim=True))
        weights = 1 - (q_values_history[:, int(Decision.UNSURE)].unsqueeze(1).exp() /
                       q_values_history.exp().sum(dim=1, keepdim=True))  # 计算所有采取过的动作sure的权重

        # print('weighted history', weights.shape, features.shape, q_values_history.shape)
        # print(weights)

        weighted_history = (abs_features_history * weights).sum(dim=0)
        # print('weighted history', weights.shape, features.shape, q_values_history.shape, weighted_history.shape)

        return weighted_history.squeeze()

    def step(self, new_action, q_values):
        '''API to take an action
        
        Arguments:
            action {int} -- action to be taken
            values {tensor} -- values given by Q-learning model
        
        Returns:
            next_state, reward, done, info -- similar to open AI gym
        '''

        # print(values.shape)
        assert ((new_action == int(Decision.SAME)) or
                (new_action == int(Decision.DIFFERENT)) or
                (new_action == int(Decision.UNSURE))), 'action is not valid'

        # append the values to history
        self.q_values_history.append(q_values)  # q_values表示3个动作对应的Q值
        next_state, reward, done, info = None, None, None, None

        if new_action == int(Decision.SAME):
            # if action is SAME and if the ID's are same, reward +1
            if self.pid1 == self.pid2:
                reward = +1
            else:
                reward = -1

            done = True
            next_state = None

        elif new_action == int(Decision.DIFFERENT):
            # if action is DIFFERENT and if the ID's are not same, reward +1
            if self.pid1 != self.pid2:
                reward = +1
            else:
                reward = -1

            done = True
            next_state = None

        else:
            # if action is UNSURE and if there features are depleted, reward -1
            if self.cur_step_index >= self.N:
                reward = -1
                done = True
                next_state = None
            else:

                # if there are more frames in the episode, determine the content for next_state
                next_state_feature = self.abs_features[self.cur_step_index]  # self.cur_step_index表示要匹配的图像对序号，初始序号为1
                next_state_history = self.get_weighted_history(self.abs_features[:self.cur_step_index],
                                                               torch.cat(self.q_values_history))
                next_state_norm = self.abs_features_norm[:self.cur_step_index + 1]  # 得到过去和下一组待匹配轨迹的特征差的二范数
                next_state = self.create_state(next_state_feature, next_state_history,
                                               next_state_norm)  # 简单拼接：当前，过去，未来，得到（4099）
                self.cur_step_index += 1

                done = False
                reward = self.reward_per_step  # 0.2

        return next_state, reward, done, info

    def create_state(self, feature, history, features_norm):
        # 用feature history features_norm的三个衍生数值，拼成shape为4099的张量，作为当前的观察，即当前状态
        current_observation = torch.cat([feature, history, features_norm.min().view(-1),
                                         features_norm.max().view(-1), features_norm.mean().view(-1)])
        return current_observation

    def reset(self):
        # give out the 0th frame features
        self.q_values_history = []
        self.cur_step_index = 1
        current_observation = self.create_state(self.abs_features[0], self.abs_features[0], self.abs_features_norm[0])
        # print(current_observation.shape)
        return current_observation


# experience replay buffer
class Memory:
    def __init__(self, size):
        self.size = size
        self.samples = deque()  # 双向队列

    def append(self, x):
        # reduce the size until it become self.size 
        if isinstance(x, collections.Iterable):
            # if it is array, the add it
            in_items_len = len(x)
            while (len(x) + in_items_len) >= self.size:
                x.popleft()

            self.samples += x  # 向双向队列中添加一项
        else:
            # if it is single element, append it
            while (len(x) + 1) >= self.size:
                x.popleft()

            self.samples.append(x)

    def sample(self, n):
        # sample random n samples
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


class Brain(nn.Module):
    """Core logic of DQN
    """

    def __init__(self, nStateDim, nActions):
        super().__init__()

        # an MLP for state-action value function
        self.state_action_value = nn.Sequential(
            nn.Linear(nStateDim, 128),  # nStateDim: 4099
            nn.ReLU(inplace=True),
            # nn.Linear(128, 128),
            # nn.ReLU(inplace=True),            
            nn.Linear(128, nActions)
        )

    def forward(self, x):
        # if np array, convert into torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().cuda()  # 创建张量

        # if state has no batch size, create a dummy batch with one sample
        if x.ndimension() == 1:
            x = x.unsqueeze(0)

        return self.state_action_value(x)  # 通过MLP


class Agent(nn.Module):
    def __init__(self, feature_extractor, args, nBufferSize=MEMORY_CAPACITY):  # MEMORY_CAPACITY：10000
        super().__init__()
        self.feature_extractor = feature_extractor

        # input is different features, history, 3 handcrafted features
        self.brain = Brain(feature_extractor.feat_dim * 2 + 3,
                           3)  # feat_dim=2048，feature_extractor.feat_dim * 2 + 3=4099
        if not args.use_cpu:
            self.brain = self.brain.cuda()

        self.memory = Memory(nBufferSize)
        self.reward_per_step = args.reward_per_step

        # optimizer setup
        self.steps = 0
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def act(self, current_state, epsilon):
        '''epsilon greedy action selection
        
        Arguments:
            s {np array} -- state selection
            epsilon {float} -- epsilon value
        
        Returns:
            action -- action index
        '''

        q_values = self.brain(current_state)  # 得到3个动作对应的数值
        if epsilon > np.random.uniform():  # episilon表示采取random action的概率
            action = np.random.choice(3)
        else:
            with torch.no_grad():
                action = torch.argmax(q_values, dim=1).item()

        return q_values, action  # 返回当前状态得到的Q值、根据Q值选取or随机选取的动作

    def get_epsilon(self):
        '''get epsilon value
        
        Returns:
            float -- epsilon value
        '''
        # MIN_EPSILON:0.01   MAX_EPSILON:1  LAMBDA=0.0001
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)
        # 1 - (epoch - 1) * 0.1
        # self.epsilon = max(0.05, self.epsilon)
        # MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)
        self.steps += 1
        return self.epsilon

    def play_one_episode(self, is_test=False):
        '''play one episode of cart pole
        
        Keyword Arguments:
            is_test {bool} -- is it a test episode? (default: {False})
        
        Returns:
            samples, total_reward, iters -- collected samples, rewards and total iterations
        '''

        current_state = self.env.reset()
        total_cur_episode_reward = 0
        match_times = 0
        samples = []
        done = False
        # print('start')

        while not done:
            # sample an action according to epsilon greedy strategy
            epsilon = 0
            if not is_test:  # 如果是在训练，则计算epsilon；如果在测试，则默认0
                epsilon = self.get_epsilon()
            else:
                epsilon = 0

            # print(current_state.shape)
            q_values, new_action = self.act(current_state, epsilon)
            # print(q_values.shape)
            next_state, reward, done, info = self.env.step(new_action, q_values)
            # print('next', action, reward, done)
            # input()
            match_times += 1
            total_cur_episode_reward += reward
            samples.append([current_state, new_action, reward, next_state])
            current_state = next_state

        return samples, total_cur_episode_reward, match_times, q_values

    def collect_data(self, dataloader, args):
        '''collect data and put it into replay memory
        
        Arguments:
            datacol_num_episodes {int} -- number of episodes to run and collect data
        
        Returns:
            avg_reward -- average reward in data collection
        '''

        total_datacol_rewards = 0

        self.feature_extractor.eval()  # 指明非训练模式

        if not args.use_cpu:
            self.feature_extractor = self.feature_extractor.cuda()

        n = 0
        pid1_collect = []
        pid2_collect = []
        for i, data in enumerate(tqdm(dataloader)):
            n += 1  # test: 迭代多少次
            p1_track, pid1, p2_track, pid2 = data
            pid1_collect.append(pid1)
            pid2_collect.append(pid2)
            if not args.use_cpu:
                p1_track = p1_track.cuda()
                p2_track = p2_track.cuda()

            with torch.no_grad():
                _, p1_track_frames_features = self.feature_extractor(p1_track)  # 每张图像特征：1,2,2048
                _, p2_track_frames_features = self.feature_extractor(p2_track)
            # p1_track_frames_features.squeeze(): 1,2,2048=>2,2048
            self.env = Environment({'features': p1_track_frames_features.squeeze(), 'id': pid1},
                                   {'features': p2_track_frames_features.squeeze(), 'id': pid2}, self.reward_per_step)

            # play each episode and put the samples into memory
            samples, total_cur_episode_reward, match_times, _ = self.play_one_episode()
            self.memory.append(samples)
            total_datacol_rewards += total_cur_episode_reward

        avg_reward = total_datacol_rewards / len(dataloader)
        return avg_reward

    def get_target_Q_values(self, rewards, next_states, target_model):
        '''get the target Q values for Q-learning
        
        Arguments:
            rewards {tensor} -- reward for each transition
            next_states {array of list} -- state description
            model {DQ network} -- target model
        
        Returns:
            Q target -- Q target values
        '''

        target_Q_values = torch.zeros(len(next_states), 1).to('cuda')  # （batch_size,1）

        for i, next_state in enumerate(
                next_states):  # 循环遍历 next_states，注意next_state为None表示为episode结束step标记，不为None，说明当前还未结束
            if next_state is None:  # 如果结束了，则直接将奖励作为target_Q_values
                target_Q_values[i, 0] = rewards[i]
            else:  # 如果是未结束的episode，则在target_model中得到对应的q_values，取最大q_value，计算rewards[i] + GAMMA * max_q_value得到target_Q_values
                state_torch = next_state.unsqueeze(0).cuda()
                with torch.no_grad():
                    q_values = target_model(state_torch)

                max_q_value = torch.max(q_values, dim=1)[0].item()
                target_Q_values[i, 0] = rewards[i] + GAMMA * max_q_value

        return target_Q_values

    def tensorize_samples(self, samples):
        '''collect the current states, rewards, actions, next states from the sampled data from 
        experience replay buffer
        
        Arguments:
            samples {array of list} -- data sampled from replay buffer
        
        Returns:
            current_states, actions, rewards, next_states -- separated data
        '''

        current_states, actions, rewards, next_states = [], [], [], []
        for _, data in enumerate(samples):
            current_states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])

        current_states = torch.stack(current_states)
        actions = torch.from_numpy(np.array(actions)).unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float)).float().unsqueeze(1)
        return current_states, actions, rewards, next_states

    def train(self, args, num_runs=100):
        '''train the DQN for num_runs iterations
        
        Keyword Arguments:
            num_runs {int} -- number of iterations to train (default: {100})
        
        Returns:
            avg_train_loss -- average train loss
        '''

        # backup the brain for fixed target
        target_model = copy.deepcopy(self.brain)
        total_loss = 0
        self.brain.train()  # MLP是待训练模型，特征提取模型是非训练模型
        self.feature_extractor.eval()
        if not args.use_cpu:
            self.brain = self.brain.cuda()

        for i in tqdm(range(num_runs)):  # 每个epoch中要迭代的次数
            # sample data from memory
            data = self.memory.sample(BATCH_SIZE)
            cur_states, actions, rewards, next_states = self.tensorize_samples(data)  # 将list转换为张量

            if not args.use_cpu:
                cur_states = cur_states.cuda().detach().clone()
                actions = actions.cuda().detach().clone()
                rewards = rewards.cuda().detach().clone()

            # get max Q values of next state to form fixed target
            target_Q_values = self.get_target_Q_values(rewards, next_states,
                                                       target_model)  # 得到batch_size个target_Q_value

            # update the model
            cur_Q_values = self.brain(cur_states)
            self.optimizer.zero_grad()
            # loss = torch.mean((target_Q - currentQ.gather(1, actions))**2)
            actions = actions.type(dtype=torch.LongTensor)
            actions = actions.cuda().detach().clone()
            loss = torch.mean((target_Q_values - cur_Q_values.gather(1, actions)) ** 2)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return total_loss / num_runs

    def select_random_features(self, source_features, num_features):
        # source features is of size #len x #feat_dim
        # we have to select random #num_features
        num_features = min(num_features, source_features.shape[0])
        indices = np.random.choice(range(num_features), size=num_features, replace=False)  # 不放回抽样
        return source_features[indices]

    def test(self, queryloader, galleryloader, args, ranks=[1, 5, 10, 20]):
        self.brain.eval()
        self.feature_extractor.eval()

        buffer_file = osp.join(args.buffer_file, args.dataset + '_test_features.pth')
        if not os.path.exists(buffer_file):
            # if the buffer files saved with features already existing, load buffer file
            # if not, extract the features from feature extractor
            query_track_imgs_feat, query_track_mean_feat, query_pids, query_cids = [], [], [], []
            args.logger('extracting query features')
            for batch_idx, (imgs, pids, cids) in enumerate(tqdm(queryloader)):  # 1980个轨迹片段
                if not args.use_cpu:
                    imgs = imgs.cuda()

                with torch.no_grad():
                    if imgs.ndimension() <= 5:
                        imgs = imgs.unsqueeze(0)

                    track_num, per_track_len, s, c, h, w = imgs.size()  # s=???
                    assert (track_num == 1)
                    imgs = imgs.view(track_num * per_track_len, s, c, h, w)
                    track_imgs_feature = get_features(self.feature_extractor, imgs,
                                                      args.test_num_tracks)  # 该步骤的目的就是给图像分批提取特征，减轻gpu压力
                    mean_features = torch.mean(track_imgs_feature, 0)  # track_imgs_feature(40,2048),

                    track_imgs_feature = track_imgs_feature.data.cpu()
                    track_mean_feature = mean_features.data.cpu()
                    torch.cuda.empty_cache()

                    query_track_imgs_feat.append(track_imgs_feature)
                    query_track_mean_feat.append(track_mean_feature)
                    query_pids.extend(pids)
                    query_cids.extend(cids)

            query_track_mean_feat = torch.stack(query_track_mean_feat)
            query_pids = np.asarray(query_pids)
            query_cids = np.asarray(query_cids)

            args.logger("Extracted features for query set, obtained {}-by-{} matrix".format(query_track_mean_feat.size(0),
                                                                                      query_track_mean_feat.size(1)))

            gallery_track_imgs_feat, gallery_track_mean_feat, gallery_pids, gallery_cids = [], [], [], []
            args.logger('extracting gallery features')
            for batch_idx, (imgs, pids, cids) in enumerate(tqdm(galleryloader)):
                if not args.use_cpu:
                    imgs = imgs.cuda()

                with torch.no_grad():
                    if imgs.ndimension() <= 5:
                        imgs = imgs.unsqueeze(0)

                    track_num, per_track_len, s, c, h, w = imgs.size()
                    assert (track_num == 1)
                    imgs = imgs.view(track_num * per_track_len, s, c, h, w)
                    track_imgs_feature = get_features(self.feature_extractor, imgs, args.test_num_tracks)
                    mean_features = torch.mean(track_imgs_feature, 0)

                    track_imgs_feature = track_imgs_feature.data.cpu()
                    mean_features = mean_features.data.cpu()
                    torch.cuda.empty_cache()

                    gallery_track_imgs_feat.append(track_imgs_feature)
                    gallery_track_mean_feat.append(mean_features)
                    gallery_pids.extend(pids)
                    gallery_cids.extend(cids)

            gallery_track_mean_feat = torch.stack(gallery_track_mean_feat)
            gallery_pids = np.asarray(gallery_pids)
            gallery_cids = np.asarray(gallery_cids)

            args.logger("Extracted features for gallery set, obtained {}-by-{} matrix".format(gallery_track_mean_feat.size(0),
                                                                                        gallery_track_mean_feat.size(
                                                                                            1)))

            torch.save({'query': {'track_mean_feat': query_track_mean_feat,
                                  'track_imgs_feat': query_track_imgs_feat,
                                  'pids': query_pids,
                                  'cids': query_cids},
                        'gallery': {'track_mean_feat': gallery_track_mean_feat,
                                    'track_imgs_feat': gallery_track_imgs_feat,
                                    'pids': gallery_pids,
                                    'cids': gallery_cids}
                        }, buffer_file)

        else:
            # load the buffer file
            args.logger('loading and extraction information/features from file: {}'.format(buffer_file))
            buffer = torch.load(buffer_file)
            query_track_mean_feat = buffer['query']['track_mean_feat']
            query_track_imgs_feat = buffer['query']['track_imgs_feat']
            query_cids = buffer['query']['cids']
            query_pids = buffer['query']['pids']
            gallery_track_mean_feat = buffer['gallery']['track_mean_feat']
            gallery_track_imgs_feat = buffer['gallery']['track_imgs_feat']
            gallery_cids = buffer['gallery']['cids']
            gallery_pids = buffer['gallery']['pids']

        args.logger("Computing distance matrix for all frames evaluation (baseline)")
        query_track_num, gallery_track_num = query_track_mean_feat.size(0), gallery_track_mean_feat.size(0)  # 1980, 9330
        distmat = torch.pow(query_track_mean_feat, 2).sum(dim=1, keepdim=True).expand(query_track_num, gallery_track_num) + \
                  torch.pow(gallery_track_mean_feat, 2).sum(dim=1, keepdim=True).expand(gallery_track_num, query_track_num).t()
        distmat.addmm_(query_track_mean_feat, gallery_track_mean_feat.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()

        args.logger("Computing CMC and mAP")
        cmc, mAP = evaluateMultiShot(distmat, query_pids, gallery_pids, query_cids, gallery_cids)
        args.logger("Results ----------")
        args.logger("mAP: {:.1%}".format(mAP))
        args.logger("CMC curve")
        for r in ranks:
            args.logger("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))  # '{:*<10}'.format('分割线') => '分割线*******'
        args.logger("------------------")

        args.logger('Computing distance matrix from DQN network')
        distmat = torch.zeros(query_track_num, gallery_track_num)
        instance_rewards = torch.zeros(query_track_num, gallery_track_num)
        comparisons = torch.zeros(query_track_num, gallery_track_num)
        args.logger('query_track_num: {}, gallery_track_num: {}'.format(query_track_num, gallery_track_num))
        for i, query_pid in tqdm(enumerate(query_pids)):
            q_features = self.select_random_features(query_track_imgs_feat[i], args.rl_k)
            for j, gallery_pid in enumerate(gallery_pids):
                # args.logger(query_track_imgs_feat[i].shape, gallery_track_imgs_feat[j].shape)
                g_features = self.select_random_features(gallery_track_imgs_feat[j], args.rl_k)

                if not args.use_cpu:
                    q_features = q_features.cuda()
                    g_features = g_features.cuda()

                # args.logger(q_features.shape, g_features.shape)
                self.env = Environment({'features': q_features,
                                        'id': query_pid},
                                       {'features': g_features,
                                        'id': gallery_pid}, args.reward_per_step)
                _, reward, match_times, q_vals = self.play_one_episode(is_test=True)
                instance_rewards[i, j] = reward
                comparisons[i, j] = match_times
                distmat[i, j] = (q_vals[:, 1] - q_vals[:, 0]).item()
                # break

        args.logger("Computing CMC and mAP (+ve distmat)")
        cmc, mAP = evaluateMultiShot(distmat, query_pids, gallery_pids, query_cids, gallery_cids)
        args.logger("Results ----------")
        args.logger("mAP: {:.1%}".format(mAP))
        args.logger("CMC curve")
        for r in ranks:
            args.logger("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        args.logger("------------------")

        args.logger("Computing CMC and mAP (-ve distmat)")
        cmc, mAP = evaluateMultiShot(distmat, query_pids, gallery_pids, query_cids, gallery_cids)
        args.logger("Results ----------")
        args.logger("mAP: {:.1%}".format(mAP))
        args.logger("CMC curve")
        for r in ranks:
            args.logger("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        args.logger("------------------")

        args.logger('average rewards: {}'.format(instance_rewards.mean().item()))
        args.logger('average comparison: {}'.format(comparisons.mean().item()))

        return cmc[0]
