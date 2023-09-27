# -------------------------------------------------------------------------------
# Description:  主文件，包含训练、测试等代码，参数的设置
# Reference:
# Author: Sophia
# Date:   2021/7/6
# -------------------------------------------------------------------------------
import sys
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from util.utils import time_now
import numpy as np
from dataloader.data_loader_instance import LoaderMultiShotStage2
from MultiShotReID import RLMultiShotReid
from ops_argparse import stage2_ops
import warnings

warnings.filterwarnings('ignore')


def main():
    # init ops
    config = stage2_ops()

    # init dataloader
    dataloader = LoaderMultiShotStage2(config)

    base = RLMultiShotReid(config)

    if config.mode == 'train':
        config.logger("==> Train")

        if config.start_train_epoch != 0:  # resume model from the resume_train_epoch
            config.logger(
                'Time: {}, resume training from the epoch (model {})'.format(time_now(), config.start_train_epoch))

        best_cmc = -np.inf
        for epoch in range(config.start_train_epoch, config.total_train_epochs):

            cur_epoch = epoch + 1
            # in each epoch, collect data
            average_data_collection_reward = base.agent_model.collect_data(dataloader.trainloader, config)
            # train the network
            average_train_loss = base.agent_model.train(args=config, num_runs=config.num_train_iterations)

            config.logger(
                'Time: {}; Epoch: {}/{}; average_data_collection_reward: {}, average_train_loss: {}'.format( \
                    time_now(), cur_epoch, config.total_train_epochs, average_data_collection_reward, average_train_loss))

            # test
            if (cur_epoch) >= config.start_eval and config.eval_step > 0 and \
                    (cur_epoch) % config.eval_step == 0 or \
                    (cur_epoch) == config.total_train_epochs:
                config.logger("==> Test")
                cmc = base.agent_model.test(dataloader.queryloader, dataloader.galleryloader, config)

                print(average_data_collection_reward, average_train_loss, cmc)

                is_best = True if cmc >= best_cmc else False
                if is_best: best_cmc = cmc

                # save model  sophia
                if cur_epoch >= config.start_save_model_epoch and (
                        cur_epoch) % config.save_model_steps == 0 or cur_epoch == config.total_train_epochs - 1:
                    base.save_model(cur_epoch, cmc, is_best)

    elif config.mode == 'test':

        # test
        config.logger("Evaluate only")
        cmc = base.agent_model.test(dataloader.queryloader, dataloader.galleryloader, config)

        # save model  sophia
        base.save_model(-1, cmc, False)  # 保存测试结果


if __name__ == '__main__':
    main()  # main 函数的作用是，循环epoch，每次epoch调用一次train函数
