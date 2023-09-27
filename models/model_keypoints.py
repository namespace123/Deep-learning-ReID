# -------------------------------------------------------------------------------
# Description:  
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/9/4
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn


from models.model_keypoints.pose_hrnet import get_pose_net
from models.model_keypoints.config import cfg as pose_config
from models.model_keypoints.pose_processor import HeatmapProcessor, HeatmapProcessor2

class ScoremapComputer(nn.Module):

    def __init__(self, norm_scale):
        super(ScoremapComputer, self).__init__()

        # init skeleton model
        self.keypoints_predictor = get_pose_net(pose_config, False)
        self.keypoints_predictor.load_state_dict(torch.load(pose_config.TEST.MODEL_FILE))
        # self.heatmap_processor = HeatmapProcessor(normalize_heatmap=True, group_mode='sum', gaussian_smooth=None)
        self.heatmap_processor = HeatmapProcessor2(normalize_heatmap=True, group_mode='sum', norm_scale=norm_scale)

    def _slow_forward(self, x):
        heatmap = self.keypoints_predictor(x)  # before normalization
        scoremap, keypoints_confidence, keypoints_location = self.heatmap_processor(heatmap)  # after normalization
        return scoremap.detach(), keypoints_confidence.detach(),keypoints_location.detach(),


