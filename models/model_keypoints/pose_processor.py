import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .config import cfg as pose_config
from .gaussian_blur import GaussianBlur


class HeatmapProcessor(nn.Module):
    """post process of the heatmap, group and normalize"""

    def __init__(self, normalize_heatmap=False, group_mode="sum", gaussian_smooth=None):
        super(HeatmapProcessor, self).__init__()
        self.num_joints = pose_config.MODEL.NUM_JOINTS
        self.groups = pose_config.MODEL.JOINTS_GROUPS
        self.gaussian_smooth = gaussian_smooth
        assert group_mode in ['sum', 'max'], "only support sum or max"
        self.group_mode = group_mode
        print("groupmod", self.group_mode)
        self.normalize_heatmap = normalize_heatmap
        if self.normalize_heatmap:
            print("normalize scoremap")
        else:
            print("no normalize scoremap")
        if self.gaussian_smooth:
            kernel, sigma = self.gaussian_smooth
            self.gaussian_blur = GaussianBlur(kernel, sigma)
            print("gaussian blur:", kernel, sigma)
        else:
            self.gaussian_blur = None
            print("no gaussian blur")

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.interpolate(x, [16, 8], mode='bilinear', align_corners=False)
        n, c, h, w = x.shape

        if not self.training:
            # if in eval phase, we calculate the max value and its position of each channel of heatmap
            n, c, h, w = x.shape

            x_reshaped = x.reshape((n, c, -1))
            idx = torch.argmax(x_reshaped, 2)
            max_response, _ = torch.max(x_reshaped, 2)

            idx = idx.reshape((n, c, 1))
            max_response = max_response.reshape((n, c))
            max_index = torch.empty((n, c, 2))
            max_index[:, :, 0] = idx[:, :, 0] % w  # column
            max_index[:, :, 1] = idx[:, :, 0] // w  # row

        if self.gaussian_blur:
            x = self.gaussian_blur(x)

        if self.group_mode == 'sum':
            heatmap = torch.sum(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2 = torch.mean(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi = torch.sum(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i = torch.mean(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)


        elif self.group_mode == 'max':
            heatmap, _ = torch.max(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2, _ = torch.max(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi, _ = torch.max(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i, _ = torch.max(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)

        if self.normalize_heatmap:
            heatmap = self.normalize(heatmap)

        if self.training:
            return heatmap
        else:
            return heatmap, max_response_2, max_index

    def normalize(self, in_tensor):
        n, c, h, w = in_tensor.shape
        in_tensor_reshape = in_tensor.reshape((n, c, -1))

        normalized_tensor = F.softmax(in_tensor_reshape, dim=2)
        normalized_tensor = normalized_tensor.reshape((n, c, h, w))

        return normalized_tensor


class HeatmapProcessor2:

    def __init__(self, normalize_heatmap=True, group_mode="sum", norm_scale=1.0):

        self.num_joints = pose_config.MODEL.NUM_JOINTS
        self.groups = pose_config.MODEL.JOINTS_GROUPS

        self.group_mode = group_mode
        self.normalize_heatmap = normalize_heatmap
        self.norm_scale = norm_scale
        assert group_mode in ['sum', 'max'], "only support sum or max"

    def __call__(self, x):
        x = F.interpolate(x, [16, 8], mode='bilinear', align_corners=False)
        n, c, h, w = x.shape

        x_reshaped = x.reshape((n, c, -1))
        idx = torch.argmax(x_reshaped, 2)
        max_response, _ = torch.max(x_reshaped, 2)

        idx = idx.reshape((n, c, 1))
        max_response = max_response.reshape((n, c))
        max_index = torch.empty((n, c, 2))
        max_index[:, :, 0] = idx[:, :, 0] % w  # column
        max_index[:, :, 1] = idx[:, :, 0] // w  # row

        if self.group_mode == 'sum':
            heatmap = torch.sum(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2 = torch.mean(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi = torch.sum(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i = torch.mean(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)

        elif self.group_mode == 'max':
            heatmap, _ = torch.max(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2, _ = torch.max(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi, _ = torch.max(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i, _ = torch.max(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)

        if self.normalize_heatmap:
            heatmap = self.normalize(heatmap, self.norm_scale)

        return heatmap, max_response_2, max_index

    def normalize(self, in_tensor, norm_scale):
        n, c, h, w = in_tensor.shape
        in_tensor_reshape = in_tensor.reshape((n, c, -1))

        normalized_tensor = F.softmax(norm_scale * in_tensor_reshape, dim=2)
        normalized_tensor = normalized_tensor.reshape((n, c, h, w))

        return normalized_tensor
