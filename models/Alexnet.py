from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
import torchvision

# for multi-shot ReID via Sequential Decision Making
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=True)
        
        # feature extractor
        self.base = alexnet.features
        self.feat_dim = 256

        # classifier
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))

        # collect individual frame features and global video features by avg pooling
        spatial_out = self.base(x)
        individual_img_features = F.avg_pool2d(spatial_out, spatial_out.shape[2:]).view(b*t, -1)

        # format into video, sequence way
        individual_img_features = individual_img_features.view(b, t, -1)

        # prepare for video level features
        individual_features_permuted = individual_img_features.permute(0,2,1)
        video_features = F.avg_pool1d(individual_features_permuted, t)
        video_features = video_features.view(b, self.feat_dim)

        if not self.training:
            return video_features, individual_img_features
        
        y = self.classifier(video_features)
        return y, video_features, individual_img_features
