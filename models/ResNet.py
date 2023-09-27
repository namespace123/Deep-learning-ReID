from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from models.HorizontalMaxPool2d import HorizontalMaxPool2d
from torchvision import models

__all__ = ['ResNet50', 'ResNet101', 'ResNet50M']


# for multi-shot ReID via Sequential Decision Making
class ResNet50TP(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50TP, self).__init__()
        resnet50 = models.resnet.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])  # 倒数第2层是AvgPool层，跟图像大小有关；倒数第一层是FC层，跟类别个数有关；因此去掉这两层
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        track_num = x.size(0)
        per_track_len = x.size(1)
        # reshape: (imgs_num, 3, 256, 128)
        x = x.view(track_num * per_track_len, x.size(2), x.size(3),
                   x.size(4))  # track_num * per_track_len: total frames

        # collect individual frame features and global video features by avg pooling
        spatial_out = self.base(x)  # 2,2048,8,4
        avg_spatial_out = F.avg_pool2d(spatial_out, spatial_out.size()[2:])  # 2,2048,1,1
        individual_img_features = avg_spatial_out.view(track_num, per_track_len, -1)  # 1,2,2048
        individual_features_permuted = individual_img_features.permute(0, 2, 1)  # 1,2048,2
        video_features = F.avg_pool1d(individual_features_permuted, per_track_len)  # 1,2048,1
        video_features = video_features.view(track_num, self.feat_dim)  # 1,2048

        if not self.training:
            # (track_num, 2048) 每个轨迹特征, (track_num, per_track_len, 2048) 每张图像特征
            return video_features, individual_img_features

        y = self.classifier(video_features)
        return y, video_features, individual_img_features


class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax, metric'}, aligned=False, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)  # 加载预训练模型
        # print(resnet50)  # 倒数第2层是AvgPool层，跟图像大小有关；倒数第一层是FC层，跟类别个数有关；因此需要先去掉这两层
        self.base = nn.Sequential(*list(resnet50.children())[:-2])  # 去掉最后的两层
        if self.loss != {'metrics'}:
            self.classifier = nn.Linear(2048, num_classes)
        # self.feat_dim = 2048  # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        if self.aligned:
            if self.training:
                x = self.base(x)
                local_feats = self.conv1(self.horizon_pool(self.relu(self.bn(x))))  # 32,128,1,1
                local_feats = local_feats.view(local_feats.size()[0:3])  # 32,128,1
                local_feats = local_feats / torch.pow(local_feats, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()

                x = F.avg_pool2d(x, x.size()[2:])
                f = x.view(x.size(0), -1)
                y = self.classifier(f)
                return y, f, local_feats
            else:
                x = self.base(x)
                local_feats = self.horizon_pool(x)
                local_feats = local_feats.view(local_feats.size()[0:3])
                local_feats = local_feats / torch.pow(local_feats, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()

                x = F.avg_pool2d(x, x.size()[2:])
                f = x.view(x.size(0), -1)
                return f, local_feats
        else:
            x = self.base(x)  # 32, 3, 256, 128 -> 32, 2048, 8, 4
            x = F.avg_pool2d(x, x.size()[2:])  # 32,2048,1,1
            feats = x.view(x.size(0), -1)  # 32,2048   将x展平成二维，但保证第一维不变，第二维自动生成
            # 对每一行数据进行归一化，形状不变
            # feats = 1.*feats / (torch.norm(feats, 2, dim=-1, keepdim=True).expand_as(feats) + 1e-12)

            if not self.training:
                return feats
            if self.loss == {'softmax'}:
                y = self.classifier(feats)
                return y
            elif self.loss == {'metrics'}:
                return feats
            elif self.loss == {'softmax', 'metrics'}:
                y = self.classifier(feats)
                return y, feats
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))



class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048  # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """

    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072  # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)

        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


if __name__ == '__main__':
    model = ResNet50(num_classes=751, loss={'softmax, metric'}, aligned=True)
    imgs = torch.Tensor(32, 3, 256, 128)  # batchsize,channel,height,width
    y, f, local_feature = model(imgs)  # local_feature.shape: torch.Size([32, 128, 8, 1])
    print(y.shape)
    print(f.shape)
    print(local_feature.shape)
