# -------------------------------------------------------------------------------
# Description:  对于常用的模型，可以通过pytorch库非常方便地实现
# Description:
# Description:
# Reference:
# Author: Sophia
# Date:   2021/7/7
# -------------------------------------------------------------------------------
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax, metric'}, **kwargs):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)  # 加载预训练模型
        # print(resnet50)  # 倒数第2层是AvgPool层，跟图像大小有关；倒数第一层是FC层，跟类别个数有关；因此需要先去掉这两层
        self.base = nn.Sequential(*list(resnet50.children())[:-2])  # 去掉最后的两层
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)  # 将x展平成二维，但保证第一维不变，第二维自动生成

        # 对每一行数据进行归一化，形状不变
        # f = 1.*f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)

        if not self.training:  # 如果非训练，则返回特征f；若是训练，则返回类别y
            return f
        y = self.classifier(f)
        return y



if __name__ == '__main__':
    model = ResNet50(num_classes=751)
    imgs = torch.Tensor(32, 3, 256, 128)  # batchsize,channel,height,width
    f = model(imgs)

























