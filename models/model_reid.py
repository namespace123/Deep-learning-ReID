# -------------------------------------------------------------------------------
# Description:  HOReid 用到的一些 layer
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/9/2
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision

# 根据不同的网络层，定义不同的初始化方式
def weights_init_kaiming(m):
    classname = m.__class__.__name__  # get the param type
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)  # 常数填充张量
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:  # affine为True，表示在BatchNorm中weight&bias将被使用
            nn.init.constant_(m.weight, 1.0)  # 缩放参数
            nn.init.constant_(m.bias, 0.0)  # 平移参数

def weights_init_classifier(m):
    classname = m.__class__.__name__  # get the param type
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



# CNN backbone: ResNet50
# removing its global average pooling (GAP) layer and fully connected layer
class Encoder(nn.Module):

    def __init__(self, class_num):
        super(Encoder, self).__init__()
        self.class_num = class_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        # cnn backbone
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no rule
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

    def forward(self, x):
        feature_map = self.resnet_conv(x)
        return feature_map

class BNClassifier(nn.Module):

    def __init__(self, input_dim, class_num):
        super(BNClassifier, self).__init__()

        self.input_dim = input_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.input_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.input_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)  # 将该权重初始化方式应用到所有子模块
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # a batch nor malization layer and a fully connect layer followed by a softmax function.
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score  # (8,1024),(8,702)

class BNClassifiers(nn.Module):

    def __init__(self, input_dim, class_num, branch_num):  #  branch_num=14
        super(BNClassifiers, self).__init__()

        self.input_dim = input_dim
        self.class_num = class_num
        self.branch_num = branch_num

        for i in range(self.branch_num):
            #  setattr(object, name, values) 给对象的属性赋值，若属性不存在，先创建再赋值。
            setattr(self, 'classifier_{}'.format(i), BNClassifier(self.input_dim, self.class_num))

    def __call__(self, feature_vector_list):
        assert len(feature_vector_list) == self.branch_num

        # bnneck for each sub_branch_feature
        bn_feature_vector_list, cls_score_list = [], []
        for i in range(self.branch_num):  # 14个关键点特征，逐个处理
            feature_vector_i = feature_vector_list[i]
            classifier_i = getattr(self, 'classifier_{}'.format(i))
            bn_feature_vector_i, cls_score_i = classifier_i(feature_vector_i)  # (8,1024),(8,702)
            bn_feature_vector_list.append(bn_feature_vector_i)
            cls_score_list.append(cls_score_i)

        return bn_feature_vector_list, cls_score_list  # (14,8,1024),(14,8,702) 每个关键点下的批次特征&one-hot预测值

if __name__ == '__main__':
    weights_init_kaiming(11)




