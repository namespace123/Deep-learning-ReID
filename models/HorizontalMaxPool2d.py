# -------------------------------------------------------------------------------
# Description:  水平pooling
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/13
# -------------------------------------------------------------------------------
import torch.nn as nn

class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()

    def forward(self, x):
        input_size = x.size()  # N*C*H*W
        return nn.functional.max_pool2d(input=x, kernel_size=(1, input_size[3]))

#

# if __name__ == '__main__':
#     import torch
#     x = torch.Tensor(32, 2048, 8, 4)  # 256*128的图片，batch_size=32,经过resnet50输出的feature map
#     hp = HorizontalMaxPool2d()
#     y = hp(x)
#     print(y.shape)  # torch.Size([32, 2048, 8, 1])























































