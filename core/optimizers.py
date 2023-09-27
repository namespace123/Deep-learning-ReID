# -------------------------------------------------------------------------------
# Description:  把一些常用的optimizer函数放在同一个文件夹下，非必要
# Description:  
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/7
# -------------------------------------------------------------------------------
import torch

'''
python模块中的__all__，用于模块导入时限制，如：from module import *
此时被导入模块若定义了__all__属性，则只有__all__内指定的属性、方法、类可被导入；若没定义，则导入模块内的所有公有属性，方法和类。
'''
__all__ = ['init_optim']

def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))