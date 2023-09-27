# -------------------------------------------------------------------------------
# Description:  快速画出loss曲线
# Description:  生成loss图片，保存到log日志的同一目录下，取名与log文件一致；内容涉及正则化&plot画图技巧
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/7/12
# -------------------------------------------------------------------------------
import re
import matplotlib.pyplot as plt
import os.path as osp

# fullpath = osp.abspath('../../log/log_train_train_img_model_xent_htri20210712-191400.txt')
fullpath = osp.abspath('../log/log_train20210713-144816.txt')
# mode = {'Loss'}
mode = {'Loss', 'CLoss', 'TLoss'}

filedir, filename = osp.split(fullpath)
count = 0
Loss, CLoss, TLoss, x = [], [], [], []
with open(fullpath, 'r') as f:
    while True:
        line = f.readline()
        if line == '':
            break
        if not line.startswith('Epoch: ['):
            continue
        count += 1
        line = line.replace(' ', '').replace('\t', '')
        pattern = re.compile(r'[Loss]\w*.\w+[(](\w*.\w+)[)]')
        find_list = pattern.findall(line)
        if mode == {'Loss'}:
            Loss.append(float(find_list[0]))
        elif mode == {'Loss', 'CLoss', 'TLoss'}:
            Loss.append(float(find_list[0]))
            CLoss.append(float(find_list[1]))
            TLoss.append(float(find_list[2]))
        x.append(count)

pngName = filename.split('.')[0]

if mode == {'Loss'}:
    plt.plot(x, Loss)
elif mode == {'Loss', 'CLoss', 'TLoss'}:
    plt.plot(x, Loss, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    plt.plot(x, CLoss, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    plt.plot(x, TLoss, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    plt.legend(labels=('Loss', 'CLoss', 'TLoss'))

plt.savefig(osp.join(filedir, pngName))
plt.show()


