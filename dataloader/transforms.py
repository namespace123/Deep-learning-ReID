# -------------------------------------------------------------------------------
# Description:  单独拎出一个文件来做数据预处理，做数据增广等
# Description:
# Reference:  http://man.hubwiz.com/manual/torchvision
# Author: Sophia
# Date:   2021/7/6
# -------------------------------------------------------------------------------
import math

from torchvision.transforms import *
from PIL import Image
import random


# 随机选取一块目标区域进行擦除，即在图像中随机选取的目标区域内填充预定义好的固定值
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            # 随机出目标区域大小、长宽比，随机求得目标区域长、宽值
            target_area = random.uniform(self.sl, self.sh) * area  # 目标区域
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)  # 长宽比

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            # 当长宽比过大or过小，目标区域过于狭长时，该条件可能不成立，所以用loop=100来确保有返回值
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


# 输入一张图片，随机采集一个区域
# 代码实现一个随机裁剪的transform类，torch中已有，此处重构只为演示
class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):  # interpolation: 插值
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
            Args:
                img (PIL Image): Image to be cropped.

            Returns:
                PIL Image: Cropped image.
        """
        if random.random() < self.p:  # 不做数据增广
            return img.resize((self.width, self.height), self.interpolation)  # PIL库是先写宽度，再写高度
        # 做数据增广
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resize_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resize_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


if __name__ == '__main__':
    img = Image.open('D://data/market1501/bounding_box_train/0002_c1s1_000451_03.jpg')
    # transform = Random2DTranslation(256, 128, 0.5)
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        Random2DTranslation(256, 128, 0.5),
        transforms.RandomHorizontalFlip(),  # 0.5的概率进行水平翻转
        # transforms.ToTensor()  # 如果转换为tensor，就不能plt展示了
    ])

    img_t = transform(img)
    # print(img_t.shape)  # 如果转换为tensor，就可以查看shape：torch.Size([3, 256, 128])
    import matplotlib.pyplot as plt

    plt.figure(12)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_t)
    plt.show()
