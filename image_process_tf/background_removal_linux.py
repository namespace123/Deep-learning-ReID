# -------------------------------------------------------------------------------
# Description:  提取图像前景
# Description:  redhat
# Description:  
# Reference:
# Author: Sophia
# Date:   2021/8/25
# -------------------------------------------------------------------------------
import os
import os.path as osp
import argparse
from io import BytesIO

import numpy as np
from PIL import Image

import tensorflow as tf
import datetime
'''
对指定目录下的所有图片进行背景去除
'''
root = '/seu_share/home/liwei01/220205671/'
# params
parser = argparse.ArgumentParser(description='image background removal')
parser.add_argument('-i', '--input-dir', type=str, default=root + 'dataset/MARS/bbox_test', help="input_dir")
parser.add_argument('-o', '--output-dir', type=str, default=root + 'dataset/MARS/bbox_test_ex', help="output_dir")
# parser.add_argument('-i', '--input-dir', type=str, default='$HOME/dataset/Occluded_Duke/bounding_box_test', help="input_dir")
# parser.add_argument('-o', '--output-dir', type=str, default='$HOME/dataset/Occluded_Duke/bounding_box_test_ex2', help="output_dir")
parser.add_argument('-m', '--model', type=str, default='xception_model', help="xception_model or mobile_net_model")
parser.add_argument('-d', '--dataset', type=str, default='MARS', help="Occluded_Duke or MARS")

args = parser.parse_args()


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.GraphDef.FromString(open(osp.join(tarball_path, "frozen_inference_graph.pb"), "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        start = datetime.datetime.now()

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))

        return resized_image, seg_map


def drawSegment(baseImg, matImg, outputFilePath):
    width, height = baseImg.size
    dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = matImg[y, x]
            (r, g, b) = baseImg.getpixel((x, y))
            if color == 0:
                dummyImg[y, x, 3] = 0
            else:
                dummyImg[y, x] = [r, g, b, 255]
    img = Image.fromarray(dummyImg)
    img = img.convert('RGB')
    img.save(outputFilePath)


def run_visualization(filepath, outputFilePath, MODEL):
    """Inferences DeepLab model and visualizes result."""
    try:
        print("Trying to open : " + filepath)
        # f = open(sys.argv[1])
        jpeg_str = open(filepath, "rb").read()
        orignal_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check file: ' + filepath)
        return

    print('running deeplab on image %s...' % filepath)
    resized_im, seg_map = MODEL.run(orignal_im)

    # vis_segmentation(resized_im, seg_map)
    drawSegment(resized_im, seg_map, outputFilePath)


def check_image_path(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print("Bad parameters. Please specify input file path and output file path")
        exit()

    if os.path.exists(output_dir):
        print("output file path: ", output_dir, " exists")
    else:
        print('creating new fold: ', output_dir)
        os.makedirs(output_dir)

def listdir(path):  #传入存储的list
    list_name_l = []
    list_name_s = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_name_l.append(file_path)
            list_name_s.append(file)
    return list_name_l, list_name_s

def listfile(path):  #传入存储的list
    list_name_l = []
    list_name_s = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path):
            list_name_l.append(file_path)
            list_name_s.append(file)
    return list_name_l, list_name_s

if __name__ == '__main__':
    # check dirs
    input_dir = args.input_dir
    output_dir = args.output_dir
    check_image_path(input_dir, output_dir)
    # load model
    MODEL = DeepLabModel(osp.join('..', 'models', args.model))
    print('model loaded successfully : ' + args.model)

    if args.dataset == "Occluded_Duke":
        list_name_l2, list_name_s2 = listfile(input_dir)
        for j in range(len(list_name_l2)):
            img_file = list_name_l2[j]
            img_file_new = os.path.join(output_dir, list_name_s2[j])
            # image transform
            run_visualization(img_file, img_file_new, MODEL)

    elif args.dataset == "MARS":
        # get all sub-folds
        list_name_l, list_name_s = listdir(input_dir)
        for i in range(len(list_name_l)):

            # create new id_dir
            pid_dir = list_name_l[i]
            pid_dir_new = os.path.join(output_dir, list_name_s[i])
            if os.path.exists(pid_dir_new):
                print("id file path: ", pid_dir_new, " exists")
            else:
                os.makedirs(pid_dir_new)
                print("id file path: ", pid_dir_new, " is not existing, created it.")
            list_name_l2, list_name_s2 = listfile(pid_dir)
            for j in range(len(list_name_l2)):
                img_file = list_name_l2[j]
                img_file_new = os.path.join(pid_dir_new, list_name_s2[j])
                if os.path.exists(img_file_new):
                    print("new image file: ", img_file_new, "  exists, pass it.")
                else:
                    # image transform
                    run_visualization(img_file, img_file_new, MODEL)
