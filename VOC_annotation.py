import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

# 定义训练集与验证集的占比，如果需要测试集则秀爱train_val_percent
train_val_percent = 0.9
# 定义训练集占比，此处划分训练集与验证集为7:3
train_percent = 0.7

# 修改路径指向VOC数据集
VOCdevkit_path = 'datasets/VOCdevkit/'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    # 分别设置ground truth图与存储的txt文件的路径
    segfile_path = os.path.join(VOCdevkit_path, 'VOC2012/SegmentationClass/')
    # print(segfile_path)
    saveBase_path = os.path.join(VOCdevkit_path, 'VOC2012/ImageSets/Segmentation/')

    # 获取文件夹下所有文件，临时存储在temp_seg中，读取文件夹下的.jpg文件
    temp_seg = os.listdir(segfile_path)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    train_val_num = int(num * train_val_percent)
    train_num = int(train_val_num * train_percent)
    # >> > random.sample(range(100), 10)  # sampling without replacement
    # [30, 83, 16, 4, 8, 81, 41, 50, 18, 33]
    train_val_list = random.sample(list, train_val_num)
    train_list = random.sample(train_val_list, train_num)

    # print("Train and val size: ", train_val_num)
    print("Train size: ", train_num)
    print("Val size: ", train_val_num - train_num)
    print("Test size: ", num - train_val_num)

    # 'w'：open for writing, truncating the file first
    train_val_file = open(os.path.join(saveBase_path, 'trainval.txt'), 'w')
    test_file = open(os.path.join(saveBase_path, 'test.txt'), 'w')
    train_file = open(os.path.join(saveBase_path, 'train.txt'), 'w')
    val_file = open(os.path.join(saveBase_path, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'
        if i in train_val_list:
            train_val_file.write(name)
            if i in train_list:
                train_file.write(name)
            else:
                val_file.write(name)
        else:
            test_file.write(name)

    train_val_file.close()
    train_file.close()
    val_file.close()
    test_file.close()

    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    classes_num = np.zeros([256], np.int)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfile_path, name)
        # 检查文件是否存在
        if not os.path.exists(png_file_name):
            raise ValueError("File not found: ", png_file_name)
        png = np.array(Image.open(png_file_name), np.uint8)
        # 检查图片格式，标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类
        if len(np.shape(png)) > 2:
            print("Label image format is incorrect, which shape is: ", str(np.shape(png)), ", please check: ", name)
        # np.bincount()从0到array中的最大值，每个数出现的次数，minlength限制最小输入，不够的补0
        classes_num += np.bincount(np.reshape(png, [-1]), minlength=256)
    # print(classes_num.shape)    # classes_num.shape (256, )

    if classes_num[255] > 0 and classes_num[0] > 0 and np.sum(classes_num[1:255]) == 0:
        print("It's found that the data in the label only contains 0 and 255.")
    elif classes_num[0] > 0 and np.sum(classes_num[1:]) == 0:
        print("It's found that the data in the label only contains backgroung information.")
