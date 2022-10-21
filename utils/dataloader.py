import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# .png是8位深度的图像，其与RBG的256^3种可能值存在一一对应的关系，在此进行转换
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    # 将.png图片转化为Tensor类型（torch.zeros初始化），根据上面的colormap与class对应的关系，得到包含类别的单通道Tensor
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def read_voc_images(VOCdevkit_path, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(VOCdevkit_path, 'ImageSets/', 'Segmentation/',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        # torchvision.io.read_image当前版本无法使用，因此使用numpy进行读取，并转换为tensor，使用permute进行维度变换
        # permute参数为，以（2，0,1）为例，新tensor第一维为原tensor第三维
        features.append(Image.open(os.path.join(VOCdevkit_path, 'JPEGImages/', f'{fname}.jpg')).convert("RGB"))
        labels.append(Image.open(os.path.join(VOCdevkit_path, 'SegmentationClass/', f'{fname}.png')).convert("RGB"))
    return features, labels

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    i, j, h, w = transforms.RandomCrop.get_params(feature, output_size=(height, width))
    feature = transforms.functional.crop(feature, i, j, h, w)
    label = transforms.functional.crop(label, i, j, h, w)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""
    def __init__(self, is_train, crop_size, VOCdevkit_path):
        """
                crop_size: (h, w)
                """
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        ])
        self.crop_size = crop_size  # (h, w)
        self.colormap2label = voc_colormap2label()
        images, labels = read_voc_images(VOCdevkit_path, is_train)
        self.images = self.filter(images)  # images list
        self.labels = self.filter(labels)  # labels list
        print('read ' + str(len(self.images)) + ' examples')

    def filter(self, imgs):  # 过滤掉尺寸小于crop_size的图片
        return [img for img in imgs if (
                img.size[1] >= self.crop_size[0] and
                img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.images[idx], self.labels[idx], self.crop_size[0], self.crop_size[1])
        feature = torch.from_numpy(np.array(feature)).permute(2, 0, 1)
        label = torch.from_numpy(np.array(label)).permute(2, 0, 1)
        return feature, voc_label_indices(label, self.colormap2label)

    def __len__(self):
        return len(self.images)


# VOCdevkit_path = '../datasets/VOCdevkit/VOC2012/'
# train_features, train_labels = read_voc_images(VOCdevkit_path, True)
# # size = (224, 224)
# # print(size.type)
# feature, label = voc_rand_crop(train_features[0], train_labels[0], 224, 224)
# test = np.array(feature)
# feature = torch.from_numpy(np.array(feature)).permute(2, 0, 1)
# label = torch.from_numpy(np.array(label)).permute(2, 0, 1)
# label = voc_label_indices(label, voc_colormap2label())
#
# test2 = Image.open(os.path.join(VOCdevkit_path, 'JPEGImages/', '2007_000032.jpg')).convert("RGB")
# test3 = np.array(test2)
# print(feature.shape)
# print(label.shape)
# plt.imshow(label.numpy())
# plt.show()

# VOC_path = '../datasets/VOC2012/'
# plt.imshow(train_labels[0])
# plt.show()
# jpg_fname = VOCdevkit_path + 'JPEGImages/' + '2007_000032' +'.jpg'
# png_fname = VOCdevkit_path + 'SegmentationClass/' + '2007_000032' +'.png'
# image = Image.open(jpg_fname).convert('RGB')
# label = Image.open(png_fname).convert('RGB')
# image, label = voc_rand_crop(image, label, 224, 224)


