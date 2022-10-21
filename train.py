import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

from utils.dataloader import VOCSegDataset
from nets.fcn8 import FCN8

# This is for the progress bar.
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

# 对数据集进行可视化
# for i, (img, label) in enumerate(voc_train):
#     plt.subplot(121)
#     img_arr = img.numpy() * 255  # use np.numpy(): convert Tensor to numpy
#     img_arr = img_arr.astype('uint8')  # convert Float to Int
#     print(img_arr.shape)  # [C,H,W]
#     img_new = np.transpose(img_arr, (1, 2, 0))  # use np.transpose() convert [C,H,W] to [H,W,C]
#     plt.imshow(img_new)
#     plt.subplot(122)
#     plt.imshow(label)
#     plt.show()
#     plt.close()
#     if i == 1:
#         break

# 在训练网络前定义函数用于计算Acc 和 mIou
# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

 # 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    return acc, acc_cls, mean_iu

class Accmulator:
    def __init__(self, n):
        # Definde in: numref: sec_softmax_scratch
        self.n = n
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, key):
        return self.data[key]

def accuracy(y_hat, y):
    # 计算预测正确的数量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 通过获取最大值索引，为每个像素赋予类别
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
    # return F.cross_entropy(inputs, targets)

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)

# ------------------------------------------------------------------------

VOCdevkit_path = '/content/FCN/VOCdevkit/VOC2012/'
crop_size = (224, 224)
voc_train = VOCSegDataset(True, crop_size, VOCdevkit_path)
voc_val = VOCSegDataset(False, crop_size, VOCdevkit_path)

batch_size = 2

train_loader = DataLoader(voc_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(voc_val, batch_size=batch_size)

# parameters
num_classes = 21
n_epochs = 1000
learning_rate = 0.0001
weight_dacay = 5e-4
patience = 300
_exp_name = "sample"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

model = FCN8(num_classes, True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dacay)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_dacay)

stale = 0
best_acc = 0

for epoch in range(n_epochs):
    # 定义四个维度存储训练损失和、训练正确像素数、训练的图像数、总像素数
    metric = Accmulator(4)

    for batch in tqdm(train_loader):
        images, labels = batch

        model.train()
        optimizer.zero_grad()
        pred = model(images.to(device))
        # pred = torch.softmax(pred, dim=1)     # 添加在网络主体部分
        # print(images[0].shape)
        # print(pred[0].shape)
        # print(labels[0].shape)
        '''
            torch.Size([3, 224, 224])
            torch.Size([21, 224, 224])
            torch.Size([224, 224])
        '''
        l = loss(pred, labels.to(device))
        l.sum().backward()
        optimizer.step()

        train_loss_sum = l.sum()
        train_acc_sum = accuracy(pred, labels.to(device))
        # 此处add是累加，而不是添加内容
        metric.add(train_loss_sum, train_acc_sum, labels.shape[0], labels.numel())
        # print()
        # for i in range(4):
        #     print(metric[i])

    train_loss = metric[0] / metric[2]
    train_acc = metric[1] / metric[3]
    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")


    # ---------- Validation ----------
    model.eval()

    # 定义四个维度存储val损失和、val正确像素数、val的图像数、val总像素数
    metric_val = Accmulator(4)

    for batch in tqdm(val_loader):
        images, labels = batch

        with torch.no_grad():
            pred = model(images.to(device))

        l = loss(pred, labels.to(device))

        val_loss_sum = l.sum()
        val_acc_sum = accuracy(pred, labels.to(device))

        # 此处add是累加，而不是添加内容
        metric_val.add(train_loss_sum, train_acc_sum, labels.shape[0], labels.numel())

    val_loss = metric_val[0] / metric_val[2]
    val_acc = metric_val[1] / metric_val[3]
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

    # update logs
    filename = '/content/FCN/logs/' + f'./{_exp_name}_log.txt'
    if val_acc > best_acc:
        with open(filename, "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f} -> best")
    else:
        with open(filename, "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

    # save models
    if val_acc > best_acc:
        filename_ckpt = '/content/FCN/logs/' + f'{_exp_name}_best.ckpt'
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), filename_ckpt)  # only save best to prevent output memory exceed error
        best_acc = val_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break


