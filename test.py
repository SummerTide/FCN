import torch
from torch.nn import functional as F
import torch.nn as nn

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 通过获取最大值索引，为每个像素赋予类别
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y # cmp: tensor([[False, True, True]])
    return float(reduce_sum(astype(cmp, y.dtype)))

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)

# x的每行表示五种类别分别的概率
x = torch.tensor([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 1], [1, 4, 3, 2, 1]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 1], [1, 4, 3, 2, 1]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 1], [1, 4, 3, 2, 1]]]], dtype=torch.float).permute(0, 3, 1, 2)
y = torch.tensor([[[2, 3, 1], [2, 3, 1], [2, 3, 1]]], dtype=torch.int)
# x = torch.softmax(x, axis=1)
print(x.shape)
print(y.shape)
# torch.nn.functional.cross_entropy
loss = F.cross_entropy(x, y.long(), reduction='none').mean(1).mean(1)
print(loss)
x = argmax(x, axis=1)
print(x)

print()
print(loss.sum())
print(accuracy(x, y))
print(y.shape[0])
print(y.numel())
