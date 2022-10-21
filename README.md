# Fully Convolutional Network

### 20221021更新：

- 完成了nets下神经网络搭建的部分，包含VGG和FCN；
- 完成了VOC标签的划分代码，VOC_annotation.py；
- 完成了utils下数据加载功能，具体为dataloader.py；
- 完成了训练部分的代码编写，为train.py；



发现了一个问题，在主干网络使用预权重进行初始化后，如果在FCN中通过遍历对权重进行初始化会导致原有权重被覆盖，无法训练起来。
