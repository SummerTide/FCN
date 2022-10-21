import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch

cfgs = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

ranges = {
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))
}

class VGG(nn.Module):
    def __init__(self, features, num_classes = 21):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )
        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # 用于提取特征图，分别对应
        feature_return = []
        for index, (begin, end) in enumerate(ranges['vgg16']):
            x = self.features[begin:end](x)
            feature_return.append(x)
        return feature_return
        # pool1 = self.features[:5](x)
        # pool2 = self.features[5:10](pool1)
        # pool3 = self.features[10:17](pool2)
        # pool4 = self.features[17:24](pool3)
        # pool5 = self.features[24:](pool4)
        # return [pool1, pool2, pool3, pool4, pool5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # 测试池化层输出的尺寸
    # torch.Size([1, 64, 128, 128])
    # torch.Size([1, 128, 64, 64])
    # torch.Size([1, 256, 32, 32])
    # torch.Size([1, 512, 16, 16])
    # torch.Size([1, 512, 8, 8])
    # def test(self):
    #     x = torch.rand(size=(1, 3, 256, 256))
    #     ret = self.forward(x)
    #     for i in ret:
    #         print(i.shape)



def make_layers(cfg, batch_norm = False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def VGG16(pretrained, in_channels = 3, **kwargs):
    model = VGG(make_layers(cfgs["vgg16"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data/")
        model.load_state_dict(state_dict)
    del model.avgpool
    del model.classifier
    # print(model)
    return model

# net = VGG16(False)
# net.test()



