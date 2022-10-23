import torch
import torch.nn as nn
import numpy as np

from nets.vgg import VGG16

class FCN8(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False):
        super(FCN8, self).__init__()
        self.num_classes = num_classes
        self.backbone = VGG16(pretrained=pretrained)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.conv7 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.pool3_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.pool4_conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.upsample_4 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=8, stride=4, padding=2)
        self.upsample_8 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, padding=4)
        # self.softmax = nn.Softmax(dim=1)

        self.freeze_backbone()
        self._initialize_weights()
        self.unfreeze_backbone()

    # 测试池化层输出的尺寸
    # def test(self):
    #     x = torch.rand(size=(1, 3, 256, 256))
    #     ret = self.forward(x)
    #     print(ret.shape)

    # 这一步会将VGG的预训练权重覆盖，需要对代码进行重新编写
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        x = x.float()
        [feat1, feat2, feat3, feat4, feat5] = self.backbone.forward(x)
        # 将pool3的通道数降为类别数便于计算
        feat3_conv = self.pool3_conv(feat3)
        conv6 = self.drop6(self.relu6(self.conv6(feat5)))
        conv7 = self.conv7(conv6)
        conv7_4x = self.upsample_4(conv7)
        feat4_conv = self.pool4_conv(feat4)
        pool4_2x = self.upsample_2(feat4_conv)
        feat_combine = feat3_conv + pool4_2x + conv7_4x
        fcn_8x = self.upsample_8(feat_combine)
        # fcn_8x = self.softmax(fcn_8x)

        return fcn_8x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

# Make a 2D bilinear kernel suitable for upsampling
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center))
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


# model = FCN8()
