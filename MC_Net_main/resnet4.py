import math
from functools import partial
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [32, 64, 128, 256]
    # return [16, 32, 64, 128]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Basic_decorder_Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, if_up=True):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, in_planes, stride)
        # self.bn1 = nn.BatchNorm3d(in_planes)
        self.bn1 = nn.GroupNorm(num_groups=16, num_channels=in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(in_planes, in_planes)
        # self.bn2 = nn.BatchNorm3d(in_planes)
        self.bn2 = nn.GroupNorm(num_groups=16, num_channels=in_planes)
        self.conv3 = conv3x3x3(in_planes, in_planes)
        # self.bn3 = nn.BatchNorm3d(in_planes)
        self.bn3 = nn.GroupNorm(num_groups=16, num_channels=in_planes)
        self.conv4 = conv3x3x3(in_planes, in_planes, stride)
        # self.bn4 = nn.BatchNorm3d(in_planes)
        self.bn4 = nn.GroupNorm(num_groups=16, num_channels=in_planes)
        self.stride = stride
        self.ifup = if_up
        self.up_conv1 = nn.ConvTranspose3d(in_planes, planes, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up_conv2 = nn.ConvTranspose3d(in_planes, planes, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.up_bn = nn.BatchNorm3d(planes)
        self.up_bn = nn.GroupNorm(num_groups=16, num_channels=planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        if self.ifup:
            residual = self.up_conv1(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        if self.ifup:
            out = self.up_conv2(out)
            out = self.up_bn(out)
        else:
            out = self.conv3(out)
            out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='batchnorm', mode_upsampling = 0):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, kernel_size=stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2,
                 decorder_block=Basic_decorder_Block):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        # print(block_inplanes)
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels, # 1
                               self.in_planes, # 16
                               kernel_size=(2, 5, 7),
                               stride=(2, 2, 2),
                               padding=(0, 2, 3),
                               bias=False)
        self.upConv1 = nn.ConvTranspose3d(32,6,(2,2,2),(2,2,2))
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.decorder4=decorder_block(256,128)
        self.decorder3=decorder_block(128,64)
        self.decorder2=decorder_block(64,32)
        self.decorder1=decorder_block(32,32,if_up=False)


        self.decorder2_1 = nn.ConvTranspose3d(256, 128, (2,2,2), (2,2,2))
        self.decorder2_2 = nn.ConvTranspose3d(128, 64, (2,2,2), (2,2,2))
        self.decorder2_3 = nn.ConvTranspose3d(64, 32, (2,2,2), (2,2,2))
        self.decorder2_4 = nn.ConvTranspose3d(32,6,(2,2,2),(2,2,2))

        self.avgpool3D = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            upsample = nn.Sequential(
                conv1x1x1(planes * block.expansion, self.in_planes,  stride),
                nn.BatchNorm3d(self.in_planes))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  upsample=upsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if not self.no_max_pool:
            x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        res = [x1,x2,x3,x4]

        return res

    def decoder(self, res):
        x = self.decorder4(res[3])
        x = self.decorder3(x+res[2])
        x = self.decorder2(x+res[1])
        x = self.decorder1(x+res[0])

        x = self.upConv1(x)

        return x

    def decoder_2(self, res):
        x = self.decorder2_1(res[3])
        x = self.decorder2_2(x+res[2])
        x = self.decorder2_3(x+res[1])
        x = self.decorder2_4(x+res[0])

        return x

    def forward(self, x):
        res = self.encoder(x)
        x_0 = self.decoder(res)
        x_1 = self.decoder_2(res)

        return x_0,x_1


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
if __name__ == '__main__':
    x = torch.Tensor(1,1, 48, 128, 224)
    model = ResNet(BasicBlock,  [2, 2, 2, 2], get_inplanes(),n_input_channels=1)
    # model = generate_model(34,n_input_channels=1,n_classes=2)
    y = model(x)
    # print(model)
    # summary(model,(1,48,128,224),30)
    print(y[0].shape)