from functools import partial

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

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm='BN'):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        if norm == 'BN':
            self.norm1 = nn.BatchNorm3d(planes)
            self.norm2 = nn.BatchNorm3d(planes)
        else:
            self.norm1 = nn.GroupNorm(16, planes)
            self.norm2 = nn.GroupNorm(16, planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

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
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=0):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
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
                 self_adapt_w=True,
                 default_w=0.3,
                 no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2,
                 scale=3,
                 norm='BN'):
        super().__init__()
        self.scale = scale
        self.norm = norm
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        # print(block_inplanes)
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,  # 1
                               self.in_planes,  # 16
                               kernel_size=(2, 5, 7),
                               stride=(2, 2, 2),
                               padding=(0, 2, 3),
                               bias=False)
        # self.upConv1 = nn.ConvTranspose3d(32, 6, (2, 2, 2), (2, 2, 2))
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
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

        self.decorder2_1 = nn.ConvTranspose3d(256, 128, (2, 2, 2), (2, 2, 2))
        self.decorder2_2 = nn.ConvTranspose3d(128, 64, (2, 2, 2), (2, 2, 2))
        self.decorder2_3 = nn.ConvTranspose3d(64, 32, (2, 2, 2), (2, 2, 2))
        self.decorder2_4 = nn.ConvTranspose3d(32, 6, (2, 2, 2), (2, 2, 2))

        self.avgpool3D = nn.AdaptiveAvgPool3d((1, 1, 1))
        if self.scale == 3:
            self.fc = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion * 2, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256, n_classes))
        elif self.scale == 1:
            self.fc = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128, n_classes))
        elif self.scale == 5:
            self.fc = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion * 2 + 128, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256, n_classes))
        # self.fc_2 = nn.Linear(256,n_classes)
        # self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.adapt_w = self_adapt_w
        self.w = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
        self.default_w = torch.tensor([default_w, 1 - default_w]).cuda()

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
                if self.norm == 'BN':
                    downsample = nn.Sequential(
                        conv1x1x1(self.in_planes, planes * block.expansion, stride),
                        nn.BatchNorm3d(planes * block.expansion))
                else:
                    downsample = nn.Sequential(
                        conv1x1x1(self.in_planes, planes * block.expansion, stride),
                        nn.GroupNorm(16, planes * block.expansion))

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
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.ConvTranspose3d(planes * block.expansion, self.in_planes, stride=stride),
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

    def DiceLoss(self, output, target, eps=1e-5):  # soft dice loss

        target = target.float()

        num = 2 * (output * target).sum() + eps
        den = output.sum() + target.sum() + eps

        return 1.0 - num / den

    def softmax_dice_loss(self, output, target, alpha=1e-5):
        output = F.softmax(output, dim=1)
        A = torch.unique(target)
        # for i in range(target.shape[0]):
        #     assert len(A)>1, '无标签'
        # torch.unique:去除数组中的重复数字，并进行排序之后输出。
        loss = 0
        for i in A:
            i = i.item()
            loss += self.DiceLoss(output[:, i, ...], (target == i).float(), eps=alpha)

        diceloss = loss / len(A)

        return diceloss

    def forward_loss(self, input, input_cls, seg_label, cls_label):
        seg_label = torch.squeeze(seg_label)

        # loss_seg
        loss_dice = self.softmax_dice_loss(input, seg_label) + self.cross_entropy(input, seg_label)
        loss_seg = loss_dice
        loss_seg = loss_seg.reshape(1, 1)

        # loss_cls
        loss_cls = self.cross_entropy(input_cls, cls_label)
        loss_cls = loss_cls.reshape(1, 1)
        loss = torch.cat((loss_cls, loss_seg), dim=1)

        if self.adapt_w:
            loss = 0.5 / (self.w[0] ** 2) * loss_seg + 0.5 / (self.w[1] ** 2) * loss_cls + torch.log(
                self.w[0] * self.w[1])
        else:
            w = self.default_w
            loss = (loss * w).sum()
        every_loss = [loss_dice, loss_cls]
        return loss, every_loss

    def encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32

        if not self.no_max_pool:
            x = self.maxpool(x)
        x1 = self.layer1(x)  # 32
        x2 = self.layer2(x1)  # 64
        x3 = self.layer3(x2)  # 128
        x4 = self.layer4(x3)  # 256

        res = [x1, x2, x3, x4]

        return res

    def decoder_2(self, res):
        x5 = self.decorder2_1(res[-1])
        x6 = self.decorder2_2(x5 + res[2])
        x7 = self.decorder2_3(x6 + res[1])
        x8 = self.decorder2_4(x7+res[0])

        return [x5, x6, x7, x8]

    def scale_cls(self, scale):
        features = []
        for feature in scale:
            feature = self.avgpool3D(feature)
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.shape[0], -1)
        return features

    def forward(self, x=None, seg_label=None, cls_label=None, if_train=True):
        [x1, x2, x3, x4] = self.encoder(x)
        [x5, x6, x7, x8] = self.decoder_2([x1, x2, x3, x4])

        if self.scale == 3:
            feature = self.scale_cls([x3, x4, x5])
        elif self.scale == 1:
            feature = self.scale_cls([x4])
        else:
            feature = self.scale_cls([x2, x3, x4, x5, x6])
        x_cls = self.sigmoid(self.fc(feature))

        loss = 0
        every_loss = 0
        if if_train:
            loss, every_loss = self.forward_loss(x8, x_cls, seg_label, cls_label)

        x_cls = self.softmax(x_cls)

        return x_cls, x8, loss, every_loss, feature


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
    # x = torch.Tensor(1,1, 48, 128, 224).cuda()
    x = torch.Tensor(2, 1, 80, 144, 272)
    seg_label = torch.Tensor(2, 1, 80, 144, 272).long()
    cls_label = torch.tensor([0, 1])
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=1, n_classes=2)
    # model = generate_model(34,n_input_channels=1,n_classes=2)
    y = model(x, seg_label, cls_label)
    # print(model)
    # summary(model,(1,48,128,224),30)
    print(y[0].shape)
