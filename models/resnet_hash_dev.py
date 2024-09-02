# Adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import real_dev as real

class cosLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(cosLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        # customized initialization
        # init_factor = 2.0  # init_factor
        # init.kaiming_normal_(self.L.weight, a=init_factor, mode='fan_out', nonlinearity='relu')
        # print(self.L.weight.shape) # torch.Size([100, 640])
        # exit(0)
        self.scale = 0.09
        # print(init_factor)
        # exit(0)

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.000001)
        print(x_normalized.shape) 
        print(self.L.weight.shape) 
        exit(0)
        L_norm = torch.norm(self.L.weight, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized,weight_normalized.transpose(0,1))
        scores = cos_dist / self.scale
        return scores



class cosBinaryHashLinear(nn.Module):
    def __init__(self, indim, outdim, period):
        super(cosBinaryHashLinear, self).__init__()
        self.L = real.BinaryHashLinear(indim, outdim, period)
        # customized initialization
        # init_factor = 2.0  # init_factor
        # init.kaiming_normal_(self.L.weight, a=init_factor, mode='fan_out', nonlinearity='relu')
        # print(self.L.weight.shape) # torch.Size([100, 640])
        # exit(0)
        self.scale = 0.09
        # print(init_factor)
        # exit(0)

    def forward(self, x, task_id=None):
        if task_id is not None:
            o = self.L.o[:, int(task_id)]
            x = x*o
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.000001)
        # print(x_normalized.shape) #torch.Size([200, 512])
        # print(self.L.weight.shape) #torch.Size([512, 100])
        # exit(0)
        L_norm = torch.norm(self.L.weight, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized,weight_normalized.transpose(0,1))
        scores = cos_dist / self.scale
        return scores


class HashBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, period, stride=1):
        super(HashBasicBlock, self).__init__()
        self.conv1 = real.HashConv2d(in_planes, planes, 3, period, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv2 = real.HashConv2d(planes, planes, 3, period, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)

        self.shortcut = nn.ModuleList()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.ModuleList(
                [real.HashConv2d(in_planes, self.expansion*planes, 1, period, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False, track_running_stats=False)]
            )

    def forward(self, x, task_id):
        out = F.relu(self.bn1(self.conv1(x, task_id)))
        out = self.bn2(self.conv2(out, task_id))
        if len(self.shortcut) > 0:
            sout = self.shortcut[0](x, task_id)
            sout = self.shortcut[1](sout)
        else:
            sout = x
        out += sout 
        out = F.relu(out)
        return out


class StaticBNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(StaticBNBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn1_affine=True, bn1_track_stats=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=bn1_affine, track_running_stats=bn1_track_stats)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, time):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, None, []


class MultiOutResNet(nn.Module):
    def __init__(self, block, num_blocks, out_hash=True, num_classes=10):
        super(MultiOutResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if out_hash:
            self.linear = real.BinaryHashLinear(512*block.expansion,
                                              num_classes,
                                              1000)
        else:
            self.linear = real.MultiHeadLinear(512*block.expansion,
                                                num_classes,
                                                10)

        self.cheat_period = 1000000
        self.time_slow = 20000

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, time):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out, time)
        return out, None, []


class HashResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, num_tasks = 20, nf=64, bias=True):
        super(HashResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf, affine=False, track_running_stats=False)
        self.layer1 = nn.ModuleList(self._make_layer(block, nf * 1, num_blocks[0], stride=1, period=num_tasks))
        self.layer2 = nn.ModuleList(self._make_layer(block, nf * 2, num_blocks[1], stride=2, period=num_tasks))
        self.layer3 = nn.ModuleList(self._make_layer(block, nf * 4, num_blocks[2], stride=2, period=num_tasks))
        self.layer4 = nn.ModuleList(self._make_layer(block, nf * 8, num_blocks[3], stride=2, period=num_tasks))
        self.linear = real.BinaryHashLinear(nf * 8 *block.expansion, num_classes, period=num_tasks)
        self.pcrLinear = cosBinaryHashLinear(nf * 8 * block.expansion, num_classes, period=num_tasks)
        
        # self.cheat_period = 1000000
        # self.time_slow = 20000

    def _make_layer(self, block, planes, num_blocks, stride, period):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, period, stride))
            self.in_planes = planes * block.expansion
        return layers

    def features(self, x, task_id):
        # time = int(time)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out, task_id)
        out = self.layer1[1](out, task_id)
        out = self.layer2[0](out, task_id)
        out = self.layer2[1](out, task_id)
        out = self.layer3[0](out, task_id)
        out = self.layer3[1](out, task_id)
        out = self.layer4[0](out, task_id)
        out = self.layer4[1](out, task_id)
        # out = self.layer1[0](out, time, t)
        # out = self.layer1[1](out, time, t)
        # out = self.layer2[0](out, time, t)
        # out = self.layer2[1](out, time, t)
        # out = self.layer3[0](out, time, t)
        # out = self.layer3[1](out, time, t)
        # out = self.layer4[0](out, time, t)
        # out = self.layer4[1](out, time, t)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out, time, t)
        return out
    
    def forward(self, x, task_id=None):
        if task_id is None:
            task_id = 0

        out = self.features(x, task_id)
        logits = self.linear(out, task_id)
        # exit(0)
        return logits
    
    def pcrForward(self, x, task_id):
        out = self.features(x, task_id)
        # print(out.shape)
        # exit(0)
        logits = self.pcrLinear(out, task_id)
        # logits = self.linear(out)
        return logits, out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def StaticBNResNet18():
    return ResNet(StaticBNBasicBlock, [2,2,2,2],
                  bn1_affine=False, bn1_track_stats=False)


def OutHashResNet18():
    return MultiOutResNet(StaticBNBasicBlock, [2,2,2,2],
                          out_hash=True)


def MultiHeadResNet18():
    return MultiOutResNet(StaticBNBasicBlock, [2,2,2,2],
                          out_hash=False)

def HashResNet18(num_classes, num_tasks):
    return HashResNet(HashBasicBlock, [2,2,2,2], num_classes=num_classes, num_tasks=num_tasks)

def Reduced_HashResNet18(num_classes, num_tasks, nf=20):
    return HashResNet(HashBasicBlock, [2,2,2,2], num_classes=num_classes, num_tasks=num_tasks, nf = nf)


def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])


def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])


def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])


def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
