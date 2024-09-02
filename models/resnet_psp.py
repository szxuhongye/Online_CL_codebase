import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Define BasicBlock and ReducedResNet18 classes (as per the previous code)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

        L_norm = torch.norm(self.L.weight, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized,weight_normalized.transpose(0,1))
        scores = cos_dist / self.scale
        return scores

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, use_PSP=False, contexts=None, task_id=None, layer_index=0):
        if use_PSP:
            context_matrix_1 = torch.from_numpy(np.reshape(contexts[task_id][layer_index],
                                                           newshape=self.conv1.weight.cpu().size()).astype(np.float32)).cuda()
            context_matrix_2 = torch.from_numpy(np.reshape(contexts[task_id][layer_index + 1],
                                                           newshape=self.conv2.weight.cpu().size()).astype(np.float32)).cuda()
            out = F.relu(self.bn1(F.conv2d(x, self.conv1.weight * context_matrix_1, self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)))
            out = self.bn2(F.conv2d(out, self.conv2.weight * context_matrix_2, self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

        shortcut = self.shortcut(x)
        out += shortcut
        out = F.relu(out)
        return out

class Reduced_ResNet18_psp(nn.Module):
    def __init__(self, num_classes=10, nf=20, bias=True):
        super(Reduced_ResNet18_psp, self).__init__()
        self.in_planes = nf

        # Initial layers
        self.conv1 = conv3x3(3, nf)
        self.bn1 = nn.BatchNorm2d(nf)

        # Layer 1
        self.layer1_conv1 = BasicBlock(nf, nf)
        self.layer1_conv2 = BasicBlock(nf, nf)

        # Layer 2
        self.layer2_conv1 = BasicBlock(nf, nf * 2, stride=2)
        self.layer2_conv2 = BasicBlock(nf * 2, nf * 2)

        # Layer 3
        self.layer3_conv1 = BasicBlock(nf * 2, nf * 4, stride=2)
        self.layer3_conv2 = BasicBlock(nf * 4, nf * 4)

        # Layer 4
        self.layer4_conv1 = BasicBlock(nf * 4, nf * 8, stride=2)
        self.layer4_conv2 = BasicBlock(nf * 8, nf * 8)

        # Linear layer
        # self.linear = nn.Linear(nf * 8 * BasicBlock.expansion, num_classes, bias=bias)
        self.linear = nn.Linear(nf * 8 * BasicBlock.expansion, num_classes, bias=bias)
        self.pcrLinear = cosLinear(nf * 8 * BasicBlock.expansion, num_classes)

    def features(self, x, use_PSP=False, contexts=None, task_id=None):
        if use_PSP:
            layer_index = 0
            # print(x.shape)

            context_matrix_1 = torch.from_numpy(np.reshape(contexts[task_id][layer_index],
                                                           newshape=self.conv1.weight.cpu().size()).astype(np.float32)).cuda()
            x = F.relu(self.bn1(F.conv2d(x, self.conv1.weight * context_matrix_1, self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)))
            # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            layer_index += 1
            # print(x.shape)

            x = self.layer1_conv1(x, use_PSP, contexts, task_id, layer_index)
            layer_index += 2  # Each BasicBlock uses two convolutional layers
            x = self.layer1_conv2(x, use_PSP, contexts, task_id, layer_index)
            layer_index += 2
            # print(x.shape)

            x = self.layer2_conv1(x, use_PSP, contexts, task_id, layer_index)
            layer_index += 2
            x = self.layer2_conv2(x, use_PSP, contexts, task_id, layer_index)
            layer_index += 2
            # print(x.shape)

            x = self.layer3_conv1(x, use_PSP, contexts, task_id, layer_index)
            layer_index += 2
            x = self.layer3_conv2(x, use_PSP, contexts, task_id, layer_index)
            layer_index += 2
            # print(x.shape)

            x = self.layer4_conv1(x, use_PSP, contexts, task_id, layer_index)
            layer_index += 2
            x = self.layer4_conv2(x, use_PSP, contexts, task_id, layer_index)
            # print(x.shape)
            # exit(0)

            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)

            # context_matrix_fc = torch.from_numpy(np.diag(contexts[task_id][layer_index]).astype(np.float32)).cuda()
            # x = torch.matmul(x, context_matrix_fc)
            # x = self.linear(x)
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

            # print(x.shape)

            x = self.layer1_conv1(x)
            x = self.layer1_conv2(x)
            # print(x.shape)

            x = self.layer2_conv1(x)
            x = self.layer2_conv2(x)
            # print(x.shape)

            x = self.layer3_conv1(x)
            x = self.layer3_conv2(x)
            # print(x.shape)

            x = self.layer4_conv1(x)
            x = self.layer4_conv2(x)
            # print(x.shape)
            # exit(0)

            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            # print(x.shape)
            # exit(0)
            # x = self.linear(x)

        return x
    

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x, use_PSP=False, contexts=None, task_id=None):
        out = self.features(x, use_PSP, contexts, task_id)
        logits = self.logits(out)
        return logits
    

    def pcrForward(self, x, use_PSP=False, contexts=None, task_id=None):
        out = self.features(x, use_PSP, contexts, task_id)
        # print(out.shape)
        # exit(0)
        logits = self.pcrLinear(out)
        # logits = self.linear(out)
        return logits, out
    
