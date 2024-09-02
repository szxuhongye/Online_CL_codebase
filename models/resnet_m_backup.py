"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

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
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)
        self.pcrLinear = cosLinear(nf * 8 * block.expansion, num_classes)
        # self.online_predictor = MLP(nf * 8 * block.expansion, nf * 8 * block.expansion, nf * 4 * block.expansion)
        # self.online_predictor = MLP(nf * 8 * block.expansion, nf * 8 * block.expansion, nf * 8 * block.expansion)
        # self.attention = HierarchicalCrossAttention(nf * 8 * block.expansion, num_classes, 10)
        # self.attention2 = SimplifiedCrossAttention(nf * 8 * block.expansion, num_classes)

        # cluster_params
        self.temperature = 0.1
        self.update_subcentroids = True
        self.gamma = 0.999
        self.pretrain_subcentroids = False
        self.use_subcentroids = True
        self.centroid_contrast_loss = False
        self.centroid_contrast_loss_weights = 0.005
        self.in_channels = 512
        self.num_classes = num_classes
        self.expand_dim = False
        ####### memory_bank added #######
        self.isOnlyCE = True
        self.memory_bank = []
        self.memory_bank_label = []
        
        
        # 500 for swin-T # ResNet for 1000 # for swin-B 300 # 400 for swin-s #1000 for mobilenet-v2
        self.BS_num_limit = 500 
        print('self.BS_num_limit', self.BS_num_limit)

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        
        
        '''set the number of sub-centroids here, 4 for best performance'''
        self.num_subcentroids = 4 
        print('subcentroids_num:', self.num_subcentroids)

        # 512 for resnet18, 2048 for resnet50
        # if turn 512, then no dimension expand (2048 for expansion for resnet18)
        # 2048 for imagenet without expanding dimension
        # 1024 for swin transformer without expanding dimension
        embedding_dim = self.in_channels 
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_subcentroids, embedding_dim),
                                       requires_grad=self.pretrain_subcentroids)

        # CK times embedding_dim
        self.feat_norm = nn.LayerNorm(embedding_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        self.apply(init_weights)
        trunc_normal_(self.prototypes, std=0.02)



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print("out.shape: ", out.shape)
        # exit(0)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.online_predictor(x)
        out = self.linear(x)
        return out

    def forward(self, x):
        out = self.features(x)
        print("out.shape: ", out.shape)
        exit(0)
        # logits = self.pcrLinear(out)
        logits = self.attention(out)
        # print("logits.shape: ", logits.shape)
        # exit(0)
        return logits, out
    
    def forward2(self, x):
        out = self.features(x)
        # print("out.shape: ", out.shape)
        # logits = self.pcrLinear(out)
        logits = self.attention2(out)
        # print("logits.shape: ", logits.shape)
        # exit(0)
        return logits, out

    def pcrForward(self, x):
        # out = self.features(x)
        logits = self.pcrLinear(x)
        return logits


def Reduced_ResNet18_m(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

def Reduced_ResNet34_m(nclasses, nf=20, bias=True):
    """
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)

def Reduced_ResNet50_m(nclasses, nf=20, bias=True):
    """
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)

def Reduced_ResNet101_m(nclasses, nf=20, bias=True):
    """
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)

def Reduced_ResNet152_m(nclasses, nf=20, bias=True):
    """
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)

def ResNet18(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)

def ResNet50(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)


def ResNet101(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)


def ResNet152(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)



def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)










































class HierarchicalCrossAttention(nn.Module):
    def __init__(self, input_dim, num_clusters, num_subclasses):
        super(HierarchicalCrossAttention, self).__init__()

        self.num_clusters = num_clusters
        self.num_subclasses = num_subclasses
        self.input_dim = input_dim

        # Main class centroids
        self.class_centroids = nn.Parameter(torch.empty(num_clusters, input_dim))
        nn.init.kaiming_normal_(self.class_centroids, mode='fan_out', nonlinearity='relu')

        # subclass centroids
        self.subclass_centroids = nn.Parameter(torch.empty(num_clusters, num_subclasses, input_dim))
        nn.init.kaiming_normal_(self.subclass_centroids, mode='fan_out', nonlinearity='relu')

        # Attention weights for combining subclass representations
        self.attention_weights = nn.Parameter(torch.empty(num_subclasses, input_dim))
        nn.init.kaiming_normal_(self.attention_weights, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: (batch_size, input_dim)

        # Step 1: Interaction with subclass centroids
        # Reshape subclass centroids and input for batch processing
        subclass_centroids_reshaped = self.subclass_centroids.view(-1, self.input_dim)  # (num_clusters * num_subclasses, input_dim)

        subclass_scores = torch.einsum('ik,lk->il', x, subclass_centroids_reshaped)  # (batch_size, num_clusters * num_subclasses)

        # Calculate attention across all subclasses per cluster
        attn_weights_sub = F.softmax(subclass_scores.view(-1, self.num_clusters, self.num_subclasses), dim=2)  # (batch_size, num_clusters, num_subclasses)

        # Step 2: Aggregate subclass representations with attention
        attended_subclasses = torch.einsum('bnm,mk->bnk', attn_weights_sub, self.attention_weights)  # (batch_size, num_clusters, input_dim)

        # Step 3: Interaction with main class centroids
        similarity_scores = torch.einsum('bik,kj->bij', attended_subclasses, self.class_centroids.t())  # (batch_size, num_clusters, num_clusters)

        cluster_scores = similarity_scores.diagonal(dim1=-2, dim2=-1).squeeze(-1)  # (batch_size, num_clusters)

        return cluster_scores



class SimplifiedCrossAttention(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super(SimplifiedCrossAttention, self).__init__()

        self.num_clusters = num_clusters
        self.input_dim = input_dim

        # Main class centroids
        self.class_centroids = nn.Parameter(torch.empty(num_clusters, input_dim))
        nn.init.kaiming_normal_(self.class_centroids, mode='fan_out', nonlinearity='relu')

        # Attention weights for combining class representations
        # Attention weights for combining class representations
        self.attention_weights = nn.Parameter(torch.empty(input_dim,))  # This creates a 1D tensor
        torch.nn.init.uniform_(self.attention_weights, a=0.0, b=1.0)  # This uses a uniform distribution

    def forward(self, x):
        # x: (batch_size, input_dim)

        # Step 1: Calculate attention scores
        attn_scores = torch.einsum('ij,j->i', x, self.attention_weights)  # (batch_size,)

        # Apply softmax to get the normalized attention weights
        attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(-1)  # (batch_size, 1)

        # Step 2: Apply attention weights to the input features
        attended_x = x * attn_weights  # (batch_size, input_dim)

        # Step 3: Interaction with main class centroids
        cluster_scores = torch.einsum('ik,kj->ij', attended_x, self.class_centroids.t())  # (batch_size, num_clusters)

        return cluster_scores


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=160, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.encoder = Reduced_ResNet18_m(100)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            feat = F.normalize(self.head(feat), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def features(self, x):
        return self.encoder.features(x)
    


@torch.no_grad()
def normalize_weight(weight: nn.Parameter, dim: int = 1, keepdim: bool = True):
    """Normalizes the weight to unit length along the specified dimension."""
    weight.div_(torch.norm(weight, dim=dim, keepdim=keepdim))

class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])

    """

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.

        """
        return self.layers(x)


class SwaVProjectionHead(ProjectionHead):
    """Projection head used for SwaV.

    [0]: SwAV, 2020, https://arxiv.org/abs/2006.09882
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ):
        super(SwaVProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SwaVPrototypes(nn.Module):
    """Multihead Prototypes used for SwaV.

    Each output feature is assigned to a prototype, SwaV solves the swapped
    prediction problem where the features of one augmentation are used to
    predict the assigned prototypes of the other augmentation.

    Attributes:
        input_dim:
            The input dimension of the head.
        n_prototypes:
            Number of prototypes.
        n_steps_frozen_prototypes:
            Number of steps during which we keep the prototypes fixed.

    Examples:
        >>> # use features with 128 dimensions and 512 prototypes
        >>> prototypes = SwaVPrototypes(128, 512)
        >>>
        >>> # pass batch through backbone and projection head.
        >>> features = model(x)
        >>> features = nn.functional.normalize(features, dim=1, p=2)
        >>>
        >>> # logits has shape bsz x 512
        >>> logits = prototypes(features)

    """

    def __init__(
        self,
        input_dim: int = 128,
        n_prototypes: Union[List[int], int] = 3000,
        n_steps_frozen_prototypes: int = 0,
    ):
        super(SwaVPrototypes, self).__init__()
        # Default to a list of 1 if n_prototypes is an int.
        self.n_prototypes = (
            n_prototypes if isinstance(n_prototypes, list) else [n_prototypes]
        )
        self._is_single_prototype = True if isinstance(n_prototypes, int) else False
        self.heads = nn.ModuleList(
            [nn.Linear(input_dim, prototypes) for prototypes in self.n_prototypes]
        )
        # print(self.heads)
        # exit(0)
        self.n_steps_frozen_prototypes = n_steps_frozen_prototypes

    def forward(self, x, step=None) -> Union[torch.Tensor, List[torch.Tensor]]:
        self._freeze_prototypes_if_required(step)
        out = []
        for layer in self.heads:
            out.append(layer(x))
        return out[0] if self._is_single_prototype else out

    def normalize(self):
        """Normalizes the prototypes so that they are on the unit sphere."""
        for layer in self.heads:
            normalize_weight(layer.weight)

    def _freeze_prototypes_if_required(self, step):
        if self.n_steps_frozen_prototypes > 0:
            if step is None:
                raise ValueError(
                    "`n_steps_frozen_prototypes` is greater than 0, please"
                    " provide the `step` argument to the `forward()` method."
                )
            self.requires_grad_(step >= self.n_steps_frozen_prototypes)


class SinkhornClusteringLayer(nn.Module):
    def __init__(self, input_dim, main_classes, sub_classes, reg=0.1, num_iters=5, epsilon=1e-3, momentum=0.99):
        super(SinkhornClusteringLayer, self).__init__()
        self.reg = reg
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.momentum = momentum
        # Initialize clustering weights using Kaiming Normal initialization
        self.cluster_weights = nn.Parameter(torch.Tensor(input_dim, main_classes, sub_classes))
        init.kaiming_normal_(self.cluster_weights, mode='fan_out', nonlinearity='relu')
        self.cluster_weights.requires_grad = False

    def sinkhorn_torch(self, Q):
        # Regularization term
        reg = self.reg

        # Initialize u, v
        u = torch.ones(Q.size(1), Q.size(2), device=Q.device)
        v = torch.ones(Q.size(1), Q.size(2), device=Q.device)

        # Sinkhorn iterations
        for _ in range(self.num_iters):
            u_prev = u.clone()

            # Update u and v
            u = 1.0 / torch.matmul(Q, v.unsqueeze(-2)).squeeze(-2)
            v = 1.0 / torch.matmul(Q.transpose(1, 2), u.unsqueeze(-2)).squeeze(-2)

            # Check for convergence
            if torch.max(torch.abs(u - u_prev)) < self.epsilon:
                break

        # Update Q with Sinkhorn balancing
        Q = Q * u.unsqueeze(1) * v.unsqueeze(0)
        return Q

    def forward(self, x):
        # Dot product with clustering layer
        cluster_output = torch.matmul(x, self.cluster_weights)

        # Find the main class with maximum probability
        main_class_probabilities, main_class_indices = cluster_output.max(dim=-1)

        # Update the clustering layer with Sinkhorn and momentum
        new_cluster_weights = self.sinkhorn_torch(self.cluster_weights)
        self.cluster_weights.data = self.momentum * self.cluster_weights.data + (1 - self.momentum) * new_cluster_weights.data

        return main_class_indices, main_class_probabilities

# # Example usage
# input_dim, main_classes, sub_classes = 160, 100, 10
# model = SinkhornClusteringLayer(input_dim, main_classes, sub_classes)

# # Dummy input (output from backbone)
# backbone_output = torch.randn(20, 160)

# # Forward pass
# labels, probabilities = model(backbone_output)
