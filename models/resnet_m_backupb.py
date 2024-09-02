"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
"""
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
import math

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
    def __init__(self, block, num_blocks, num_classes, nf, bias, memory_bank_size):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)
        # self.pcrLinear = cosLinear(nf * 8 * block.expansion, num_classes)
        # print(self.pcrLinear.L.weight.shape)
        # exit(0)
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
        self.num_classes = num_classes
        self.expand_dim = False
        ####### memory_bank added #######
        self.isOnlyCE = True
        self.memory_bank = []
        self.memory_bank_label = []

        # 500 for swin-T # ResNet for 1000 # for swin-B 300 # 400 for swin-s #1000 for mobilenet-v2
        self.BS_num_limit = memory_bank_size 
        print('self.BS_num_limit', self.BS_num_limit)

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        
        
        '''set the number of sub-centroids here, 4 for best performance'''
        self.num_subcentroids = 10 
        print('subcentroids_num:', self.num_subcentroids)

        # 512 for resnet18, 2048 for resnet50
        # if turn 512, then no dimension expand (2048 for expansion for resnet18)
        # 2048 for imagenet without expanding dimension
        # 1024 for swin transformer without expanding dimension
        self.embedding_dim = nf * 8 * block.expansion
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_subcentroids, self.embedding_dim),
                                       requires_grad=self.pretrain_subcentroids)

        # CK times self.embedding_dim
        self.feat_norm = nn.LayerNorm(self.embedding_dim)
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
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.online_predictor(x)
        out = self.linear(x)
        return out

    def forward2(self, x):
        out = self.features(x)
        print("out.shape: ", out.shape)
        exit(0)
        # logits = self.pcrLinear(out)
        logits = self.attention(out)
        # print("logits.shape: ", logits.shape)
        # exit(0)
        return logits, out


    def pcrForward(self, x):
        # out = self.features(x)
        logits = self.pcrLinear(x)
        return logits

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    @staticmethod
    def momentum_update(old_value, new_value, momentum, debug=False):
        update = momentum * old_value + (1 - momentum) * new_value
        if debug:
            print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
                torch.norm(update, p=2)))
        return update

    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def subcentroids_learning(self, _c, out_seg, gt_seg, masks): 
        # masks: n x m x k, _c: n x d ,self.prototypes: k x m x d, out_seg: n x k
        # k: num_class, m: num_subcentroids # n should turn into (# batch* batch_size)
        # find pixels that are correctly classified
        pred_seg = torch.max(out_seg, 1)[1] # out_seg.shape = batch_size x class number
        
        mask = (gt_seg == pred_seg.view(-1))

        # compute cosine similarity
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t()) #.cpu() # .cpu() ##

        # compute logits and apply temperature
        centroid_logits = cosine_similarity / self.temperature
        centroid_target = gt_seg.clone().float()

        # clustering for each class
        centroids = self.prototypes.data.clone()
        for k in range(self.num_classes):
            # get initial assignments for the k-th class
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            # clustering q.shape = n x self.num_subcentroids
            # q, indexs = distributed_sinkhorn(init_q) ##
            q, indexs = AGD_torch_no_grad_gpu(init_q)

            # binary mask for pixels of the k-th class
            # (1: correctly classified, 0: incorrectly classified)
            m_k = mask[gt_seg == k]

            # feature embedding for pixels of the k-th class
            c_k = _c[gt_seg == k, ...]

            # tile m_k to have the same shape as q
            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_subcentroids) # n x self.num_subcentroids

            # find pixels that are assigned to each subcentroids as well as correctly classified
            # element-wise dimension doesn't change as m_k_tile
            m_q = q * m_k_tile  # n x self.num_subcentroids 

            # tile m_k to have the same shape as c_k
            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            # find pixels with label k that are correctly classified
            c_q = c_k * c_k_tile  # n x self.embedding_dim # c_k and c_k_tile same dimension

            # summarize feature embeddings # matrix multi
            f = m_q.transpose(0, 1) @ c_q  # self.num_subcentroids x self.embedding_dim

            # num assignments for each subcentroids
            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_subcentroids is True:
                # normalize embeddings
                f = F.normalize(f, p=2, dim=-1)

                # update subcentroids (equation 14) # all in k loop therefore specific [n!=0, :]
                new_value = self.momentum_update(old_value=centroids[k, n != 0, :], new_value=f[n != 0, :], momentum=self.gamma, debug=False)
                centroids[k, n != 0, :] = new_value # for every k, update once

            centroid_target[gt_seg == k] = indexs.float() + (self.num_subcentroids * k)

        self.prototypes = nn.Parameter(F.normalize(centroids, p=2, dim=-1),
                                       requires_grad=self.pretrain_subcentroids)

        # sync across gpus
        # if self.use_subcentroids is True and dist.is_available() and dist.is_initialized():
        #     centroids = self.prototypes.data.clone()
        #     dist.all_reduce(centroids.div_(dist.get_world_size()))
        #     self.prototypes = nn.Parameter(centroids, requires_grad=self.pretrain_subcentroids)

        return centroid_logits, centroid_target

    def forward(self, x, gt_label, **kwargs):

        fea = self.features(x)
        '''for subcentroids training, this this'''
        inputs = self.pre_logits(fea) # (batch_size x 512)

        batch_size = inputs.shape[0]

        if self.isOnlyCE is True:
            #Save to memory bank
            BS_num = self.dequeue_and_enqueue(inputs, gt_label, batch_size)
            
            if BS_num>=self.BS_num_limit:
                # print("BS_num:", BS_num)
                self.isOnlyCE = False
            else:
                seg_logits = self.forward_train(inputs, gt_label)
                losses = F.cross_entropy(seg_logits, gt_label)

            
        if self.pretrain_subcentroids is False and self.use_subcentroids is True and self.isOnlyCE is False:
        
            # concat data from memory_bank
            inputs = torch.cat(self.memory_bank,0)
            gt_label = torch.cat(self.memory_bank_label,0)

            seg_logits, contrast_logits, contrast_target = self.forward_train(inputs, gt_label=gt_label)
            # seg_logits.shape=batch_size x 10  gt_label.shape=batch_size
            # losses = F.cross_entropy(seg_logits, gt_label, **kwargs)
            
            #release memory bank space
            self.memory_bank = []
            self.memory_bank_label = []

            if self.centroid_contrast_loss is True and self.isOnlyCE is False: # changes here: and self.isOnlyCE is False: # Only happens apply once.
                 loss_centroid_contrast = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=255)
                 losses['loss_centroid_contrast'] = loss_centroid_contrast * self.centroid_contrast_loss_weights
            
            #initialize isOnlyCE to True
            self.isOnlyCE = True
        # return seg_logits, fea, losses
        return seg_logits, fea

    def forward_train(self, x, gt_label=None):
        """Forward function for both train and test."""
        # get higher dimension if comment out, no dimension expanded

        # print("x_shape", x.shape)
        x = self.feat_norm(x)
        x = self.l2_normalize(x)

        # should add x here.

        self.prototypes.data.copy_(self.l2_normalize(self.prototypes))

        # n: h*w, k: num_class, m: num_subcentroids # n should turn into (# batch* batch_size)
        masks = torch.einsum('nd,kmd->nmk', x, self.prototypes) # originally nmk
        
        out_cls = torch.amax(masks, dim=1)   

        out_cls = self.mask_norm(out_cls)

        if self.pretrain_subcentroids is False and self.use_subcentroids is True and gt_label is not None and self.isOnlyCE is False:
            contrast_logits, contrast_target = self.subcentroids_learning(x, out_cls, gt_label, masks)
            return out_cls, contrast_logits, contrast_target

        else:
            return out_cls
    
    def dequeue_and_enqueue(self, inputs, gt_label, batch_size):

        # inputs_all = concat_all_gather(inputs)
        # label_all = concat_all_gather(gt_label)

        # self.memory_bank.append(inputs_all)
        # self.memory_bank_label.append(label_all)

        self.memory_bank.append(inputs)
        self.memory_bank_label.append(gt_label)

        return len(self.memory_bank)
    
    def custom_deepcopy(self):
        # 创建一个新的 ResNet 实例，不带任何初始化参数
        new_model = ResNet.__new__(ResNet)

        # 复制所有可以被 deepcopy 的属性
        for name, value in self.__dict__.items():
            if name not in ["memory_bank", "memory_bank_label"]:
                setattr(new_model, name, copy.deepcopy(value))

        # 对于不需要复制的属性，可以选择不设置，或者赋予默认值
        new_model.memory_bank = []
        new_model.memory_bank_label = []
        new_model.prototypes = None  # 或者根据需要给一个合适的初始值

        return new_model

    
    


def Reduced_ResNet18_m(nclasses, nf=20, bias=True, memory_bank_size = 500):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, memory_bank_size)

def Reduced_ResNet34_m(nclasses, nf=20, bias=True, memory_bank_size = 500):
    """
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias, memory_bank_size)

def Reduced_ResNet50_m(nclasses, nf=20, bias=True, memory_bank_size = 500):
    """
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias, memory_bank_size)

def Reduced_ResNet101_m(nclasses, nf=20, bias=True, memory_bank_size = 500):
    """
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias, memory_bank_size)

def Reduced_ResNet152_m(nclasses, nf=20, bias=True, memory_bank_size = 500):
    """
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias, memory_bank_size)

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
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """ 
    #rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    #tensors_gather[rank] = tensor
    output = torch.cat(tensors_gather, dim=0)
    
    return output

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".
    
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

# K * #Guass as input
@torch.no_grad()
def AGD_torch_no_grad_gpu(M, maxIter=20, eps=0.05):
    M = M.t() # [#Guass, K]
    p = M.shape[0] # #Guass
    n = M.shape[1] # K 
    
    X = torch.zeros((p,n), dtype=torch.float64).cuda()

    r = torch.ones((p,), dtype=torch.float64).to(M.device) / p # .to(L.device) / K
    c = torch.ones((n,), dtype=torch.float64).to(M.device) / n # .to(L.device) / B 

    max_el = torch.max(abs(M)) #np.linalg.norm(M, ord=np.inf)
    gamma = eps/(3*math.log(n)) 

    A = torch.zeros((maxIter, 1), dtype=torch.float64).to(M.device) #init array of A_k
    L = torch.zeros((maxIter, 1), dtype=torch.float64).to(M.device) #init array of L_k

    # set initial values for APDAGD
    L[0,0] = 1; #set L_0

    #set starting point for APDAGD
    y = torch.zeros((n+p, maxIter), dtype=torch.float64).cuda() #init array of points y_k for which usually the convergence rate is proved (eta)
    z = torch.zeros((n+p, maxIter), dtype=torch.float64).cuda() #init array of points z_k. this is the Mirror Descent sequence. (zeta)    
    j = 0
    # main cycle of APDAGD
    for k in range(0,(maxIter-1)):
                         
        L_t = (2**(j-1))*L[k,0] #current trial for L            
        a_t = (1  + torch.sqrt(1 + 4*L_t*A[k,0]))/(2*L_t) #trial for calculate a_k as solution of quadratic equation explicitly
        A_t = A[k,0] + a_t; #trial of A_k
        tau = a_t / A_t; #trial of \tau_{k}     
        x_t = tau*z[:,k] + (1 - tau)*y[:,k]; #trial for x_k
        
        lamb = x_t[:n,]
        mu = x_t[n:n+p,]    
        
        # 1) [K,1] * [1, #Gauss] --> [K, #Gauss].T -->[#Gauss, K]; 2) [K, 1] * [#Guass, 1].T --> [K, #Guass]--.T--> [#Guass, K]
        M_new = -M - torch.matmul(lamb.reshape(-1,1).cuda(), torch.ones((1,p), dtype=torch.float64).cuda()).T - torch.matmul(torch.ones((n,1), dtype=torch.float64).cuda(), mu.reshape(-1,1).T.cuda()).T

        X_lamb = torch.exp(M_new/gamma)
        sum_X = torch.sum(X_lamb)
        X_lamb = X_lamb/sum_X
        grad_psi_x_t = torch.zeros((n+p,), dtype=torch.float64).cuda() 
        grad_psi_x_t[:p,] = r - torch.sum(X_lamb, axis=1)
        grad_psi_x_t[p:p+n,] = c - torch.sum(X_lamb, axis=0).T

        #update model trial
        z_t = z[:,k] - a_t*grad_psi_x_t #trial of z_k 
        y_t = tau*z_t + (1 - tau)*y[:,k] #trial of y_k

        #calculate function \psi(\lambda,\mu) value and gradient at the trial point of y_{k}
        lamb = y_t[:n,]
        mu = y_t[n:n+p,]           
        M_new = -M - torch.matmul(lamb.reshape(-1,1).cuda(), torch.ones((1,p), dtype=torch.float64).cuda()).T - torch.matmul(torch.ones((n,1), dtype=torch.float64).cuda(), mu.reshape(-1,1).T.cuda()).T
        Z = torch.exp(M_new/gamma)
        sum_Z = torch.sum(Z)

        X = tau*X_lamb + (1-tau)*X #set primal variable 
            # break
             
        L[k+1,0] = L_t
        j += 1
    
    X = X.t()

    indexs = torch.argmax(X, dim=1)
    G = F.gumbel_softmax(X, tau=0.5, hard=True)

    return G.to(torch.float32), indexs # change into G as well 










































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
