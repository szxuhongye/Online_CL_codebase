import copy
import torch
import random
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from typing import List, Optional, Tuple, Union
from utils.buffer.buffer import Buffer, Second_Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss
from functools import wraps
from torchvision import transforms as T
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
import math



class ProxyContrastiveReplay_ca(ContinualLearner):
    """
        Proxy-based Contrastive Replay,
        Implements the strategy defined in
        "PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning"
        https://arxiv.org/abs/2304.04408

        This strategy has been designed and tested in the
        Online Setting (OnlineCLScenario). However, it
        can also be used in non-online scenarios
        """
    def __init__(self, 
                model, 
                opt, 
                params,
                augment_fn=None,
                augment_fn2=None,
                moving_average_decay = 0.99,
                use_momentum = True
            ):
        super(ProxyContrastiveReplay_ca, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.params = params
        if params.second_buffer:
            self.second_buffer = params.second_buffer
            # self.ratio = params.ratio
            self.buffer2 = Second_Buffer(model, params)
        # print(params)
        # exit(0)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters

        # cluster_params
        self.temperature = 0.1
        self.update_subcentroids = True
        self.gamma = 0.999
        self.pretrain_subcentroids = False
        self.use_subcentroids = True
        self.centroid_contrast_loss = False
        self.centroid_contrast_loss_weights = 0.005
        if self.data == 'core50': 
            self.num_classes = 50 
        else : self.num_classes = 100
        self.expand_dim = False
        ####### memory_bank added #######
        self.isOnlyCE = True
        self.memory_bank = []
        self.memory_bank_label = []

        # 500 for swin-T # ResNet for 1000 # for swin-B 300 # 400 for swin-s #1000 for mobilenet-v2
        self.BS_num_limit = 100
        print('self.BS_num_limit', self.BS_num_limit)     
        
        '''set the number of sub-centroids here, 4 for best performance'''
        self.num_subcentroids = 10 
        print('subcentroids_num:', self.num_subcentroids)
        dim_objects = {
            'bird100': 7840,
            'cifar100': 160,
            'core50': 2560,
            'food100': 7840,
            'mini_imagenet': 640,
            'places100': 7840,
        }

        # 512 for resnet18, 2048 for resnet50
        # if turn 512, then no dimension expand (2048 for expansion for resnet18)
        # 2048 for imagenet without expanding dimension
        # 1024 for swin transformer without expanding dimension
        self.embedding_dim = dim_objects[self.data]
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_subcentroids, self.embedding_dim),
                                       requires_grad=self.pretrain_subcentroids)

        # CK times self.embedding_dim
        self.feat_norm = nn.LayerNorm(self.embedding_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        self.apply(init_weights)
        trunc_normal_(self.prototypes, std=0.02)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


        """
        cifar100': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        ]),
        """

    # @torch.no_grad()    
    # def copy_params(self):
    #     for model_pair in self.model_pairs:           
    #         for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
    #             param_m.data.copy_(param.data)  # initialize
    #             param_m.requires_grad = False  # not update by gradient    

    # @torch.no_grad()        
    # def _momentum_update(self):
    #     for model_pair in self.model_pairs:           
    #         for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
    #             param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model

        self.model = self.model.train()
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                           
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)


                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x_combine = torch.cat((batch_x, batch_x_aug))
                batch_y_combine = torch.cat((batch_y, batch_y))


                # for j in range(self.mem_iters):

                if self.second_buffer:
                    mem_x1, mem_y1 = self.buffer.retrieve(x=batch_x, y=batch_y)
                    mem_x2, mem_y2 = self.buffer2.retrieve(x=batch_x, y=batch_y)
                    mem_x = torch.cat((mem_x1, mem_x2), dim=0)
                    mem_y = torch.cat((mem_y1, mem_y2), dim=0)
                    # exit(0)
                else:
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                
                
                if mem_x.size(0) <= 0:
                    inputs =  self.model(batch_x_combine)
                    logits, feas = self.forward(inputs, batch_y_combine)
                    # novel_loss = 0*self.criterion(logits, batch_y_combine)
                    # novel_loss = 0*cluster_triplet_loss(logits, batch_y_combine)
                    # print("novel_loss.shape: ", novel_loss.shape) # torch.Size([])
                    novel_loss = self.criterion(logits, batch_y_combine)
                else:
                    # mem_x, mem_y = Rotation(mem_x, mem_y)
                    mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu()) for idx in range(mem_x.size(0))])
                    mem_x = maybe_cuda(mem_x, self.cuda)
                    mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                    mem_y = maybe_cuda(mem_y, self.cuda)
                    mem_x_combine = torch.cat([mem_x, mem_x_aug])
                    mem_y_combine = torch.cat([mem_y, mem_y])
                    combined_x = torch.cat([mem_x_combine, batch_x_combine])
                    combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                    # print("combined_x.shape: ", combined_x.shape) # torch.Size([20, 3, 32, 32])
                    # print("combined_labels.shape: ", combined_labels.shape) # torch.Size([40])
                    # exit(0)
                    combined_inputs = self.model(combined_x)
                    combined_logits, combined_feas = self.forward(combined_inputs, combined_labels)

                    # combined_feas = torch.cat([mem_fea, feas])
                    
                    # combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels] # proxy
                    combined_feas_aug = torch.mean(self.model.prototypes, dim=1)[combined_labels] # proxy

                    combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                    combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

                    combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                        combined_feas_aug)
                    combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
                    # print("combined_feas_normalized.shape: ", combined_feas_normalized.shape) # torch.Size([40, 160])
                    # print("combined_feas_aug_normalized.shape: ", combined_feas_aug_normalized.shape) # torch.Size([40, 160])
                    cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                                            combined_feas_aug_normalized.unsqueeze(1)],
                                            dim=1)
                    # print("cos_features.shape: ", cos_features.shape) # torch.Size([40, 2, 160])
                    # exit(0)
                    # PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
                    # novel_loss += PSC(features=cos_features, labels=combined_labels)
                    novel_loss = self.criterion(logits, batch_y_combine)
                    novel_loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()
                # update mem
                self.buffer.update(batch_x, batch_y)
                if self.second_buffer:
                    self.buffer2.update(batch_x, batch_y)

        self.after_train()
    
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
                                       requires_grad=self.pretrain_subcentroids).to(self.device)

        # sync across gpus
        # if self.use_subcentroids is True and dist.is_available() and dist.is_initialized():
        #     centroids = self.prototypes.data.clone()
        #     dist.all_reduce(centroids.div_(dist.get_world_size()))
        #     self.prototypes = nn.Parameter(centroids, requires_grad=self.pretrain_subcentroids)

        return centroid_logits, centroid_target

    def forward(self, x, gt_label, **kwargs):

        '''for subcentroids training, this this'''
        inputs = self.pre_logits(x) # (batch_size x 512)

        batch_size = inputs.shape[0]

        if self.isOnlyCE is True:
            #Save to memory bank
            BS_num = self.dequeue_and_enqueue(inputs, gt_label, batch_size)
            
            if BS_num>=self.BS_num_limit:
                # print("BS_num:", BS_num)
                self.isOnlyCE = False
            else:
                # print(inputs.device, gt_label.device) # cuda:0 cuda:0
                # exit(0)
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
        return seg_logits, x

    def forward_train(self, x, gt_label=None):
        """Forward function for both train and test."""
        # get higher dimension if comment out, no dimension expanded

        # print("x_shape", x.shape)
        # print(x.device)
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


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)













