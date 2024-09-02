import copy
import torch
import random
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from typing import List, Optional, Tuple, Union
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss
from functools import wraps
from torchvision import transforms as T
# from lightly.loss import SwaVLoss





# Adapted from https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py


# helper functions


def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


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




class ProxyContrastiveReplay_m(ContinualLearner):
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
        super(ProxyContrastiveReplay_m, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
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
        
        


        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((32, 32)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.use_momentum = params.use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        # self.online_predictor = self.model.online_predictor


        
    
    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.model)
        set_requires_grad(target_encoder, False)
        return target_encoder
    
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.model)


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


                for j in range(self.mem_iters):
                    logits, feas= self.model.forward2(batch_x_combine)
                    centroids = self.model.attention2.class_centroids[batch_y_combine]
                    Cri = ClusterContrastiveLoss()
                    novel_loss = 0*self.criterion(feas, centroids)
                    # print("novel_loss.shape: ", novel_loss.shape) # torch.Size([])
                    self.opt.zero_grad()
                    

                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        # mem_x, mem_y = Rotation(mem_x, mem_y)
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu()) for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])
                        # print("mem_x_combine.shape: ", mem_x_combine.shape) # torch.Size([20, 3, 32, 32])


                        mem_logits, mem_feas= self.model.forward2(mem_x_combine)
                        novel_loss += F.cross_entropy(logits, batch_y_combine)
                        # print("mem_fea.shape: ", mem_fea.shape) # torch.Size([20, 160]
                        # print("feas.shape: ", feas.shape) # torch.Size([20, 160]

                        combined_feas = torch.cat([mem_feas, feas])
                        combined_logits = torch.cat([mem_logits, logits])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        # novel_loss += infoNCE_loss(combined_logits, combined_labels)
                        # combined_feas_aug = self.model.attention.class_centroids[combined_labels] # proxy
                        # centroids_vectors = self.model.attention2.class_centroids[combined_labels] # proxy
                        combined_feas_aug = self.model.attention2.class_centroids[combined_labels] # proxy

                        # combined_feas_aug = self.model.attention.class_centroids[combined_labels] # proxy
                        # combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels] # proxy

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
                        PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
                        novel_loss += PSC(features=cos_features, labels=combined_labels)

                        
                        
                        # novel_loss += Cri(combined_feas, centroids_vectors)

                    novel_loss.backward()
                    self.opt.step()
                # update mem
                self.buffer.update(batch_x, batch_y)

        self.after_train()
    




class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits, labels):
        # logits: [batch_size, num_clusters] - the raw, unnormalized scores for each cluster
        # labels: [batch_size] - the index of the positive class for each input

        # Gather the logits for the positive targets. We need to add a dimension for gather to work correctly
        pos_indices = labels.view(-1, 1)  # Add the extra dimension
        pos_logits = logits.gather(1, pos_indices)  # Extract the logits of the positive samples

        # Calculate the log of the sum of the exponentials of the logits
        log_exp_sum = torch.logsumexp(logits / self.temperature, dim=1, keepdim=True)

        # Now, we compute the InfoNCE loss which is -log(exp(pos_logits/temperature) / sum(exp(logits/temperature)))
        loss = log_exp_sum - (pos_logits / self.temperature)

        return loss.mean()  # Taking the mean over the batch


def cluster_triplet_loss(outputs, targets, margin=1.0):
    """
    计算聚类的三元组损失。

    参数:
        outputs (tensor): 模型的输出，形状为 (batch_size, num_clusters)。这应该是每个输入与每个聚类中心的相似度得分。
        targets (tensor): 真实的类别标签，形状为 (batch_size)。这应是每个输入的正确聚类索引。
        margin (float): 三元组损失中使用的边缘值。这是正样本和负样本之间的期望最小差异。

    返回:
        tensor: 批次中所有输入的总损失。
    """
    # 获取每个样本的正确聚类得分（正样本）
    positive_scores = outputs[range(outputs.shape[0]), targets]

    # 为了找到负样本，我们取除了正确聚类以外的最大得分
    # 首先，我们将正样本得分设置为一个非常大的负值，这样它们就不会被选为最大值或次最大值
    mask = torch.ones_like(outputs)  # 创建一个和outputs形状相同的掩码
    mask[range(outputs.shape[0]), targets] = 0  # 将正样本位置的掩码设为0

    # 应用掩码，将正样本得分替换为极大的负值
    masked_outputs = outputs * mask + (1 - mask) * (-1e10)

    # 使用topk找到每行的最大和次大值
    top2_scores = torch.topk(masked_outputs, 2, dim=1).values  # 返回每行的最大和次大值

    # 负样本得分是次大值，位于返回的top2_scores的最后一列
    negative_scores = top2_scores[:, 1]

    # 计算损失：确保正确的聚类比错误的聚类更接近，至少有一个“margin”的差距
    loss = F.relu(negative_scores - positive_scores + margin)

    return loss.mean()


def infoNCE_loss(outputs, targets, temperature=0.1):
    """
    计算InfoNCE损失。

    参数:
        outputs (tensor): 模型的输出，形状为 (batch_size, num_clusters)。这应该是每个输入与每个聚类中心的相似度得分。
        targets (tensor): 真实的类别标签，形状为 (batch_size)。这应是每个输入的正确聚类索引。
        temperature (float): 缩放因子，用于调整相似度分数的温度。

    返回:
        tensor: 批次中所有输入的平均损失。
    """
    # 将输出转换为对数概率形式
    log_probs = F.log_softmax(outputs / temperature, dim=1)

    # 构造一个适当形状的targets，以便与log_probs一起使用
    targets = targets.view(-1, 1)  # 给targets增加一个维度以便得到适当的广播

    # 使用gather方法从log_probs张量中提取目标索引处的对数概率
    positive_log_probs = log_probs.gather(dim=1, index=targets).squeeze()  # 结果是一个一维张量

    # InfoNCE损失是正样本对数概率的负数
    loss = -positive_log_probs.mean()  # 计算批次中所有样本的平均损失

    return loss


class ClusterContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ClusterContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, correct_cluster_features):
        """
        计算模型的损失。这里我们假设`features`是模型的输出，其形状为[bsz, feature_dim]，
        `correct_cluster_features` 是每个样本的正确聚类中心，形状为 [bsz, feature_dim]。

        参数:
            features (tensor): 形状为[bsz, feature_dim]的隐藏向量。
            correct_cluster_features (tensor): 每个样本的正确聚类中心。
            
        返回:
            一个损失标量。
        """
        if features.dim() != 2 or correct_cluster_features.dim() != 2:
            raise ValueError('`features` and `correct_cluster_features` need to be 2D tensors')

        # 计算特征和正确聚类中心之间的相似度矩阵
        similarity_matrix = torch.matmul(features, correct_cluster_features.T) / self.temperature

        # 使用softmax函数得到概率分布。由于我们比较的是每个特征与它的正确聚类中心，
        # 我们期望每行（每个样本）的最大值（最相似）是在对角线上。
        probabilities = torch.softmax(similarity_matrix, dim=1)

        # 创建一个标识矩阵，指示哪些是正确的聚类中心。
        batch_size = features.size(0)
        labels = torch.arange(batch_size).to(features.device)
        mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)

        # 计算对数概率，并应用掩码选择正确聚类的概率
        log_prob = torch.log(probabilities + 1e-10)  # 避免log(0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 计算损失作为负对数概率的均值
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()

        return loss











