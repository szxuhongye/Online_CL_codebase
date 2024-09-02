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
from lightly.loss import SwaVLoss





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




class ProxyContrastiveReplay_swav(ContinualLearner):
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
        super(ProxyContrastiveReplay_swav, self).__init__(model, opt, params)
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
                # print(batch_y)
                # exit(0)
                batch_x_aug1 = torch.stack([self.augment1(batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                batch_x_aug2 = torch.stack([self.augment2(batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                
                # batch_x_aug1 = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                #                            for idx in range(batch_x.size(0))])
                # batch_x_aug2 = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                #                            for idx in range(batch_x.size(0))])
                
             
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug1 = maybe_cuda(batch_x_aug1, self.cuda)
                batch_x_aug2 = maybe_cuda(batch_x_aug2, self.cuda)

                # print("batch_x.shape: ", batch_x.shape) # torch.Size([10, 3, 32, 32])

                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x_combine1 = torch.cat((batch_x, batch_x_aug1))
                batch_x_combine2 = torch.cat((batch_x, batch_x_aug2))
                batch_y_combine = torch.cat((batch_y, batch_y))

                # print("batch_x_combine.shape: ", batch_x_combine.shape) # torch.Size([20, 3, 32, 32])

                for j in range(self.mem_iters):
                    _ , feas1= self.model.forward(batch_x_combine1)
                    # feas2= self.model.forward(batch_x_combine2)
                    # logits1 = self.model.pcrForward(feas1)
                    # print("logits.shape: ", logits.shape) # torch.Size([20, 100])
                    # print("feas.shape: ", feas.shape) # torch.Size([20, 160])
                    novel_loss = 0*self.criterion(feas1, batch_y_combine)
                    # print("novel_loss.shape: ", novel_loss.shape) # torch.Size([])
                    self.opt.zero_grad()


                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:

                        # mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                        #                          for idx in range(mem_x.size(0))])
                        mem_x_aug1 = torch.stack([self.augment1(mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x_aug2 = torch.stack([self.augment2(mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        # mem_x_aug1 = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                        #                          for idx in range(mem_x.size(0))])
                        # mem_x_aug2 = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                        #                          for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug1 = maybe_cuda(mem_x_aug1, self.cuda)
                        mem_x_aug2 = maybe_cuda(mem_x_aug2, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine1 = torch.cat([mem_x, mem_x_aug1])
                        mem_x_combine1 = torch.cat([mem_x_combine1, batch_x_combine1])
                        mem_x_combine2 = torch.cat([mem_x, mem_x_aug2])
                        mem_x_combine2 = torch.cat([mem_x_combine2, batch_x_combine2])
                        mem_y_combine = torch.cat([mem_y, mem_y])


                        # print("mem_x_combine1.shape: ", mem_x_combine1.shape) # mem_x_combine1.shape:  torch.Size([532, 3, 32, 32])

                        logit1 , combined_feas1 = self.model.forward(mem_x_combine1)
                        logit2 , combined_feas2 = self.model.forward(mem_x_combine2)
                        # print(combined_feas1.shape) # torch.Size([532, 100])
                        # exit(0)

                        # combined_feas = torch.cat([mem_fea, feas])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        # print("combined_labels.shape: ", combined_labels.shape) # torch.Size([40])
                        # combined_feas_aug = self.model.prototypes.heads[0].weight[combined_labels] # proxy
                        # print(combined_feas_aug.shape) # proxy # torch.Size([100, 640])
                        # exit(0)


                        ################ PCR part #################
                        # combined_feas_aug = self.model.prototypes.heads[0].weight[combined_labels] # proxy
                        combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels] # proxy
                        # print(combined_feas_aug.shape) # proxy # torch.Size([1024, 100])
                        # print(combined_feas1.shape) # torch.Size([1024, 100])
                        # exit(0)
                        # combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels] # proxy
                        # print("combined_feas1.shape: ", combined_feas1.shape) # torch.Size([40, 160])
                        # print("combined_feas_aug.shape: ", combined_feas_aug.shape) # torch.Size([40, 160])


                        

                        combined_feas_norm = torch.norm(combined_feas1, p=2, dim=1).unsqueeze(1).expand_as(combined_feas1)
                        combined_feas_normalized = combined_feas1.div(combined_feas_norm + 0.000001)

                        combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                            combined_feas_aug)
                        combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
                        # print("combined_feas_normalized.shape: ", combined_feas_normalized.shape) # torch.Size([40, 160])
                        # print("combined_feas_aug_normalized.shape: ", combined_feas_aug_normalized.shape) # torch.Size([40, 160])
                        cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                                                  combined_feas_aug_normalized.unsqueeze(1)],
                                                 dim=1)
                        # print("cos_features.shape: ", cos_features.shape) # torch.Size([40, 2, 160])
                        PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
                        # cri = SwaVLoss()
                        # print(logit1.shape, logit2.shape)
                        # swavloss = cri([logit1], [logit2])
                        # print("swavloss: ", swavloss)
                        # exit(0)
                        # novel_loss += PSC(features=cos_features, labels=combined_labels) + swavloss
                        novel_loss += PSC(features=cos_features, labels=combined_labels)
                        ################ PCR part #################



                        # print(novel_loss)
                        # print("novel_loss.shape: ", novel_loss.shape) # torch.Size([]) torch.Size([40]) torch.Size([40])
                        

                    novel_loss.backward()
                    self.opt.step()
                # update mem
                self.buffer.update(batch_x, batch_y)

        self.after_train()