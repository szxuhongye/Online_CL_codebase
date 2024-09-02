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
from utils.loss import SupConLoss, SupConLoss_Sub
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
        self.psc = params.PSC
        self.onlypsc = params.onlyPSC
        self.second_buffer = params.second_buffer
        if self.second_buffer:
            self.buffer2 = Second_Buffer(model, params)
        # print(params)
        # exit(0)
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
        self.model = self.model.cuda()
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
                    logits, feas = self.model.forward(batch_x_combine, batch_y_combine)
                    novel_loss = 0*self.criterion(logits, batch_y_combine)
                # novel_loss = 0*cluster_triplet_loss(logits, batch_y_combine)
                # print("novel_loss.shape: ", novel_loss.shape) # torch.Size([])
                    self.opt.zero_grad()
                    if self.second_buffer:
                        mem_x1, mem_y1 = self.buffer.retrieve(x=batch_x, y=batch_y)
                        mem_x2, mem_y2 = self.buffer2.retrieve(x=batch_x, y=batch_y)
                        mem_x = torch.cat((mem_x1, mem_x2), dim=0)
                        mem_y = torch.cat((mem_y1, mem_y2), dim=0)
                        # exit(0)
                    else:
                        mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        self.opt.zero_grad()  
                        # mem_x, mem_y = Rotation(mem_x, mem_y)
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu()) for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])
                        # print("mem_x_combine.shape: ", mem_x_combine.shape) # torch.Size([20, 3, 32, 32])
                        mem_logits, mem_fea= self.model(mem_x_combine, mem_y_combine)
                        # combined_logits, combined_feas = self.model(combined_x, combined_labels)
                        combined_feas = torch.cat([mem_fea, feas])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        combined_logits = torch.cat((mem_logits, logits))
                        # combined_feas = torch.cat([mem_fea, feas])
                        
                        # combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels] # proxy
                        # prototype_vector = copy.deepcopy(self.model.prototypes)
                        if self.onlypsc:
                            # --------------------------------------------
                            combined_feas_aug = self.model.prototypes[combined_labels] # proxy
                        # print("combined_feas_aug.shape: ", combined_feas_aug.shape) # torch.Size([40, 4, 160])

                            contrast_feas = combined_feas_aug.reshape(-1, combined_feas_aug.shape[2])
                            contrast_labels = combined_labels.repeat_interleave(combined_feas_aug.shape[1])
                            # print("contrast_feas.shape: ", contrast_feas.shape) # torch.Size([160, 160])
                            # print("contrast_labek.shape: ", contrast_label.shape) # torch.Size([160])

                            # combined_feas_aug = torch.mean(self.model.prototypes, dim=1)[combined_labels] # proxy

                            combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                            combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

                            contrast_feas_norm = torch.norm(contrast_feas, p=2, dim=1).unsqueeze(1).expand_as(
                                contrast_feas)
                            contrast_feas_normalized = contrast_feas.div(contrast_feas_norm + 0.000001)

                            PSC = SupConLoss_Sub(temperature=0.09)
                            novel_loss += PSC(combined_feas_normalized, contrast_feas_normalized, combined_labels, contrast_labels)
                            # --------------------------------------------
                        #     combined_feas_aug = torch.mean(self.model.prototypes, dim=1)[combined_labels] # proxy
                        #         # import ipdb; ipdb.set_trace()
                        #         # print(self.model.prototypes())
                        #         # print(self.model.prototypes().shape)
                        #         # exit(0)

                        #     combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                        #     combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

                        #     combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                        #         combined_feas_aug)
                        #     combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
                        #     # print("combined_feas_normalized.shape: ", combined_feas_normalized.shape) # torch.Size([40, 160])
                        #     # print("combined_feas_aug_normalized.shape: ", combined_feas_aug_normalized.shape) # torch.Size([40, 160])
                        #     cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                        #                             combined_feas_aug_normalized.unsqueeze(1)],
                        #                             dim=1)
                        #     # print("cos_features.shape: ", cos_features.shape) # torch.Size([40, 2, 160])
                        #     # exit(0)
                        #     PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
                        #     novel_loss += PSC(features=cos_features, labels=combined_labels)
                        else:
                            if self.psc:
                                combined_feas_aug = torch.mean(self.model.prototypes, dim=1)[combined_labels] # proxy
                                # import ipdb; ipdb.set_trace()
                                # print(self.model.prototypes())
                                # print(self.model.prototypes().shape)
                                # exit(0)

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

                            novel_loss += self.criterion(combined_logits, combined_labels)
                        

                    # novel_loss.backward(retain_graph=True)
                #     import ipdb; ipdb.set_trace()
                    novel_loss.backward()
                    self.opt.step()
                    # for name, param in self.model.named_parameters():
                        
                    #     if param.grad is not None:
                    #         continue
                    #         # print("\tHas gradient")
                    #     else:
                    #         print(f"Layer: {name}")
                    #         print("\tNo gradient")
                    #         exit(0)
                    
                # # update mem
                self.buffer.update(batch_x, batch_y)
                if self.second_buffer:
                    self.buffer2.update(batch_x, batch_y)

        self.after_train()

    

    






def cluster_triplet_loss(outputs, targets, margin=1.0):
    """
    Compute the triplet loss for clustering.

    Arguments:
        outputs (tensor): The model's outputs, with shape (batch_size, num_clusters). This should represent the similarity scores of each input with each cluster center.
        targets (tensor): The true class labels, with shape (batch_size). These should be the correct clustering indices for each input.
        margin (float): The margin value used in the triplet loss. This is the expected minimum difference between the positive and negative samples.

    Returns:
        tensor: The total loss for all inputs in the batch.
    """
    # Get the correct cluster score for each sample (positive samples)
    positive_scores = outputs[range(outputs.shape[0]), targets]

    # To find the negative samples, we take the highest score other than the correct cluster
    # First, we set the scores of the positive samples to a very large negative value so they are not selected as the maximum or second maximum
    mask = torch.ones_like(outputs)  # Create a mask the same shape as outputs
    mask[range(outputs.shape[0]), targets] = 0  # Set the mask for the positive samples' locations to 0

    # Apply the mask, replacing the scores of the positive samples with a large negative value
    masked_outputs = outputs * mask + (1 - mask) * (-1e10)

    # Use topk to find the maximum and second maximum values in each row
    top2_scores = torch.topk(masked_outputs, 2, dim=1).values  # Return the maximum and second maximum values per row

    # The score for the negative samples is the second maximum, found in the last column of the returned top2_scores
    negative_scores = top2_scores[:, 1]

    # Compute the loss: ensure the correct cluster is closer than the incorrect ones by at least a margin
    loss = F.relu(negative_scores - positive_scores + margin)

    return loss.mean()













