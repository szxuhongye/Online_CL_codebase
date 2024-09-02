import torch
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from copy import deepcopy
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda
from utils.setup_elements import transforms_match, transforms_aug
from torch.utils import data
from torch import nn
from torch.nn import functional as F
from utils.buffer.buffer import Buffer, Second_Buffer


class CLSER(ContinualLearner):
    def __init__(self, model, opt, params):
        super(CLSER, self).__init__(model, opt, params)
        self.params = params
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.device = next(self.model.parameters()).device
        # self.device = get_device()
        self.plastic_model = deepcopy(self.model).to(self.device)
        self.stable_model = deepcopy(self.model).to(self.device)
        # set regularization weight
        self.reg_weight = params.reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = params.plastic_model_update_freq
        self.plastic_model_alpha = params.plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = params.stable_model_update_freq
        self.stable_model_alpha = params.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0


        self.num_classes = 50 if params.data == 'core50' else 100
        self.task = 0
        if params.second_buffer:
            self.second_buffer = params.second_buffer
            self.buffer2 = Second_Buffer(model, params)

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        # losses_batch = AverageMeter()
        # losses_mem = AverageMeter()
        # acc_batch = AverageMeter()
        # acc_mem = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                inputs = batch_x_aug
                labels = batch_y

                real_batch_size = batch_x.shape[0]

                self.opt.zero_grad()
                loss = 0

                if self.params.second_buffer:
                    mem_x1, mem_y1 = self.buffer.retrieve(x=batch_x, y=batch_y)
                    mem_x2, mem_y2 = self.buffer2.retrieve(x=batch_x, y=batch_y)
                    mem_x = torch.cat((mem_x1, mem_x2), dim=0)
                    mem_y = torch.cat((mem_y1, mem_y2), dim=0)
                    # exit(0)
                else:
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                
                if mem_x.size(0) > 0:
                    # sample from buffer
                    mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                for idx in range(mem_x.size(0))])
                    mem_x = maybe_cuda(mem_x, self.cuda)
                    mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                    mem_y = maybe_cuda(mem_y, self.cuda)
                    mem_logits = self.model.forward(mem_x_aug)


                    stable_model_logits = self.stable_model(mem_x_aug)
                    plastic_model_logits = self.plastic_model(mem_x_aug)

                    stable_model_prob = F.softmax(stable_model_logits, 1)
                    plastic_model_prob = F.softmax(plastic_model_logits, 1)

                    label_mask = F.one_hot(mem_y, num_classes=stable_model_logits.shape[-1]) > 0
                    sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
                    sel_idx = sel_idx.unsqueeze(1)

                    ema_logits = torch.where(
                        sel_idx,
                        stable_model_logits,
                        plastic_model_logits,
                    )

                    l_cons = torch.mean(self.consistency_loss(self.model(mem_x_aug), ema_logits.detach()))
                    l_reg = self.params.reg_weight * l_cons
                    loss += l_reg


                    inputs = torch.cat((batch_x_aug, mem_x_aug))
                    labels = torch.cat((batch_y, mem_y))
                
                outputs = self.model(inputs)
                ce_loss = self.criterion(outputs, labels)
                loss += ce_loss

                # Log values
                # if hasattr(self, 'writer'):
                #     self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
                #     self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)
                loss.backward()
                self.opt.step()

                self.buffer.update(batch_x, batch_y)
                if self.params.second_buffer:
                    self.buffer2.update(batch_x, batch_y)

                # Update the ema model
                self.global_step += 1
                if torch.rand(1) < self.plastic_model_update_freq:
                    self.update_plastic_model_variables()

                if torch.rand(1) < self.stable_model_update_freq:
                    self.update_stable_model_variables()

        self.after_train()

    # def update_plastic_model_variables(self):
    #     alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
    #     for ema_param, param in zip(self.plastic_model.parameters(), self.model.parameters()):
    #         ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    # def update_stable_model_variables(self):
    #     alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
    #     for ema_param, param in zip(self.stable_model.parameters(), self.model.parameters()):
    #         ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    
    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
