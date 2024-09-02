import torch
from torch.utils import data
from utils.buffer.buffer import Buffer, Second_Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.lipschitz import LipOptimizer
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda
from utils.setup_elements import transforms_match, transforms_aug


class ERACE_Lipschitz(LipOptimizer):
    def __init__(self, model, opt, params):
        super(ERACE_Lipschitz, self).__init__(model, opt, params)
        self.params = params
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = 50 if params.data == 'core50' else 100
        self.task = 0
        if params.second_buffer:
            self.second_buffer = params.second_buffer
            self.buffer2 = Second_Buffer(model, params)
        self.model.set_return_prerelu(True)

    
    def buffer_retrieve(self, batch_x, batch_y):
        if self.params.second_buffer:
            mem_x1, mem_y1 = self.buffer.retrieve(x=batch_x, y=batch_y)
            mem_x2, mem_y2 = self.buffer2.retrieve(x=batch_x, y=batch_y)
            mem_x = torch.cat((mem_x1, mem_x2), dim=0)
            mem_y = torch.cat((mem_y1, mem_y2), dim=0)
        else:
            mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
        
        
        return mem_x, mem_y

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        self.init_net(train_loader)
        # set up model
        self.model = self.model.train()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                present = batch_y.unique()
                self.seen_so_far = maybe_cuda(self.seen_so_far, self.cuda)
                self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
                for j in range(self.mem_iters):
                    logits = self.model.forward(batch_x_aug)
                    mask = torch.zeros_like(logits)
                    mask[:, present] = 1

                    self.opt.zero_grad()
                    if self.seen_so_far.max() < (self.num_classes - 1):
                        mask[:, self.seen_so_far.max():] = 1 

                    if self.task > 0:
                        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

                    loss = self.criterion(logits, batch_y)
                    loss_re = torch.tensor(0.)
                    loss = loss + loss_re

                    mem_x, mem_y = self.buffer_retrieve(batch_x, batch_y)
                    
                    if mem_x.size(0) > 0:
                        # sample from buffer
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_logits = self.model.forward(mem_x_aug)
                        loss_re = self.criterion(mem_logits, mem_y)
                    
                    if self.params.buffer_lip_lambda>0:
                        mem_x, mem_y = self.buffer_retrieve(batch_x, batch_y)
                        if mem_x.size(0) > 0:
                            # sample from buffer
                            mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                    for idx in range(mem_x.size(0))])
                            mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                            _, buf_output_features = self.model(mem_x_aug, returnt='full')

                            lip_inputs = [mem_x_aug] + buf_output_features[:-1]
                            loss_lip_buffer = self.buffer_lip_loss(lip_inputs)
                            loss += self.params.buffer_lip_lambda * loss_lip_buffer
                    
                    if self.params.budget_lip_lambda>0:
                        mem_x, mem_y = self.buffer_retrieve(batch_x, batch_y)
                        if mem_x.size(0) > 0:
                            # sample from buffer
                            mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                    for idx in range(mem_x.size(0))])
                            mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                            _, buf_output_features = self.model(mem_x_aug, returnt='full')

                            lip_inputs = [mem_x_aug] + buf_output_features[:-1]
                            loss_lip_buffer = self.budget_lip_loss(lip_inputs)
                            loss += self.params.budget_lip_lambda * loss_lip_buffer
                    
                    
                    # backward
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                self.buffer.update(batch_x, batch_y)
                if self.params.second_buffer:
                    self.buffer2.update(batch_x, batch_y)
        self.after_train()