from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from torch.utils import data
from utils.buffer.buffer import Buffer
from utils.utils import maybe_cuda, AverageMeter
import torch


class GEM(ContinualLearner):
    def __init__(self, model, opt, params):
        super(GEM, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)

        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)

        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                logits = self.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                _, pred_label = torch.max(logits, 1)
                correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)

                # update tracker
                acc_batch.update(correct_cnt, batch_y.size(0))
                losses_batch.update(loss, batch_y.size(0))

                # backward
                self.opt.zero_grad()
                loss.backward()

                if self.task_seen > 0:
                    # sample from memory of previous tasks
                    mem_x, mem_y = self.buffer.buffer_img, self.buffer.buffer_label
                    # mem_x, mem_y = self.buffer.retrieve()

                    # gradient computed using current batch
                    grad_current = [p.grad.clone() for p in self.model.parameters() if p.requires_grad]

                    # check and project if necessary
                    for tt in self.old_labels:
                        self.opt.zero_grad()
                        mem_y_filtered = mem_y[mem_y == tt]
                        if len(mem_y_filtered) == 0:
                            continue

                        mem_logits = self.forward(mem_x[mem_y == tt])
                        mem_loss = self.criterion(mem_logits, mem_y[mem_y == tt])
                        mem_loss.backward()
                        grad_mem = [p.grad.clone() for p in self.model.parameters() if p.requires_grad]

                        # project if violating
                        dot_product = sum(torch.sum(gc * gm) for gc, gm in zip(grad_current, grad_mem))
                        if dot_product < 0:
                            proj = sum(torch.sum(gm * gm) for gm in grad_mem)
                            grad_current = [gc - dot_product / proj * gm for gc, gm in zip(grad_current, grad_mem)]

                            # replace old grads with projected ones
                            for g, p in zip(grad_current, self.model.parameters()):
                                if p.requires_grad:
                                    p.grad.data.copy_(g)

                self.opt.step()
                self.buffer.update(batch_x, batch_y)
                # if i % 100 == 1 and self.verbose:
                #     print(
                #         '==>>> it: {}, avg. loss: {:.6f}, '
                #         'running train acc: {:.3f}'
                #             .format(i, losses_batch.avg(), acc_batch.avg())
                #     )

        self.after_train()
