from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from torch.utils import data
from utils.buffer.buffer import Buffer
from utils.utils import maybe_cuda, AverageMeter
import torch
import torch.nn.functional as F
from argparse import ArgumentParser


class DERPP(ContinualLearner):
    def __init__(self, model, opt, params):
        super(DERPP, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.params = params
    

    def add_agent_args(self, parser: ArgumentParser) -> None:
        """
        Adds the arguments used by agnet.

        Args:
            parser: the parser instance

        Returns:
            None
        """
        # parser.add_argument('--joint', type=int, choices=[0, 1], default=0,
        #                 help='Train model on Joint (single task)?')
        parser.add_argument('--derpp_alpha', type=float, default=0.5,
                            help='Penalty weight.')
        parser.add_argument('--derpp_beta', type=float, default=0.5,
                            help='Penalty weight.')

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
        # acc_batch = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):
                    logits = self.model(batch_x)
                    loss = F.cross_entropy(logits, batch_y)
                    self.opt.zero_grad()
                    loss.backward()
                    tot_loss = loss.item()

                    if self.task_seen > 0:

                        mem_x, _, mem_logits = self.buffer.retrieve(return_logits=True)
                        mem_outputs = self.model(mem_x)
                        loss_mse = self.params.derpp_alpha * F.mse_loss(mem_outputs, mem_logits)
                        loss_mse.backward()
                        tot_loss += loss_mse.item()

                        mem_x, mem_y = self.buffer.retrieve()

                        mem_outputs = self.model(mem_x)
                        loss_ce = self.params.derpp_beta * F.cross_entropy(mem_outputs, mem_y)
                        loss_ce.backward()
                        tot_loss += loss_ce.item()

                    self.opt.step()
                # update mem
                self.buffer.update(batch_x, batch_y, logits)
                # if i % 100 == 1 and self.verbose:
                #     print(
                #         '==>>> it: {}, avg. loss: {:.6f}, '
                #         'running train acc: {:.3f}'
                #             .format(i, losses_batch.avg(), acc_batch.avg())
                #     )
        self.after_train()