import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss


class ProxyContrastiveReplay(ContinualLearner):
    """
        Proxy-based Contrastive Replay,
        Implements the strategy defined in
        "PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning"
        https://arxiv.org/abs/2304.04408

        This strategy has been designed and tested in the
        Online Setting (OnlineCLScenario). However, it
        can also be used in non-online scenarios
        """
    def __init__(self, model, opt, params):
        super(ProxyContrastiveReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
    

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

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

                # print("batch_x.shape: ", batch_x.shape) # torch.Size([10, 3, 32, 32])

                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x_combine = torch.cat((batch_x, batch_x_aug))
                batch_y_combine = torch.cat((batch_y, batch_y))

                # print("batch_x_combine.shape: ", batch_x_combine.shape) # torch.Size([20, 3, 32, 32])

                for j in range(self.mem_iters):
                    logits, feas= self.model.pcrForward(batch_x_combine)
                    # print("logits.shape: ", logits.shape) # torch.Size([20, 100])
                    # print("feas.shape: ", feas.shape) # torch.Size([20, 160])
                    # exit(0)   
                    novel_loss = 0*self.criterion(logits, batch_y_combine)
                    # print("novel_loss.shape: ", novel_loss.shape)
                    # exit(0)
                    self.opt.zero_grad()


                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        # mem_x, mem_y = Rotation(mem_x, mem_y)
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])
                        # print("mem_x_combine.shape: ", mem_x_combine.shape) # torch.Size([20, 3, 32, 32])


                        mem_logits, mem_fea= self.model.pcrForward(mem_x_combine)
                        # print("mem_fea.shape: ", mem_fea.shape) # torch.Size([20, 160]
                        # print("feas.shape: ", feas.shape) # torch.Size([20, 160]

                        combined_feas = torch.cat([mem_fea, feas])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels] # proxy

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


                    novel_loss.backward()
                    self.opt.step()
                # update mem
                self.buffer.update(batch_x, batch_y)

        self.after_train()