import os
import copy
import torch
import numpy as np
from torch.utils import data
from utils.buffer.buffer import Buffer, Second_Buffer
from utils.buffer.buffer_utils import random_retrieve, get_grad_vector
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss
import torch.nn.functional as F
import torch.nn as nn
from argparse import ArgumentParser
import math
# epsilon = 1E-20
# torch.autograd.set_detect_anomaly(True)

class SuperPCR(ContinualLearner):
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
        super(SuperPCR, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.params = params
        if params.second_buffer:
            self.second_buffer = params.second_buffer
            self.buffer2 = Second_Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        # context_vectors, _ = create_context_vectors(self.model, params.num_tasks, True, True)
        # self.context = context_vectors
        self.task = 0
        # for i in range(17):
        # print(len(self.context[0]))
        # for name, params in self.model.named_parameters():
        #     if name.endswith('weight'):     # only weight, not bias
        #         if 'conv' in name and 'bn' not in name and 'shortcut' not in name:               
        #             print(name, params.size())

        # exit(0)
        ############################################################
        # self.device = next(self.model.parameters()).device
        # # self.device = get_device()
        # self.temp = copy.deepcopy(self.model).to(self.device)

        # # self.temp = copy.deepcopy(self.net).to(self.device)
        # self.temp_opt = torch.optim.SGD(self.temp.parameters(), lr=0.01)

        # lr = params.lr
        # weight_decay = 0.0001
        # self.delta = 0.00001
        # self.tau = 0.00001

        # self.logsoft = nn.LogSoftmax(dim=1)
        # self.checkpoint = None
        # self.fish = {}
        # for name, param in self.model.named_parameters():
        #     self.fish[name] = torch.zeros_like(param).to(self.device)

        # self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        ############################################################
    

    # def unlearn(self, inputs, labels):

    #     self.temp.load_state_dict(self.model.state_dict())
    #     self.temp.train()
    #     outputs, _ = self.temp.pcrForward(inputs)
    #     # logits, feas= self.model.pcrForward(batch_x_combine)
    #     loss = - F.cross_entropy(outputs, labels)
    #     self.temp_opt.zero_grad()
    #     loss.backward()
    #     self.temp_opt.step()

    #     for (model_name, model_param), (temp_name, temp_param) in zip(self.model.named_parameters(), self.temp.named_parameters()):
    #         weight_update = temp_param - model_param
    #         model_param_norm = model_param.norm()
    #         weight_update_norm = weight_update.norm() + epsilon
    #         norm_update = model_param_norm / weight_update_norm * weight_update
    #         identity = torch.ones_like(self.fish[model_name])
    #         with torch.no_grad():
    #             # model_param.add_(self.delta * torch.mul(1.0/(identity + 0.001*self.fish[model_name]), norm_update + 0.001*torch.randn_like(norm_update)))
    #             model_param.data = model_param.data + self.delta * torch.mul(1.0/(identity + 0.001*self.fish[model_name]), norm_update + 0.001*torch.randn_like(norm_update))


    def add_agent_args(parser: ArgumentParser) -> None:
        """
        Adds the arguments used by agnet.

        Args:
            parser: the parser instance

        Returns:
            None
        """
        # parser.add_argument('--joint', type=int, choices=[0, 1], default=0,
        #                 help='Train model on Joint (single task)?')
        parser.add_argument('--label_perc', type=float, default=1,
                        help='Percentage in (0-1] of labeled examples per task.')





    def add_data(self, tensor_list, new_data1, new_data2, new_data3, new_data4, current_index1, current_index2=None):
        """
        Adds new_data1 and new_data2 to the first and second row of tensor_list, respectively.
        Each new data piece is slightly increased by a very small value to avoid confusion with zero,
        and the index is updated.

        Args:
        - tensor_list (torch.Tensor): The current tensor list.
        - new_data1 (torch.Tensor): New data to be added to the first row, a one-dimensional tensor.
        - new_data2 (torch.Tensor): New data to be added to the second row, a one-dimensional tensor.
        - current_index (int): The current index where the filling is up to.

        Returns:
        - (torch.Tensor, int): The updated tensor_list and current_index.
        """
        # Calculate the length of the new data, assuming new_data1 and new_data2 are the same length
        new_data_length1 = new_data1.shape[0]
        new_data_length2 = new_data2.shape[0]
        new_data_length3 = new_data3.shape[0]
        new_data_length4 = new_data4.shape[0]
        
        # Add a very small value to the new data to distinguish it from zero
        new_data1_adjusted = new_data1 + 1e-6 
        new_data2_adjusted = new_data2 + 1e-6
        new_data3_adjusted = new_data3 + 1e-6
        new_data4_adjusted = new_data4 + 1e-6
        
        # Add the new data to the correct position in the first and second row, respectively
        # if current_index2 is not None:
        #     tensor_list[0, current_index1: current_index1 + new_data_length1] = new_data1_adjusted
        #     tensor_list[1, current_index2: current_index2 + new_data_length2] = new_data2_adjusted
        #     tensor_list[2, current_index1: current_index1 + new_data_length1] = new_data3_adjusted
        #     current_index1 += new_data_length1
        #     current_index2 += new_data_length2
        #     return tensor_list, current_index1, current_index2
        # else:
        tensor_list[0, current_index1: current_index1 + new_data_length1] = new_data1_adjusted
        tensor_list[1, current_index1: current_index1 + new_data_length1] = new_data2_adjusted
        tensor_list[2, current_index1: current_index1 + new_data_length1] = new_data3_adjusted
        tensor_list[3, current_index1: current_index1 + new_data_length1] = new_data4_adjusted
        current_index1 += new_data_length1

        return tensor_list, current_index1
        
        

    def train_learner(self, x_train, y_train, run=0, batch=0):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()
        if self.params.hardness_analysis:
            measure = Hardness(self.params)
            if self.params.second_buffer:
                batch_hardness = torch.zeros((8, self.epoch))
            else:
                batch_hardness = torch.zeros((4, self.epoch))


        # for ep in range(self.epoch):
        #     if self.params.hardness_analysis:
        #         if self.params.second_buffer:
        #             hardness_list1 = torch.zeros((4, 5000))
        #             hardness_list2 = torch.zeros((4, 5000))
        #             current_index1 = 0
        #             current_index2 = 0
        #         else:
        #             hardness_list = torch.zeros((4, 5000))
        #             current_index = 0
        #     for i, batch_data in enumerate(train_loader):
        #         # batch update
        #         batch_x, batch_y = batch_data
        #         batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
        #                                    for idx in range(batch_x.size(0))])
        #         batch_x = maybe_cuda(batch_x, self.cuda)
        #         batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)

        #         # print("batch_x.shape: ", batch_x.shape) # torch.Size([10, 3, 32, 32])

        #         batch_y = maybe_cuda(batch_y, self.cuda)
        #         batch_x_combine = torch.cat((batch_x, batch_x_aug))
        #         batch_y_combine = torch.cat((batch_y, batch_y))

        #         # print("batch_x_combine.shape: ", batch_x_combine.shape) # torch.Size([20, 3, 32, 32])

        #         for j in range(self.mem_iters):
        #             # logits, feas= self.model.pcrForward(batch_x_combine, use_PSP=True, contexts=self.context, task_id=self.task)
        #             logits, feas= self.model.pcrForward(batch_x_combine, task_id=self.task)


        #             if self.params.second_buffer:
        #                 if self.params.buffer_analyze:
        #                     mem_x1, mem_y1, buffer_x1, buffer_y1 = self.buffer.retrieve()
        #                     mem_x2, mem_y2, buffer_x2, buffer_y2 = self.buffer2.retrieve()
        #                 else:
        #                     mem_x1, mem_y1 = self.buffer.retrieve()
        #                     mem_x2, mem_y2 = self.buffer2.retrieve()
        #                 mem_x = torch.cat((mem_x1, mem_x2), dim=0)
        #                 mem_y = torch.cat((mem_y1, mem_y2), dim=0)

        #             else:
        #                 if self.params.buffer_analyze:
        #                     mem_x, mem_y, buffer_x, buffer_y = self.buffer.retrieve()
        #                 else:
        #                     mem_x, mem_y = self.buffer.retrieve()
                    

        #             if self.params.hardness_analysis and mem_x.size(0) > 0:
        #                 if self.params.second_buffer:
        #                     scores1, _, flip_sum1, entropy_pre1, entropy_post1  = measure.calculate(self.model, mem_x1, mem_y1)
        #                     hardness_list1, current_index1 = self.add_data(hardness_list1, scores1, flip_sum1, entropy_pre1, entropy_post1, current_index1)
        #                     scores2, _, flip_sum2, entropy_pre2, entropy_post2  = measure.calculate(self.model, mem_x2, mem_y2)
        #                     hardness_list2, current_index2 = self.add_data(hardness_list2, scores2, flip_sum2, entropy_pre2, entropy_post2, current_index2)
        #                 else:
        #                     scores, _, flip_sum, entropy_pre, entropy_post = measure.calculate(self.model, mem_x, mem_y)
        #                     hardness_list, current_index = self.add_data(hardness_list, scores, flip_sum, entropy_pre, entropy_post, current_index)
        #                     # print(flip_sum)
        #                     # print(entropy_pre)
        #                     # exit(0)

        #             # for param in self.model.parameters():
        #             #     print(param.grad)
        #             # print("logits.shape: ", logits.shape) # torch.Size([20, 100])
        #             # print("feas.shape: ", feas.shape) # torch.Size([20, 160])
        #             # exit(0)
        #             # if self.triplet:
        #             #     if self.top5:
        #             #         novel_loss = 0*cluster_triplet_loss_v2(outputs=logits, targets=batch_y_combine)
        #             #     else:
        #             #         novel_loss = 0*cluster_triplet_loss(outputs=logits, targets=batch_y_combine)
        #             # else:
        #             novel_loss = 0*self.criterion(logits, batch_y_combine)
        #             # novel_loss = self.criterion(logits, batch_y_combine)

        #             # if batch == 1:
        #             #     import ipdb; ipdb.set_trace()
        #             # print("novel_loss: ", novel_loss)
        #             # print("novel_loss.shape: ", novel_loss.shape)
        #             # exit(0)
        #             # if self.params.refresh:
        #             #     self.unlearn(inputs=batch_x_combine, labels=batch_y_combine)


        #             self.opt.zero_grad()
        #             # for param in self.buffer.model.parameters():
        #             #     print(param.grad)
        #             # exit(0)
        #             # if not self.params.second_buffer:
        #             #     mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    
                    
        #             if mem_x.size(0) > 0:

        #                 # mem_x, mem_y = Rotation(mem_x, mem_y)
        #                 mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
        #                                          for idx in range(mem_x.size(0))])
        #                 mem_x = maybe_cuda(mem_x, self.cuda)
        #                 mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
        #                 mem_y = maybe_cuda(mem_y, self.cuda)
        #                 mem_x_combine = torch.cat([mem_x, mem_x_aug])
        #                 mem_y_combine = torch.cat([mem_y, mem_y])
        #                 # print("mem_x_combine.shape: ", mem_x_combine.shape) # torch.Size([20, 3, 32, 32])


        #                 # mem_logits, mem_fea= self.model.pcrForward(mem_x_combine, use_PSP=True, contexts=self.context, task_id=self.task)
        #                 mem_logits, mem_fea= self.model.pcrForward(mem_x_combine, task_id=self.task)
                        
        #                 combined_feas = torch.cat([mem_fea, feas])
        #                 combined_labels = torch.cat((mem_y_combine, batch_y_combine))
        #                 combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels] # proxy
        #                 # combined_feas_aug = torch.mean(self.model.prototypes, dim=1)[combined_labels] # proxy

        #                 combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
        #                 combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

        #                 combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
        #                     combined_feas_aug)
        #                 combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
        #                 # print("combined_feas_normalized.shape: ", combined_feas_normalized.shape) # torch.Size([40, 160])
        #                 # print("combined_feas_aug_normalized.shape: ", combined_feas_aug_normalized.shape) # torch.Size([40, 160])
        #                 cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
        #                                         combined_feas_aug_normalized.unsqueeze(1)],
        #                                         dim=1)
        #                 # print("cos_features.shape: ", cos_features.shape) # torch.Size([40, 2, 160])
        #                 # exit(0)
        #                 PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
        #                 novel_loss += PSC(features=cos_features, labels=combined_labels)
                        # print("novel_loss: ", novel_loss)
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

                for j in range(self.mem_iters):
                    logits = self.model.forward(batch_x_combine, self.task)

                    if self.params.second_buffer:
                        if self.params.buffer_analyze:
                            mem_x1, mem_y1, buffer_x1, buffer_y1 = self.buffer.retrieve()
                            mem_x2, mem_y2, buffer_x2, buffer_y2 = self.buffer2.retrieve()
                        else:
                            mem_x1, mem_y1 = self.buffer.retrieve()
                            mem_x2, mem_y2 = self.buffer2.retrieve()
                        mem_x = torch.cat((mem_x1, mem_x2), dim=0)
                        mem_y = torch.cat((mem_y1, mem_y2), dim=0)

                    else:
                        if self.params.buffer_analyze:
                            mem_x, mem_y, buffer_x, buffer_y = self.buffer.retrieve()
                        else:
                            mem_x, mem_y = self.buffer.retrieve()


                    loss = self.criterion(logits, batch_y_combine)
                    loss_re = torch.tensor(0.)

                    if self.params.second_buffer:
                        mem_x1, mem_y1 = self.buffer.retrieve(x=batch_x, y=batch_y)
                        mem_x2, mem_y2 = self.buffer2.retrieve(x=batch_x, y=batch_y)
                        mem_x = torch.cat((mem_x1, mem_x2), dim=0)
                        mem_y = torch.cat((mem_y1, mem_y2), dim=0)
                        # exit(0)
                    else:
                        mem_x, mem_y = self.buffer.retrieve()
                    if mem_x.size(0) > 0:
                        # sample from buffer
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)

                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])
                        mem_logits = self.model.forward(mem_x_combine, self.task)
                        loss_re = self.criterion(mem_logits, mem_y_combine)
                    
                    loss = loss + loss_re


                    loss.backward()
                    # novel_loss.backward()
                    self.opt.step()
                # update mem
                self.buffer.update(batch_x, batch_y)
                if self.params.second_buffer:
                    self.buffer2.update(batch_x, batch_y)





            if self.params.hardness_analysis:
                if self.params.second_buffer:
                    # hardness_list1 = hardness_list1[:, :current_index1]
                    # hardness_list2 = hardness_list2[:, :current_index2]
                    non_zero_avg_row1 = torch.masked_select(hardness_list1[0], hardness_list1[0] != 0).mean()
                    non_zero_avg_row2 = torch.masked_select(hardness_list1[1], hardness_list1[1] != 0).mean()
                    non_zero_avg_row3 = torch.masked_select(hardness_list1[2], hardness_list1[2] != 0).mean()
                    non_zero_avg_row4 = torch.masked_select(hardness_list1[3], hardness_list1[3] != 0).mean()
                    non_zero_avg_row5 = torch.masked_select(hardness_list2[0], hardness_list2[0] != 0).mean()
                    non_zero_avg_row6 = torch.masked_select(hardness_list2[1], hardness_list2[1] != 0).mean()
                    non_zero_avg_row7 = torch.masked_select(hardness_list2[2], hardness_list2[2] != 0).mean()
                    non_zero_avg_row8 = torch.masked_select(hardness_list2[3], hardness_list2[3] != 0).mean()
                    batch_hardness[0, ep] = non_zero_avg_row1
                    batch_hardness[1, ep] = non_zero_avg_row2
                    batch_hardness[2, ep] = non_zero_avg_row3
                    batch_hardness[3, ep] = non_zero_avg_row4
                    batch_hardness[4, ep] = non_zero_avg_row5
                    batch_hardness[5, ep] = non_zero_avg_row6
                    batch_hardness[6, ep] = non_zero_avg_row7
                    batch_hardness[7, ep] = non_zero_avg_row8
                else:
                    non_zero_avg_row1 = torch.masked_select(hardness_list[0], hardness_list[0] != 0).mean()
                    non_zero_avg_row2 = torch.masked_select(hardness_list[1], hardness_list[1] != 0).mean()
                    non_zero_avg_row3 = torch.masked_select(hardness_list[2], hardness_list[2] != 0).mean()
                    non_zero_avg_row4 = torch.masked_select(hardness_list[3], hardness_list[3] != 0).mean()
                    batch_hardness[0, ep] = non_zero_avg_row1
                    batch_hardness[1, ep] = non_zero_avg_row2
                    batch_hardness[2, ep] = non_zero_avg_row3
                    batch_hardness[3, ep] = non_zero_avg_row4

            if self.params.save_cp:
                if (ep + 1) % 50 == 0:
                    filename = f"{self.params.exp}_checkpoint_run{run}_batch{batch}_epoch{ep}.pth"
                    filepath = os.path.join(self.params.cp_path, filename)
                    grads = {name: parameter.grad.clone() for name, parameter in self.model.named_parameters() if parameter.grad is not None}

                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'grads': grads,
                        'optimizer_state_dict': self.opt.state_dict(),
                        'buffer': self.buffer.state_dict(),
                        'context': self.context if hasattr(self, 'context') else None,
                        'params': self.params, 
                        'mem_size': self.mem_size,
                        'eps_mem_batch': self.eps_mem_batch,
                        'mem_iters': self.mem_iters,
                        'buffer2': self.buffer2.state_dict() if hasattr(self, 'buffer2') else None,
                    }

                    torch.save(checkpoint, filepath)
            if self.params.buffer_analyze:
                if (ep + 1) % 10 == 0:
                    #save mem_x, mem_y, buffer_x, buffer_y into a numpy file
                    if self.params.second_buffer: # concat mem_x1 with mem_x2, buffer_x1 with buffer_x2, etc.
                        mem_x1, mem_x2 = mem_x1.cpu().numpy(), mem_x2.cpu().numpy()
                        mem_y1, mem_y2 = mem_y1.cpu().numpy(), mem_y2.cpu().numpy()
                        buffer_x1, buffer_x2 = buffer_x1.cpu().numpy(), buffer_x2.cpu().numpy()
                        buffer_y1, buffer_y2 = buffer_y1.cpu().numpy(), buffer_y2.cpu().numpy()
                        # use np.concatenate to concatenate the arrays
                        mem_x = np.concatenate((mem_x1, mem_x2), axis=0)
                        mem_y = np.concatenate((mem_y1, mem_y2), axis=0)
                        buffer_x = np.concatenate((buffer_x1, buffer_x2), axis=0)
                        buffer_y = np.concatenate((buffer_y1, buffer_y2), axis=0)
                        batch_x = batch_x.cpu().numpy()
                        batch_y = batch_y.cpu().numpy()

                        filename = f"{self.params.exp}_run{run}_batch{batch}_epoch{ep}_mem_2buffer.npz"
                        filepath = os.path.join(self.params.buffer_analyze_path, filename)
                        # np.save(filepath, [mem_x, mem_y, buffer_x, buffer_y])
                        np.savez(filepath, batch_x=batch_x, batch_y=batch_y, mem_x=mem_x, mem_y=mem_y, buffer_x=buffer_x, buffer_y=buffer_y)
                    else:
                        mem_x = mem_x.cpu().numpy()
                        mem_y = mem_y.cpu().numpy()
                        buffer_x = buffer_x.cpu().numpy()
                        buffer_y = buffer_y.cpu().numpy()
                        batch_x = batch_x.cpu().numpy()
                        batch_y = batch_y.cpu().numpy()
                        filename = f"{self.params.exp}_run{run}_batch{batch}_epoch{ep}_mem_1buffer.npz"
                        filepath = os.path.join(self.params.buffer_analyze_path, filename)
                        np.savez(filepath, batch_x=batch_x, batch_y=batch_y, mem_x=mem_x, mem_y=mem_y, buffer_x=buffer_x, buffer_y=buffer_y)
        
        if self.params.hardness_analysis:
            batch_hardness_numpy = batch_hardness.cpu().numpy()
            analysis_filename = f"{self.params.exp}_run{run}_batch{batch}_hardness.npy"
            analysis_filepath = os.path.join(self.params.analysis_path, analysis_filename)
            np.save(analysis_filepath, batch_hardness_numpy)
        
        self.after_train()

        
    # def end_task(self, x_train, y_train):
    #     train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
    #                                 drop_last=True)

    #     self.temp.load_state_dict(self.model.state_dict())
    #     fish = {}
    #     for name, param in self.temp.named_parameters():
    #         fish[name] = torch.zeros_like(param).to(self.device)

    #     for j, data in enumerate(train_loader):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(self.device), labels.to(self.device)
    #         for ex, lab in zip(inputs, labels):
    #             self.temp_opt.zero_grad()
    #             output, _ = self.temp.pcrForward(ex.unsqueeze(0))
    #             loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
    #                                 reduction='none')
    #             exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
    #             loss = torch.mean(loss)
    #             loss.backward()


    #             for name, param in self.temp.named_parameters():
    #                 # print(type(param.grad))\
    #                 # print('name:', name)
    #                 # print('param.grad: ', param.grad)
    #                 # print(type(exp_cond_prob))
    #                 if param.grad is None:
    #                     continue
    #                 else:
    #                     fish[name] +=  exp_cond_prob * param.grad ** 2

    #     for name, param in self.temp.named_parameters():
    #         fish[name] /= (len(train_loader) * self.params.batch)
    
    #     for key in self.fish:
    #         self.fish[key] *= self.tau
    #         self.fish[key] += fish[key].to(self.device)


    #     # self.checkpoint = self.model.get_params().data.clone()
    #     self.temp_opt.zero_grad()


def context_multiplication(model, contexts, task_index):
    """
    """
    new_model = copy.deepcopy(model)
    
    layer_index = 0
    for name, params in new_model.named_parameters():
        if name.endswith('weight'):  # only weight, not bias 
            if 'conv' in name and 'bn' not in name and 'shortcut' not in name:
                with torch.no_grad():                 
                    import ipdb; ipdb.set_trace()
                    context_matrix = torch.from_numpy(np.reshape(contexts[task_index][layer_index],
                                                                 newshape=params.size()).astype(np.float32)).cuda()
                    new_params = params * context_matrix
                    params.copy_(new_params)  
                    layer_index += 1
        else:
            continue
            
    return new_model


def random_binary_array(size):
    """
    Create an array of 'size' length consisting only of numbers -1 and 1 (approximately 50% each).

    :param size: length of the created array
    :return: binary numpy array with values -1 or 1
    """
    vec = np.random.uniform(-1, 1, size)
    vec[vec < 0] = 1
    vec[vec >= 0] = -1
    return vec

def create_context_vectors(model, num_tasks, element_wise, use_PSP=False):
    """
    Create random binary context vectors for all model layers.
    Return together with layer dimension side, which is a list of 0 (first dimension taken for context size)
    and 1 (second dimension taken for context size).

    :param model: torch model instance
    :param num_tasks: number of tasks
    :param element_wise: boolean - if True, the number of context values in self attention part is the same as number of parameters
    :param use_PSP: boolean - if True, PSP method is used, meaning we need set of contexts for each task (including the first)
    :return: context_vectors (shape=(num_tasks-1, num of model layers)), layer_dimension (length=num of model layers)
    """
    context_vectors = []
    layer_dimension = []
    n = num_tasks if use_PSP else num_tasks - 1
    for t in range(n):    # our contexts only needed between tasks, i.e. len(contexts)=num_task-1
        task_contexts = []
        count = 0
        for name, params in model.named_parameters():
            if name.endswith('weight'):     # only weight, not bias
                if 'conv' in name and 'bn' not in name and 'shortcut' not in name:
                    
                    vector_size = math.prod(params.size())
                    count += 1
                    # print(name, params.size(), count)
                    if t == 0:
                        layer_dimension.append(2)   # to apply element-wise product
                else:
                    continue
                # elif 'self_attn' not in name:     # FC layer
                #     vector_size = params.size()[1]
                #     if t == 0:
                #         layer_dimension.append(1)
                # else:   # not FC layer (e.g., Wq, Wk, Wv in multi-head attention)
                #     if element_wise:
                #         vector_size = params.size()[0] * params.size()[1]
                #         if t == 0:
                #             layer_dimension.append(2)
                #     else:
                #         vector_size = params.size()[0]
                #         if t == 0:
                #             layer_dimension.append(0)

                binary_context_vector = random_binary_array(vector_size)
                task_contexts.append(binary_context_vector)
                # print(len(task_contexts))
        context_vectors.append(task_contexts)
        # exit(0)
    return context_vectors, layer_dimension






class Hardness(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch
    
    def temperature_scaled_softmax(self, logits, temperature=1.0):
        """Compute the temperature-scaled softmax of a given tensor of logits.
        """
        assert temperature > 0, "Temperature must be positive."
        scaled_logits = logits / temperature
        softmax_probs = F.softmax(scaled_logits, dim=-1)
        return softmax_probs

    def calculate(self, model, sub_x, sub_y, **kwargs):
        # sub_x, sub_y = random_retrieve(buffer, self.subsample)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(model.parameters, grad_dims)
        
        
        model_temp = self.get_future_step_parameters(model, grad_vector, grad_dims)
        with torch.no_grad():
            if self.params.agent == 'PCR':
                logits_pre, _ = model.pcrForward(sub_x)
                if isinstance(logits_pre, tuple):
                    logits_pre = logits_pre[1]
                logits_post, _ = model_temp.pcrForward(sub_x)
                flip_abs = torch.abs(logits_pre - logits_post)
                flip_sum = torch.sum(flip_abs, dim=1)
                if isinstance(logits_post, tuple):
                    logits_post = logits_post[1]
            else:
                logits_pre = model.forward(sub_x)
                if isinstance(logits_pre, tuple):
                    logits_pre = logits_pre[1] 
                logits_post = model_temp.forward(sub_x)
                if isinstance(logits_post, tuple):
                    logits_post = logits_post[1]
            # exit(0)
            probabilities_pre = self.temperature_scaled_softmax(logits_pre, temperature=1.2)
            
            entropy_pre = -torch.sum(probabilities_pre * torch.log(probabilities_pre), dim=1)

            
            probabilities_post = self.temperature_scaled_softmax(logits_post, temperature=1.2)
            entropy_post = -torch.sum(probabilities_post * torch.log(probabilities_post), dim=1)


            flip_abs = torch.abs(logits_pre - logits_post)
            flip_sum = torch.sum(flip_abs, dim=1)
            pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
            post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
            scores = post_loss - pre_loss
            # big_ind = scores.sort(descending=True)[1][:self.num_retrieve]
            big_ind = scores.sort(descending=True)[1]
            
        return scores, big_ind, flip_sum, entropy_pre, entropy_post

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """

        new_model = copy.deepcopy(model)
        # except RuntimeError as e:
        #     new_model = model.custom_deepcopy()
        # self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        # print(new_model)
        # for name, param in new_model.named_parameters():
        #     print(f"Parameter name: {name}")
        # exit(0)
        self.overwrite_grad(new_model.named_parameters(), grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.params.learning_rate * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for name, param in pp:
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            # print(beg, en) # 0, 540 for PCR
            # print(new_grad[beg: en].shape) # torch.Size([640000]) for ca # torch.Size([540]) for PCR
            # print(param.data.size()) # torch.Size([1000, 640]) for ca # torch.Size([20, 3, 3, 3]) for PCR
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1
