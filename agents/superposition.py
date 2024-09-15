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

class Superposition(ContinualLearner):
    def __init__(self, model, opt, params):
        super(Superposition, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.params = params
        if params.second_buffer:
            self.second_buffer = params.second_buffer
            self.buffer2 = Second_Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.task = 0

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


        for ep in range(self.epoch):
            if self.params.hardness_analysis:
                if self.params.second_buffer:
                    hardness_list1 = torch.zeros((4, 5000))
                    hardness_list2 = torch.zeros((4, 5000))
                    current_index1 = 0
                    current_index2 = 0
                else:
                    hardness_list = torch.zeros((4, 5000))
                    current_index = 0
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
                    # logits, feas= self.model.pcrForward(batch_x_combine)
                    logits, _, _ = self.model.forward(batch_x_combine, self.task)


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
                            if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
                                mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                            else:
                                mem_x, mem_y = self.buffer.retrieve()
                    

                    # if self.params.hardness_analysis and mem_x.size(0) > 0:
                    #     if self.params.second_buffer:
                    #         scores1, _, flip_sum1, entropy_pre1, entropy_post1  = measure.calculate(self.model, mem_x1, mem_y1)
                    #         hardness_list1, current_index1 = self.add_data(hardness_list1, scores1, flip_sum1, entropy_pre1, entropy_post1, current_index1)
                    #         scores2, _, flip_sum2, entropy_pre2, entropy_post2  = measure.calculate(self.model, mem_x2, mem_y2)
                    #         hardness_list2, current_index2 = self.add_data(hardness_list2, scores2, flip_sum2, entropy_pre2, entropy_post2, current_index2)
                    #     else:
                    #         scores, _, flip_sum, entropy_pre, entropy_post = measure.calculate(self.model, mem_x, mem_y)
                    #         hardness_list, current_index = self.add_data(hardness_list, scores, flip_sum, entropy_pre, entropy_post, current_index)
                            # print(flip_sum)
                            # print(entropy_pre)
                            # exit(0)
                    novel_loss = self.criterion(logits, batch_y_combine)


                    self.opt.zero_grad()
                    # for param in self.buffer.model.parameters():
                    #     print(param.grad)
                    # exit(0)
                    # if not self.params.second_buffer:
                    #     mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    
                    
                    # if mem_x.size(0) > 0:

                    #     # mem_x, mem_y = Rotation(mem_x, mem_y)
                    #     mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                    #                              for idx in range(mem_x.size(0))])
                    #     mem_x = maybe_cuda(mem_x, self.cuda)
                    #     mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                    #     mem_y = maybe_cuda(mem_y, self.cuda)
                    #     mem_x_combine = torch.cat([mem_x, mem_x_aug])
                    #     mem_y_combine = torch.cat([mem_y, mem_y])
                    #     # print("mem_x_combine.shape: ", mem_x_combine.shape) # torch.Size([20, 3, 32, 32])


                    #     mem_logits, mem_fea= self.model.pcrForward(mem_x_combine)
                        
                    #     # if self.triplet:
                    #     #     # triplet loss
                    #     #     combined_feas = torch.cat([mem_fea, feas])
                    #     #     combined_logits = torch.cat([mem_logits, logits])
                    #     #     combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                    #     #     if self.top5:
                    #     #         novel_loss += cluster_triplet_loss_v2(outputs=combined_logits, targets=combined_labels)
                    #     #     else:
                    #     #         novel_loss += cluster_triplet_loss(outputs=combined_logits, targets=combined_labels)
                    #     # else:
                    #     combined_feas = torch.cat([mem_fea, feas])
                    #     combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                    #     combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels] # proxy
                    #     # combined_feas_aug = torch.mean(self.model.prototypes, dim=1)[combined_labels] # proxy

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


                    novel_loss.backward()
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

        

