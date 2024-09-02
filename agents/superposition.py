import os
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
import numpy as np
import torch
from torch.utils import data
from utils.buffer.buffer import Buffer, Second_Buffer
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda
from utils.setup_elements import transforms_match, transforms_aug


class Superposition(ContinualLearner):
    def __init__(self, model, opt, params):
        super(Superposition, self).__init__(model, opt, params)
        self.params = params
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        # self.seen_so_far = torch.tensor([]).long()
        self.num_classes = 50 if params.data == 'core50' else 100
        self.task = 0
        if params.second_buffer:
            self.second_buffer = params.second_buffer
            self.buffer2 = Second_Buffer(model, params)

    def train_learner(self, x_train, y_train, run=0, batch=0):
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
                        # backward
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                self.buffer.update(batch_x, batch_y)
                if self.params.second_buffer:
                    self.buffer2.update(batch_x, batch_y)


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
        self.after_train()
        
    

    # def evaluate(self, test_loaders):
    #     self.model.eval()
    #     acc_array = np.zeros(len(test_loaders))
    #     with torch.no_grad():

    #         for task, test_loader in enumerate(test_loaders):
    #             acc = AverageMeter()
    #             for i, (batch_x, batch_y) in enumerate(test_loader):
    #                 batch_x = maybe_cuda(batch_x, self.cuda)
    #                 batch_y = maybe_cuda(batch_y, self.cuda)
                    
    #                 logits_list = []

    #                 for time in range(self.params.num_tasks):
    #                     logits = self.model.forward(batch_x, time)  # 传递 time 变量
    #                     logits_list.append(logits)

    #                 logits_stack = torch.stack(logits_list, dim=1)

    #                 max_logits, _ = torch.max(logits_stack, dim=1)
    #                 _, pred_label = torch.max(max_logits, 1)

    #                 correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)

    #                 if self.params.error_analysis:
    #                     correct_lb += [task] * len(batch_y)
    #                     for i in pred_label:
    #                         predict_lb.append(self.class_task_map[i.item()])
    #                     if task < self.task_seen-1:
    #                         # old test
    #                         total = (pred_label != batch_y).sum().item()
    #                         wrong = pred_label[pred_label != batch_y]
    #                         error += total
    #                         on_tmp = sum([(wrong == i).sum().item() for i in self.new_labels_zombie])
    #                         oo += total - on_tmp
    #                         on += on_tmp
    #                         old_class_score.update(logits[:, list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item(), batch_y.size(0))
    #                     elif task == self.task_seen -1:
    #                         # new test
    #                         total = (pred_label != batch_y).sum().item()
    #                         error += total
    #                         wrong = pred_label[pred_label != batch_y]
    #                         no_tmp = sum([(wrong == i).sum().item() for i in list(set(self.old_labels) - set(self.new_labels_zombie))])
    #                         no += no_tmp
    #                         nn += total - no_tmp
    #                         new_class_score.update(logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
    #                     else:
    #                         pass
    #                 acc.update(correct_cnt, batch_y.size(0))
    #             acc_array[task] = acc.avg()
    #     print(acc_array)
    #     if self.params.error_analysis:
    #         self.error_list.append((no, nn, oo, on))
    #         self.new_class_score.append(new_class_score.avg())
    #         self.old_class_score.append(old_class_score.avg())
    #         print("no ratio: {}\non ratio: {}".format(no/(no+nn+0.1), on/(oo+on+0.1)))
    #         print(self.error_list)
    #         print(self.new_class_score)
    #         print(self.old_class_score)
    #         self.fc_norm_new.append(self.model.linear.weight[self.new_labels_zombie].mean().item())
    #         self.fc_norm_old.append(self.model.linear.weight[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
    #         self.bias_norm_new.append(self.model.linear.bias[self.new_labels_zombie].mean().item())
    #         self.bias_norm_old.append(self.model.linear.bias[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
    #         print(self.fc_norm_old)
    #         print(self.fc_norm_new)
    #         print(self.bias_norm_old)
    #         print(self.bias_norm_new)
    #         with open('confusion', 'wb') as fp:
    #             pickle.dump([correct_lb, predict_lb], fp)
    #     return acc_array