
import torch
import random
from utils.utils import maybe_cuda
import torch.nn.functional as F
import torch.nn as nn
from utils.buffer.buffer_utils import random_retrieve, get_grad_vector
import copy
import numpy as np
import time
from continuum.continuum import continuum
from continuum.data_utils import setup_test_loader
from utils.name_match import agents
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
from experiment.metrics import compute_performance, single_run_avg_end_fgt
from experiment.tune_hyperparam import tune_hyper
from types import SimpleNamespace
from utils.io import load_yaml, save_dataframe_csv, check_ram_usage
import pandas as pd
import os
import pickle
# from utils.buffer.buffer import Buffer, Second_Buffer


def load_weight(agent, checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.opt.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.params = checkpoint['params']
    agent.mem_size = checkpoint['mem_size']
    agent.eps_mem_batch = checkpoint['eps_mem_batch']
    agent.mem_iters = checkpoint['mem_iters']
    agent.triplet = checkpoint['triplet']
    agent.top5 = checkpoint['top5']
    # agent.buffer = Buffer(model, params)
    agent.buffer.load_state_dict(checkpoint['buffer'])
    
    if checkpoint['buffer2'] is not None:
        agent.buffer2 = Second_Buffer(agent, params)
        agent.buffer2.load_state_dict(checkpoint['buffer2'])
    saved_grads = checkpoint['grads']
    for name, param in agent.model.named_parameters():
        if name in saved_grads:
            param.grad = saved_grads[name]

    return agent

class Loss_change(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch

    def calculate(self, model, sub_x, sub_y, **kwargs):
        # sub_x, sub_y = random_retrieve(buffer, self.subsample)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(model.parameters, grad_dims)
        # print(grad_vector)
        # print(sub_x.shape)
        # print
        
        
        model_temp = self.get_future_step_parameters(model, grad_vector, grad_dims)
        with torch.no_grad():
            if self.params.agent == 'PCR':
                logits_pre, _ = model.pcrForward(sub_x)
                if isinstance(logits_pre, tuple):
                    logits_pre = logits_pre[1] 
                logits_post, _ = model_temp.pcrForward(sub_x)
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

            flip_abs = torch.abs(logits_pre - logits_post)
            flip_sum = torch.sum(flip_abs, dim=1)
            pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
            post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
            scores = post_loss - pre_loss
            # big_ind = scores.sort(descending=True)[1][:self.num_retrieve]
            big_ind = scores.sort(descending=True)[1]
            
        return scores, big_ind, flip_sum

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


class Uncertainty(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch
        if params.data == 'cifar10':
            self.num_classes = 10
        elif params.data == 'core50':
            self.num_classes = 50
        else:
            self.num_classes = 100
        self.ucr_max = params.ucr_max

    def calculate(self, model, sub_x, sub_y, **kwargs):
        # sub_x, sub_y = random_retrieve(buffer, self.subsample)
        if sub_x.size(0) > 0:
            # exit(0)
            with torch.no_grad():
                model.eval()

                logits, emb =  model.pcrForward(sub_x)
                embDim = emb.shape[-1]

                preds = F.softmax(logits, dim=1)
                exoponential = torch.exp(logits)
                sums = exoponential.sum(dim=1)

                softmax_derivative = ((exoponential * sums.unsqueeze(1)) - exoponential **2) * (1/sums.unsqueeze(1))**2
                J = torch.einsum('bi,bj->bij', (softmax_derivative, emb))
                JJ = torch.zeros((emb.shape[0], self.num_classes, self.num_classes*emb.shape[1])).cuda()
                entropies = torch.zeros(sub_x.size(0)).cuda()
                
                for i in range(self.num_classes):
                    JJ[:,i,i*emb.shape[1]:(i+1)*emb.shape[1]] = J[:,i,:]


                
                # print('grads.shape', grads.shape)
                self.sigma = torch.eye(embDim * self.num_classes).cuda()
                for i in range(sub_x.size(0)):
                    grads = self.get_grad_vector(logits, emb, i, num_classes=self.num_classes)
                    sigma2 = self.sigma @ torch.outer(grads, grads) @ self.sigma
                    sigma3 = sigma2/ (1 + grads @ self.sigma @ grads)
                    self.sigma = (self.sigma - sigma3)

                
            
                # Prrediction dist (Normal distribution over weights -> Normal distribution over outputs)
                    sigma = torch.einsum('bcn,nn,bnl->bcl', (JJ, self.sigma, JJ.transpose(1,2)))
            
                # Laplace bridge  (Normal distribution over outputs -> Dirichlet distribution)
                    _ , K = preds.size(0), preds.size(-1)
                    sigma_diagonal = torch.diagonal(sigma, dim1=1, dim2=2)
                    sum_exp = torch.sum(torch.exp(-1*preds), dim=1).view(-1,1)
                    alphas = 1/sigma_diagonal * (1 - 2/K + torch.exp(preds)/K**2 * sum_exp)
                    alphas_i = alphas[i]
                # Entropy of Dirichlet distribution
                    dist= torch.distributions.Dirichlet(alphas_i)
                    entropies[i] = dist.entropy()
                    
                # if self.ucr_max:
                #     ind = entropies.sort(descending=True)[1][:self.num_retrieve]
                # else:
                #     ind = entropies.sort(descending=False)[1][:self.num_retrieve]

                return entropies
        else:
            return sub_x, sub_y
    


    def get_grad_vector(self, logits, emb, i, num_classes, **kwargs):
        with torch.no_grad():
            emb = emb[i]  #
            batchProbs = F.softmax(logits, dim=1)[i]  
            # print('batchProbs shape: ', batchProbs.shape)
            # exit(0)
            maxInds = torch.argmax(batchProbs)  
            embDim = emb.shape[0]  # Get the size of the embedding dimension
            embedding = torch.zeros(embDim * num_classes, device=emb.device)  # Ensure the tensor is on the same device
            order = torch.argsort(batchProbs, descending=True)  
            probs = batchProbs[order]
            for c in range(num_classes):
                if c == maxInds:
                    embedding[embDim * c : embDim * (c+1)] = copy.deepcopy(emb) * (1 - probs[c])
                else:
                    embedding[embDim * c : embDim * (c+1)] = copy.deepcopy(emb) * (-1 * probs[c])

            probs = probs / torch.sum(probs)

            sqrt_probs = torch.sqrt(probs)
            sqrt_probs_repeated = sqrt_probs.repeat_interleave(embDim)
            embedding = embedding * sqrt_probs_repeated
            
        return embedding.float()  



parameters = {
    'num_runs': 1,
    'seed': 0,
    'val_size': 0.1,
    'num_val': 3,
    'num_runs_val': 3,
    'error_analysis': False,  
    'verbose': True,  
    'store': False,  
    'save_path': './',
    'imagenet_path': './imagenet1k',
    'agent': 'PCR',
    'update': 'random',
    'retrieve': 'random',
    'second_buffer': False,  
    'update2': 'random',
    'retrieve2': 'random',
    # 'ratio': 0.2,  
    'optimizer': 'SGD',
    'learning_rate': 0.1,
    'epoch': 1,
    'batch': 10,
    'test_batch': 128,
    'weight_decay': 0,
    'num_tasks': 20,
    'fix_order': False,  
    'plot_sample': False, 
    'data': "cifar100",
    'cl_type': "nc",
    'ns_factor': (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6),
    'ns_type': 'noise',
    'ns_task': (1, 1, 2, 2, 2, 2),
    'online': True, 
    'use_momentum': True,  
    'mem_size': 10000,
    'eps_mem_batch': 10,
    'sub_eps_mem_batch': 2,
    'lambda_': 100,
    'alpha': 0.9,
    'fisher_update_after': 50,
    'subsample': 50,
    'gss_mem_strength': 10,
    'gss_batch_size': 10,
    'k': 5,
    'aser_type': "asvm",
    'n_smp_cls': 2.0,
    'stm_capacity': 1000,
    'classifier_chill': 0.01,
    'log_alpha': -300,
    'minlr': 0.0005,
    'clip': 10.,
    'mem_epoch': 70,
    'labels_trick': False,  
    'separated_softmax': False,  
    'kd_trick': False, 
    'kd_trick_star': False,  
    'review_trick': False,  
    'ncm_trick': False,  
    'mem_iters': 1,
    'min_delta': 0.,
    'patience': 0,
    'cumulative_delta': False,  
    'temp': 0.07,
    'buffer_tracker': False,  
    'warmup': 4,
    'head': 'mlp',
    'exp': 'PCR',
    'Triplet': False, 
    'top5': False,  
    'mem_bank_size': 1,
    'num_subcentroids': 4,
    'PSC': False,  
    'onlyPSC': False,  
    'gamma': 0.999,
    'buffer_lip_lambda': 0.5,
    'budget_lip_lambda': 0.5,
    'headless_init_act': "relu",
    'grad_iter_step': -2,
    'lr': 0.0001,
    'optim_wd': 0.,
    'optim_mom': 0.,
    'optim_nesterov': 0,
    'ignore_other_metrics': 0,
    'debug_mode': 0,
    'reg_weight': 0.1,
    'stable_model_update_freq': 0.70,
    'stable_model_alpha': 0.999,
    'plastic_model_alpha': 0.999,
    'ucr_max': True,  
    'save_cp': False,  
    # 'cp_name': 'checkpoint.pth',
    'cp_path': './checkpoint',
}



from types import SimpleNamespace

# 假设params是之前定义的字典
args = SimpleNamespace(**parameters)
params = args
params.cuda = torch.cuda.is_available()
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if params.cuda:
    torch.cuda.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
params.trick = {'labels_trick': args.labels_trick, 'separated_softmax': args.separated_softmax,
                  'kd_trick': args.kd_trick, 'kd_trick_star': args.kd_trick_star, 'review_trick': args.review_trick,
                  'ncm_trick': args.ncm_trick}

start = time.time()
print('Setting up data stream')
data_continuum = continuum(params.data, params.cl_type, params)
data_end = time.time()
print('data setup time: {}'.format(data_end - start))
accuracy_list = []
params_keep = []
data_continuum.new_run()
model = setup_architecture(params)
model = maybe_cuda(model, params.cuda)
opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
agent = agents[params.agent](model, opt, params)

checkpoint_path = f'./checkpoint/PCR_cifar100_batch10_epoch50_random_cp_logepoch_checkpoint_run0_batch19_epoch49.pth'
agent_ld = load_weight(agent, checkpoint_path)

# prepare val data loader
test_loaders = setup_test_loader(data_continuum.test_data(), params)

def temperature_scaled_softmax(logits, temperature=1.0):
    """
    对给定的logits应用温度缩放softmax。
    """
    assert temperature > 0, "Temperature must be positive."
    scaled_logits = logits / temperature
    log_softmax_probs = F.log_softmax(scaled_logits, dim=-1)
    return log_softmax_probs


# 需要评估的不同温度值
temperatures = np.arange(0.1, 10.1, 0.1)

# 对应每个温度的平均损失值
temperature_losses = []
criterion = nn.NLLLoss()

for temp in temperatures:
    total_loss = 0.0
    total_samples = 0
    for task, test_loader in enumerate(test_loaders):
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = maybe_cuda(batch_x, params.cuda)
            batch_y = maybe_cuda(batch_y, params.cuda)
            with torch.no_grad():
                agent_ld.model.eval()

                logits, _ =  agent_ld.model.pcrForward(batch_x)
                # logits, _ = agent.model.(batch_x, batch_y)
                log_probs = temperature_scaled_softmax(logits, temp)
                loss = criterion(log_probs, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)
    
    avg_loss = total_loss / total_samples
    temperature_losses.append(avg_loss) # 将平均损失添加到列表中

# 打印不同温度下的平均损失
for temp, loss in zip(temperatures, temperature_losses):
    print(f"Temperature: {temp}, Average Loss: {loss}")

data_to_save = np.array([temperatures, temperature_losses])
np.save('temperature_losses.npy', data_to_save)
print("Data saved to temperature_losses.npy")


    # for task, test_loader in enumerate(test_loaders):
    #     for i, (batch_x, batch_y) in enumerate(test_loader):
    #         batch_x = maybe_cuda(batch_x, params.cuda)
    #         batch_y = maybe_cuda(batch_y, params.cuda)
            
# data_iterator = iter(test_loaders[1])
# batch_x, batch_y = next(data_iterator)
# batch_x, batch_y = maybe_cuda(batch_x, params.cuda), maybe_cuda(batch_y, params.cuda)
# print(torch.cuda.is_available())
# exit(0)





# UCR_measure = Uncertainty(params)

# num_epochs = 50
# num_batches = 20  
# batch_size = 128

# all_UCR = np.zeros((batch_size, num_epochs * num_batches))
# for batch in range(num_batches):
#     for epoch in range(num_epochs):
#         checkpoint_path = f'./checkpoint/PCR_cifar100_batch10_epoch50_random_cp_logepoch_checkpoint_run0_batch{batch}_epoch{epoch}.pth'
        
#         agent_ld = load_weight(agent, checkpoint_path)
        
#         # 假设x是从checkpoint或其他地方获得的输入
#         x = None  # 需要根据实际情况定义x
#         uncertainty = UCR_measure.calculate(agent_ld.model, batch_x, batch_y)
#         # 使用f(x)计算输出，并存储
#         all_UCR[:, num_epochs * batch + epoch]  = uncertainty.cpu()
#         print(f'epoch {epoch}, batch {batch} finished')
#     # use numpy to save the every batch data
#     # np.save(f'UCR_{batch}.npy', all_UCR)


# # use numpy to save the data
# np.save('all_UCR.npy', all_UCR)