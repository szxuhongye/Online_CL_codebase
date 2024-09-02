import torch
from utils.utils import maybe_cuda
import torch.nn.functional as F
import torch.nn as nn
from utils.buffer.buffer_utils import random_retrieve, get_grad_vector
from models.resnet_m import Reduced_ResNet18_m, Reduced_ResNet50_m
import copy


class MIR_retrieve(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch
        self.return_logits = params.return_logits

    def retrieve(self, buffer, **kwargs):
        if self.return_logits:
            sub_x, sub_y, sub_logits = random_retrieve(buffer, self.subsample, return_logits=True)
        else:
            sub_x, sub_y = random_retrieve(buffer, self.subsample)
        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        # print(sub_x.shape)
        # print
        
        
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            # print(grad_vector)
            # print(grad_vector.shape)
            # exit(0)
            with torch.no_grad():
                if self.params.agent == 'PCR_ca':
                    logits_pre, _ = buffer.model.forward(sub_x, sub_y)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post, _ = model_temp.forward(sub_x, sub_y)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                elif self.params.agent == 'PCR':
                    logits_pre, _ = buffer.model.pcrForward(sub_x)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post, _ = model_temp.pcrForward(sub_x)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                else:
                    logits_pre = buffer.model.forward(sub_x)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post = model_temp.forward(sub_x)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                # exit(0)
                pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                scores = post_loss - pre_loss
                big_ind = scores.sort(descending=True)[1][:self.num_retrieve]
            if self.params.buffer_analyze:
                buffer_x = buffer.buffer_img
                buffer_y = buffer.buffer_label
                return sub_x[big_ind], sub_y[big_ind], buffer_x, buffer_y
            else:
                if self.return_logits:
                    return sub_x[big_ind], sub_y[big_ind], sub_logits[big_ind]
                return sub_x[big_ind], sub_y[big_ind]
        else:
            if self.params.buffer_analyze:
                buffer_x = buffer.buffer_img
                buffer_y = buffer.buffer_label
                return sub_x, sub_y, buffer_x, buffer_y
            else:
                if self.return_logits:
                    return sub_x, sub_y, sub_logits
                return sub_x, sub_y

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        # try:
        if self.params.data == 'cifar100' and self.params.agent == 'PCR_ca':
            new_model = Reduced_ResNet18_m(100, memory_bank_size = self.params.mem_bank_size, num_subcentroids=self.params.num_subcentroids, gamma=self.params.gamma)
            new_model.embedding_dim = 160
            new_model.prototypes = nn.Parameter(torch.zeros(100, self.params.num_subcentroids, 160),
                                        requires_grad=True)
            new_model.feat_norm = nn.LayerNorm(160)
            original_state_dict = model.state_dict()
            new_model.load_state_dict(original_state_dict)
            device = next(model.parameters()).device
            new_model.to(device)

        elif self.params.data == 'mini_imagenet'and self.params.agent == 'PCR_ca':
            new_model = Reduced_ResNet18_m(100, memory_bank_size = self.params.mem_bank_size, num_subcentroids=self.params.num_subcentroids, gamma=self.params.gamma)
            # model.pcrLinear = cosLinear(640, nclass)
            new_model.embedding_dim = 640
            new_model.prototypes = nn.Parameter(torch.zeros(100, self.params.num_subcentroids, 640),
                                        requires_grad=True)
            new_model.feat_norm = nn.LayerNorm(640)
            original_state_dict = model.state_dict()
            new_model.load_state_dict(original_state_dict)
            device = next(model.parameters()).device
            new_model.to(device)
        else:
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




class Inverse_MIR_retrieve(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.sub_eps_mem_batch
        self.return_logits = params.return_logits

    def retrieve(self, buffer, **kwargs):
        if self.return_logits:
            sub_x, sub_y, sub_logits = random_retrieve(buffer, self.subsample, return_logits=True)
        else:
            sub_x, sub_y = random_retrieve(buffer, self.subsample)
        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                if self.params.agent == 'PCR_ca':
                    logits_pre, _ = buffer.model.forward(sub_x, sub_y)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post, _ = model_temp.forward(sub_x, sub_y)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                elif self.params.agent == 'PCR':
                    logits_pre, _ = buffer.model.pcrForward(sub_x)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post, _ = model_temp.pcrForward(sub_x)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                else:
                    logits_pre = buffer.model.forward(sub_x)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post = model_temp.forward(sub_x)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                scores = post_loss - pre_loss
                # big_ind = scores.sort(descending=True)[1][:self.num_retrieve]
                small_ind = scores.sort(descending=False)[1][:self.num_retrieve]
            # return sub_x[big_ind], sub_y[big_ind]
            if self.params.buffer_analyze:
                buffer_x = buffer.buffer_img
                buffer_y = buffer.buffer_label
                return sub_x[small_ind], sub_y[small_ind], buffer_x, buffer_y
            else:
                if self.return_logits:
                    return sub_x[small_ind], sub_y[small_ind], sub_logits[small_ind]
                return sub_x[small_ind], sub_y[small_ind]
        else:
            if self.params.buffer_analyze:
                buffer_x = buffer.buffer_img
                buffer_y = buffer.buffer_label
                return sub_x, sub_y, buffer_x, buffer_y
            else:
                if self.return_logits:
                    return sub_x, sub_y, sub_logits
                return sub_x, sub_y

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        # try:
        if self.params.data == 'cifar100' and self.params.agent == 'PCR_ca':
            new_model = Reduced_ResNet18_m(100, memory_bank_size = self.params.mem_bank_size, num_subcentroids=self.params.num_subcentroids, gamma=self.params.gamma)
            new_model.embedding_dim = 160
            new_model.prototypes = nn.Parameter(torch.zeros(100, self.params.num_subcentroids, 160),
                                        requires_grad=True)
            new_model.feat_norm = nn.LayerNorm(160)
            original_state_dict = model.state_dict()
            new_model.load_state_dict(original_state_dict)
            device = next(model.parameters()).device
            new_model.to(device)

        elif self.params.data == 'mini_imagenet'and self.params.agent == 'PCR_ca':
            new_model = Reduced_ResNet18_m(100, memory_bank_size = self.params.mem_bank_size, num_subcentroids=self.params.num_subcentroids, gamma=self.params.gamma)
            # model.pcrLinear = cosLinear(640, nclass)
            new_model.embedding_dim = 640
            new_model.prototypes = nn.Parameter(torch.zeros(100, self.params.num_subcentroids, 640),
                                        requires_grad=True)
            new_model.feat_norm = nn.LayerNorm(640)
            original_state_dict = model.state_dict()
            new_model.load_state_dict(original_state_dict)
            device = next(model.parameters()).device
            new_model.to(device)
        else:
            new_model = copy.deepcopy(model)
        # except RuntimeError as e:
        #     new_model = model.custom_deepcopy()
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
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
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1


class Uncertainty_retrieve(object):
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

    def retrieve(self, buffer, **kwargs):
        sub_x, sub_y = random_retrieve(buffer, self.subsample)
        if sub_x.size(0) > 0:
            # exit(0)
            with torch.no_grad():
                buffer.model.eval()

                logits, emb =  buffer.model.pcrForward(sub_x)
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
                    
                if self.ucr_max:
                    ind = entropies.sort(descending=True)[1][:self.num_retrieve]
                else:
                    ind = entropies.sort(descending=False)[1][:self.num_retrieve]

                return sub_x[ind], sub_y[ind]
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



class Hardness_retrieve(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch
        self.sub_eps_mem_batch = params.sub_eps_mem_batch

    def retrieve(self, buffer, **kwargs):
        sub_x, sub_y = random_retrieve(buffer, self.subsample)
        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        # print(sub_x.shape)
        # print
        
        
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            # print(grad_vector)
            # print(grad_vector.shape)
            # exit(0)
            with torch.no_grad():
                if self.params.agent == 'PCR_ca':
                    logits_pre, _ = buffer.model.forward(sub_x, sub_y)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post, _ = model_temp.forward(sub_x, sub_y)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                elif self.params.agent == 'PCR':
                    logits_pre, _ = buffer.model.pcrForward(sub_x)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post, _ = model_temp.pcrForward(sub_x)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                else:
                    logits_pre = buffer.model.forward(sub_x)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post = model_temp.forward(sub_x)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                # exit(0)
                # pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                # post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                # scores = post_loss - pre_loss
                flip_abs = torch.abs(logits_pre - logits_post)
                flip_sum = torch.sum(flip_abs, dim=1)
                big_ind = flip_sum.sort(descending=True)[1][:self.num_retrieve]
            return sub_x[big_ind], sub_y[big_ind]
        else:
            return sub_x, sub_y

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



class MIX_retrieve(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = params
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch
        self.sub_eps_mem_batch = params.sub_eps_mem_batch

    def retrieve(self, buffer, **kwargs):
        sub_x, sub_y = random_retrieve(buffer, self.subsample)
        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        # print(sub_x.shape)
        # print
        
        
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            # print(grad_vector)
            # print(grad_vector.shape)
            # exit(0)
            with torch.no_grad():
                if self.params.agent == 'PCR_ca':
                    logits_pre, _ = buffer.model.forward(sub_x, sub_y)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post, _ = model_temp.forward(sub_x, sub_y)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                elif self.params.agent == 'PCR':
                    logits_pre, _ = buffer.model.pcrForward(sub_x)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post, _ = model_temp.pcrForward(sub_x)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                else:
                    logits_pre = buffer.model.forward(sub_x)
                    if isinstance(logits_pre, tuple):
                        logits_pre = logits_pre[1] 
                    logits_post = model_temp.forward(sub_x)
                    if isinstance(logits_post, tuple):
                        logits_post = logits_post[1]
                # exit(0)
                pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                scores = post_loss - pre_loss
                if self.params.uniform_sampling:
                    indices = torch.randperm(len(scores))[:self.num_retrieve]
                    if self.params.buffer_analyze:
                        buffer_x = buffer.buffer_img
                        buffer_y = buffer.buffer_label
                        return sub_x[indices], sub_y[indices], buffer_x, buffer_y
                    return sub_x[indices], sub_y[indices]
                big_ind = scores.sort(descending=True)[1][:self.num_retrieve]
                small_ind = scores.sort(descending=False)[1][:self.sub_eps_mem_batch]
                output_x = torch.cat([sub_x[big_ind], sub_x[small_ind]])
                output_y = torch.cat([sub_y[big_ind], sub_y[small_ind]])

            if self.params.buffer_analyze:
                buffer_x = buffer.buffer_img
                buffer_y = buffer.buffer_label
                return output_x, output_y, buffer_x, buffer_y
            else:
                return output_x, output_y
        else:
            if self.params.buffer_analyze:
                buffer_x = buffer.buffer_img
                buffer_y = buffer.buffer_label
                return sub_x, sub_y, buffer_x, buffer_y
            else:
                return sub_x, sub_y

            return sub_x, sub_y

# flip_abs = torch.abs(logits_pre - logits_post)
# flip_sum = torch.sum(flip_abs, dim=1)
    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        # try:
        if self.params.data == 'cifar100' and self.params.agent == 'PCR_ca':
            new_model = Reduced_ResNet18_m(100, memory_bank_size = self.params.mem_bank_size, num_subcentroids=self.params.num_subcentroids, gamma=self.params.gamma)
            new_model.embedding_dim = 160
            new_model.prototypes = nn.Parameter(torch.zeros(100, self.params.num_subcentroids, 160),
                                        requires_grad=True)
            new_model.feat_norm = nn.LayerNorm(160)
            original_state_dict = model.state_dict()
            new_model.load_state_dict(original_state_dict)
            device = next(model.parameters()).device
            new_model.to(device)

        elif self.params.data == 'mini_imagenet'and self.params.agent == 'PCR_ca':
            new_model = Reduced_ResNet18_m(100, memory_bank_size = self.params.mem_bank_size, num_subcentroids=self.params.num_subcentroids, gamma=self.params.gamma)
            # model.pcrLinear = cosLinear(640, nclass)
            new_model.embedding_dim = 640
            new_model.prototypes = nn.Parameter(torch.zeros(100, self.params.num_subcentroids, 640),
                                        requires_grad=True)
            new_model.feat_norm = nn.LayerNorm(640)
            original_state_dict = model.state_dict()
            new_model.load_state_dict(original_state_dict)
            device = next(model.parameters()).device
            new_model.to(device)
        else:
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