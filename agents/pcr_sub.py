import torch
from torch.utils import data
from utils.buffer.buffer import Buffer, Second_Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss_Sub
import torch.nn as nn
import torch.nn.functional as F


class ProxyContrastiveReplay_sub(ContinualLearner):
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
        super(ProxyContrastiveReplay_sub, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.params = params
        if params.second_buffer:
            self.second_buffer = params.second_buffer
            self.buffer2 = Second_Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
    

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
                    novel_loss = 0*self.criterion(logits, batch_y_combine)

                    self.opt.zero_grad()

                    
                    if self.params.second_buffer:
                        mem_x1, mem_y1 = self.buffer.retrieve(x=batch_x, y=batch_y)
                        mem_x2, mem_y2 = self.buffer2.retrieve(x=batch_x, y=batch_y)
                        mem_x = torch.cat((mem_x1, mem_x2), dim=0)
                        mem_y = torch.cat((mem_y1, mem_y2), dim=0)
                        # exit(0)
                    else:
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

                        combined_feas = torch.cat([mem_fea, feas])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        # print("combined_feas.shape: ", combined_feas.shape) # torch.Size([40, 160])
                        # print("combined_labels.shape: ", combined_labels.shape) # torch.Size([40])
                        proxy_labels = torch.arange(0, self.model.pcrLinear_sub.prototypes.size(0)).cuda()

                        # combined_feas_aug = self.model.pcrLinear_sub.prototypes[combined_labels] # proxy
                        combined_feas_aug = self.model.pcrLinear_sub.prototypes # proxy
                        # print("combined_feas_aug.shape: ", combined_feas_aug.shape) # torch.Size([40, 4, 160])

                        contrast_feas = combined_feas_aug.reshape(-1, combined_feas_aug.shape[2])
                        contrast_labels = proxy_labels.repeat_interleave(combined_feas_aug.shape[1])
                        # print("contrast_feas.shape: ", contrast_feas.shape) # torch.Size([160, 160])
                        # print("contrast_labek.shape: ", contrast_label.shape) # torch.Size([160])

                        # combined_feas_aug = torch.mean(self.model.prototypes, dim=1)[combined_labels] # proxy

                        combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                        combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

                        contrast_feas_norm = torch.norm(contrast_feas, p=2, dim=1).unsqueeze(1).expand_as(
                            contrast_feas)
                        contrast_feas_normalized = contrast_feas.div(contrast_feas_norm + 0.000001)
                        # print("combined_feas_normalized.shape: ", combined_feas_normalized.shape) # torch.Size([40, 160])
                        # print("combined_feas_aug_normalized.shape: ", combined_feas_aug_normalized.shape) # torch.Size([40, 160])
                        # cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                        #                         combined_feas_aug_normalized.unsqueeze(1)],
                        #                         dim=1)
                        # print("cos_features.shape: ", cos_features.shape) # torch.Size([40, 2, 160])
                        # exit(0)
                        # PSC = SupConLoss_Sub(temperature=0.09)
                        
                        # novel_loss += PSC(combined_feas_normalized, contrast_feas_normalized, combined_labels, contrast_labels)
                        
                        PSC  = ImprovedSupConLoss()
                        # PSC  = ImprovedSupConWithDynamicTripletLoss()
                        novel_loss += PSC(combined_feas_normalized, contrast_feas_normalized, combined_labels, contrast_labels)


                    novel_loss.backward()
                    self.opt.step()
                # update mem
                self.buffer.update(batch_x, batch_y)
                if self.params.second_buffer:
                    self.buffer2.update(batch_x, batch_y)

        self.after_train()


class ImprovedSupConLoss(nn.Module):
    """
    This class implements an improved version of the Supervised Contrastive Learning Loss,
    supporting both supervised and unsupervised modes (like SimCLR).
    """
    def __init__(self, temperature=0.07):
        """
        Initializes the loss module.

        Args:
            temperature (float): A temperature scaling factor to adjust the sharpness of the similarity distribution.
        """
        super(ImprovedSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, contrast_feature, labels=None, contrast_labels=None):
        """
        Compute the loss.

        Args:
            features (torch.Tensor): The feature representations of shape [bsz, feature_dim].
            contrast_feature (torch.Tensor): The feature representations of contrastive samples [contrast_bsz, feature_dim].
            labels (torch.Tensor, optional): The labels for each feature in the batch [bsz].
            contrast_labels (torch.Tensor, optional): The labels for each contrast feature in the batch [contrast_bsz].

        Returns:
            torch.Tensor: The computed loss.
        """
        device = features.device

        batch_size = features.shape[0]

        # Validate the input shapes and configure labels and mask
        if labels is not None and contrast_labels is not None:
            labels = labels.contiguous().view(-1, 1)
            contrast_labels = contrast_labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, contrast_labels.T).float().to(device)
        else:
            # If no labels are provided, default to a mask that treats all pairs as positive
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        # Normalize the feature vectors to ensure a consistent scale
        anchor_feature = F.normalize(features, p=2, dim=1)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        # Compute the similarity between all feature pairs, scaled by temperature
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # Apply log-sum-exp trick for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Compute the exponential of the logits and their sum
        exp_logits = torch.exp(logits)

        # Compute log_prob and mean log-likelihood for positive pairs
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Compute the final contrastive loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class ImprovedSupConWithDynamicTripletLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=1.0, alpha=1.0, beta=0.5):
        super(ImprovedSupConWithDynamicTripletLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.supcon_loss = ImprovedSupConLoss(temperature)

    def forward(self, features, contrast_features, labels=None, contrast_labels=None):
        # Calculate the Supervised Contrastive Loss
        supcon_loss = self.supcon_loss(features, contrast_features, labels, contrast_labels)

        # Calculate the Dynamic Triplet Loss
        triplet_loss = self.dynamic_triplet_loss(features, contrast_features, labels, contrast_labels)

        # Combine the losses
        hybrid_loss = self.alpha * supcon_loss + self.beta * triplet_loss

        return hybrid_loss

    def dynamic_triplet_loss(self, features, contrast_features, labels, contrast_labels):
        """
        Compute triplet loss dynamically based on labels.
        """
        # Ensuring the input features are normalized
        features_norm = F.normalize(features, p=2, dim=1)
        contrast_norm = F.normalize(contrast_features, p=2, dim=1)

        # Calculate pairwise distance between features and contrast features
        distance_matrix = torch.cdist(features_norm, contrast_norm, p=2)

        # Initialize triplet loss
        triplet_loss = torch.tensor(0.0, device=features.device)

        # Calculate triplet loss for each feature as an anchor
        for i in range(features.shape[0]):
            # Select the positive and negative samples for the anchor
            anchor_label = labels[i]
            positive_mask = (contrast_labels == anchor_label).float()
            negative_mask = (contrast_labels != anchor_label).float()

            # Avoid including the anchor as positive
            positive_mask[i] = 0

            # Calculate the hardest positive and hardest negative
            hardest_positive = (distance_matrix[i] * positive_mask).max()
            hardest_negative = (distance_matrix[i] + (1 - negative_mask) * 1e6).min()

            # Calculate triplet loss for this anchor
            triplet_loss += F.relu(hardest_positive - hardest_negative + self.margin)

        # Average the triplet loss over all anchors
        triplet_loss /= features.shape[0]

        return triplet_loss

# Dummy data
# features = torch.randn(40, 160)
# labels = torch.randint(0, 10, (40,))
# contrast_features = torch.randn(80, 160)
# contrast_labels = torch.randint(0, 10, (80,))

# # Loss
# loss_fn = SupervisedContrastiveLoss()
# loss = loss_fn(features, labels, contrast_features, contrast_labels)
# print("Supervised Contrastive Loss:", loss.item())









