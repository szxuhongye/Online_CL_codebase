"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # print("features.shape: ", features.shape) # torch.Size([40, 2, 160])
        # exit(0)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # print(labels)
            # exit(0)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # print(features.shape)# torch.Size([40, 2, 640])
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_feature.shape) # torch.Size([80, 640])
        # exit(0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'proxy':
            anchor_feature = features[:, 0]
            contrast_feature = features[:, 1]
            anchor_count = 1
            contrast_count = 1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # print("anchor_feature.shape: ", anchor_feature.shape) # torch.Size([40, 640])
        # print("contrast_feature.shape: ", contrast_feature.shape)
        # exit(0)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # compute log_prob
        if self.contrast_mode == 'proxy':
            exp_logits = torch.exp(logits)
        else:
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(mask.shape)
        # print(log_prob.shape)
        # exit(0)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    


class SupConLoss_Sub(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07):
        super(SupConLoss_Sub, self).__init__()
        self.temperature = temperature

    def forward(self, features, contrast_feature, labels=None, contrast_labels=None, contrast_count=1):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # print("features.shape: ", features.shape) # torch.Size([40, 2, 160])
        # exit(0)

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # if labels is not None and mask is not None:
        #     raise ValueError('Cannot define both `labels` and `mask`')
        # elif labels is None and mask is None:
        #     mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        contrast_labels = contrast_labels.contiguous().view(-1, 1)
            # print(labels)
            # exit(0)
        mask = torch.eq(labels, contrast_labels.T).float().to(device)
        # else:
        #     mask = mask.float().to(device)
        # print(features.shape)# torch.Size([40, 2, 640])
        # anchor_count = 1
        # contrast_count = contrast_count
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_feature.shape) # torch.Size([80, 640])
        # exit(0)
        anchor_feature = features
        contrast_feature = contrast_feature
        


        # compute logits
        # print("anchor_feature.shape: ", anchor_feature.shape) # torch.Size([40, 640])
        # print("contrast_feature.shape: ", contrast_feature.shape)
        # exit(0)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)

        # compute log_prob
        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(mask.shape)
        # print(log_prob.shape)
        # exit(0)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss
