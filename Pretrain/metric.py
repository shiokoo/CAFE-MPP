import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NTXent(nn.Module):
    """
    Calculate the NTXent loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar samples to be close
        and those of different samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    Params:
        temperature: logits are divided by temperature before calculating the cross entropy.
                     the larger the temperature, the higher the penalty for difficult negative samples.
                     reference range: 0.01~0.1
        reduction: method for output
        negative_mode: determines how the (optional) negative_keys are handled.
                       'paired': each query sample is paired with a number of negative keys.
                       'unpaired': the set of negative keys are all unrelated to any positive keys.
    Input shape:
        query: (N,D) Tensor with query samples.(embeddings of the input)
        positive_key: (N,D) Tensor with positive samples.(embeddings of positive samples)
        negative_keys: Tensor with negative samples.(embeddings of other inputs)
            negative_mode = 'paired' >>> negative_keys: (N, M, D)
            negative_mode = 'unpaired' >>> negative_keys: (M, D)
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
        the NTXent loss
    """
    def __init__(self, device, temperature=0.1, alpha=0.3, reduction='mean', negative_mode='unpaired'):
        super(NTXent, self).__init__()
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return calculate(device=self.device,
                         query=query,
                         positive_key=positive_key,
                         negative_keys=negative_keys,
                         temperature=self.temperature,
                         alpha=self.alpha,
                         reduction=self.reduction,
                         negative_mode=self.negative_mode)

def calculate(device, query, positive_key, negative_keys=None, temperature=0.1, alpha=1.0, reduction='mean',
              negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<egative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")
    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")
    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')
    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        # [N,1]
        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)
        elif negative_mode == 'paired':
            N = query.size(0)
            diag = np.eye(N, N)
            correlated_mask = ~torch.from_numpy(diag).type(torch.bool)
            correlated_mask = correlated_mask.to(device)
            similarity_matrix_ = query @ query.T
            # other samples in the batch as negative samples
            negative_ = similarity_matrix_[correlated_mask].view(N, -1)
            negative_ = negative_ * alpha  
            query = query.unsqueeze(1)
            # negative samples constructed artificially
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

            negative_logits = torch.cat([negative_logits, negative_], dim=1)
        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.
        # Each sample is compared with the other samples in the batch.
        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2,-1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
