import torch
from torch.distributions.categorical import Categorical

def mask_tensor(categorical: Categorical, mask: torch.Tensor) -> Categorical:
    logits = categorical.logits
    masked_logits = logits + torch.log(mask.clip(min=1e-8))

    return Categorical(logits=masked_logits)
