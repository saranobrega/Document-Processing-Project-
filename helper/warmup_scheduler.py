import torch
import torch.utils.data
import torchvision
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR



class WarmupMultiStepLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps, milestones, gamma, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(self.warmup_steps)
        else:
            return self.gamma ** (sum(step >= m for m in self.milestones))